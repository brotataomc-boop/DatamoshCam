import Metal
import MetalKit
import MetalPerformanceShaders
import CoreVideo
import simd

/// The primary, real-time "logical codec engine."
///
/// Every live camera frame that arrives while recording is treated the way
/// a hardware decoder treats a P-frame: motion vectors are estimated
/// against the previous LIVE frame, then those vectors + residual are
/// applied on top of `canvas` — a texture that is deliberately never
/// refreshed with a genuine "I-frame" at a clip boundary. That single
/// substitution (apply clip B's motion to clip A's frozen pixels) is the
/// entire datamosh effect.
///
/// This is the recommended real-time path. See `BitstreamMosher.swift` for
/// the literal H.264 NAL-stripping alternative and why it isn't the default.
final class MetalMoshRenderer: NSObject, MTKViewDelegate {

    // MARK: Tunables

    private let downsampleFactor = 4   // motion estimation runs at 1/4 resolution
    private let lowResBlockSize = 8     // low-res pixels -> 32px full-res blocks
    private let searchRadius = 6        // +/- low-res pixels searched

    /// 0...2. Exposed for a "corruption amount" UI slider: near 0 keeps the
    /// frozen ghost almost static; around 1 is the classic datamosh smear;
    /// pushed higher, new detail bleeds through more aggressively.
    var residualGain: Float = 1.0

    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache!

    private let downsamplePSO: MTLComputePipelineState
    private let motionSearchPSO: MTLComputePipelineState
    private let compositePSO: MTLComputePipelineState

    // Full-resolution, ping-ponged so we never read and write the same
    // texture in the same pass.
    private var canvasTextures: [MTLTexture] = []
    private var canvasIndex = 0
    private var prevLiveTexture: MTLTexture?

    // Low-resolution scratch textures for motion estimation.
    private var curLumaTexture: MTLTexture?
    private var prevLumaTexture: MTLTexture?
    private var motionVectorTexture: MTLTexture?

    private var hasBaseline = false

    // What the MTKView actually presents. Guarded because it's written from
    // the camera-processing thread (via the command buffer's completion
    // handler) and read from the MTKView's own display-link thread.
    private let displayLock = NSLock()
    private var displayTexture: MTLTexture?

    init(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not create Metal command queue")
        }
        self.commandQueue = queue

        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not load the default Metal library — make sure MoshShaders.metal is a member of the app target")
        }
        func makePSO(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                fatalError("Missing Metal function '\(name)'")
            }
            do {
                return try device.makeComputePipelineState(function: fn)
            } catch {
                fatalError("Could not build pipeline state for '\(name)': \(error)")
            }
        }
        self.downsamplePSO = makePSO("downsampleLuma")
        self.motionSearchPSO = makePSO("blockMotionSearch")
        self.compositePSO = makePSO("motionCompensateComposite")

        super.init()

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    // MARK: - Public control

    /// Call when a new clip starts (i.e. the shutter is pressed again).
    ///
    /// Deliberately does NOT touch `canvasTextures` unless this is the very
    /// first clip of the session — leaving canvas alone is the moral
    /// equivalent of "the incoming I-frame of clip B was stripped from the
    /// bitstream": the composite kernel has no choice but to keep
    /// referencing clip A's last reconstructed picture.
    func beginNewClip(isFirstClipEver: Bool) {
        prevLiveTexture = nil
        if isFirstClipEver {
            hasBaseline = false
        }
    }

    /// Call from the camera's video data output callback. Cheap — wraps the
    /// CVPixelBuffer with zero copy and kicks off a small, fixed-size GPU
    /// pipeline. Safe to call indefinitely at 30 FPS without growing memory,
    /// since every intermediate texture is a reused, pre-sized allocation.
    func enqueue(pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        // Apple recommends periodically flushing the texture cache so old,
        // no-longer-referenced CVMetalTexture wrappers actually get
        // released. Skipping this is a common, subtle source of steadily
        // growing memory in camera + Metal pipelines.
        CVMetalTextureCacheFlush(textureCache, 0)

        guard let current = makeTexture(from: pixelBuffer) else { return }
        ensureScratchTextures(matching: current)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        if !hasBaseline {
            // The one true "I-frame" the whole session ever gets: the first
            // frame of the first-ever clip. Every clip boundary after this
            // only ever contributes motion + residual.
            copy(current, into: canvasTextures[canvasIndex], using: commandBuffer)
            prevLiveTexture = current
            hasBaseline = true
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.setDisplayTexture(current)
            }
            commandBuffer.commit()
            return
        }

        guard let prevLive = prevLiveTexture else {
            // First frame of a NEW (non-initial) clip: there's no motion to
            // measure yet, so the vector is implicitly zero and the output
            // stays exactly the frozen canvas. Smearing begins organically
            // the instant the very next frame introduces real motion.
            let outCanvas = canvasTextures[1 - canvasIndex]
            copy(canvasTextures[canvasIndex], into: outCanvas, using: commandBuffer)
            canvasIndex = 1 - canvasIndex
            prevLiveTexture = current
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.setDisplayTexture(outCanvas)
            }
            commandBuffer.commit()
            return
        }

        runMoshPipeline(current: current, prevLive: prevLive, commandBuffer: commandBuffer)
        prevLiveTexture = current
    }

    // MARK: - GPU pipeline

    private func runMoshPipeline(current: MTLTexture, prevLive: MTLTexture, commandBuffer: MTLCommandBuffer) {
        guard
            let curLuma = curLumaTexture,
            let prevLuma = prevLumaTexture,
            let mv = motionVectorTexture
        else { return }

        let canvasIn = canvasTextures[canvasIndex]
        let canvasOut = canvasTextures[1 - canvasIndex]

        // Pass 1a: downsample + luma, current frame.
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "DownsampleCurrent"
            enc.setComputePipelineState(downsamplePSO)
            enc.setTexture(current, index: 0)
            enc.setTexture(curLuma, index: 1)
            dispatchThreads(enc, downsamplePSO, size: curLuma.size2D)
            enc.endEncoding()
        }

        // Pass 1b: downsample + luma, previous live frame.
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "DownsamplePrevious"
            enc.setComputePipelineState(downsamplePSO)
            enc.setTexture(prevLive, index: 0)
            enc.setTexture(prevLuma, index: 1)
            dispatchThreads(enc, downsamplePSO, size: prevLuma.size2D)
            enc.endEncoding()
        }

        // Pass 2: block motion search, current vs. previous live frame.
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "MotionSearch"
            var params = MotionSearchParams(
                lowResSize: SIMD2<UInt32>(UInt32(curLuma.width), UInt32(curLuma.height)),
                searchRadius: Int32(searchRadius),
                blockSize: Int32(lowResBlockSize)
            )
            enc.setComputePipelineState(motionSearchPSO)
            enc.setTexture(curLuma, index: 0)
            enc.setTexture(prevLuma, index: 1)
            enc.setTexture(mv, index: 2)
            enc.setBytes(&params, length: MemoryLayout<MotionSearchParams>.stride, index: 0)
            let blocksX = (curLuma.width + lowResBlockSize - 1) / lowResBlockSize
            let blocksY = (curLuma.height + lowResBlockSize - 1) / lowResBlockSize
            dispatchThreads(enc, motionSearchPSO, size: MTLSize(width: blocksX, height: blocksY, depth: 1))
            enc.endEncoding()
        }

        // Pass 3: motion-compensate the CANVAS (not clip B's own history)
        // with those vectors, add clip B's residual, write the new canvas.
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "CompositeMosh"
            var params = CompositeParams(
                fullResSize: SIMD2<UInt32>(UInt32(current.width), UInt32(current.height)),
                lowResSize: SIMD2<UInt32>(UInt32(curLuma.width), UInt32(curLuma.height)),
                blockSize: Int32(lowResBlockSize),
                fullResPerBlock: Int32(downsampleFactor * lowResBlockSize),
                residualGain: residualGain
            )
            enc.setComputePipelineState(compositePSO)
            enc.setTexture(current, index: 0)
            enc.setTexture(prevLive, index: 1)
            enc.setTexture(canvasIn, index: 2)
            enc.setTexture(mv, index: 3)
            enc.setTexture(canvasOut, index: 4)
            enc.setBytes(&params, length: MemoryLayout<CompositeParams>.stride, index: 0)
            dispatchThreads(enc, compositePSO, size: MTLSize(width: current.width, height: current.height, depth: 1))
            enc.endEncoding()
        }

        canvasIndex = 1 - canvasIndex
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.setDisplayTexture(canvasOut)
        }
        commandBuffer.commit()
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard
            let drawable = view.currentDrawable,
            let texture = currentDisplayTexture(),
            let commandBuffer = commandQueue.makeCommandBuffer()
        else { return }

        if texture.width == drawable.texture.width, texture.height == drawable.texture.height {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(from: texture, to: drawable.texture)
            blit.endEncoding()
        } else {
            let scale = MPSImageBilinearScale(device: device)
            scale.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: drawable.texture)
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Helpers

    private func setDisplayTexture(_ texture: MTLTexture) {
        displayLock.lock()
        displayTexture = texture
        displayLock.unlock()
    }

    private func currentDisplayTexture() -> MTLTexture? {
        displayLock.lock()
        defer { displayLock.unlock() }
        return displayTexture
    }

    private func makeTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .bgra8Unorm, width, height, 0, &cvTexture
        )
        guard status == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
            return nil
        }
        return texture
    }

    private func ensureScratchTextures(matching current: MTLTexture) {
        guard canvasTextures.isEmpty || canvasTextures[0].width != current.width || canvasTextures[0].height != current.height else {
            return
        }

        func makeTex(_ w: Int, _ h: Int, _ format: MTLPixelFormat) -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(1, w), height: max(1, h), mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            guard let tex = device.makeTexture(descriptor: desc) else {
                fatalError("Could not allocate a \(w)x\(h) scratch texture")
            }
            return tex
        }

        canvasTextures = [
            makeTex(current.width, current.height, .bgra8Unorm),
            makeTex(current.width, current.height, .bgra8Unorm)
        ]
        canvasIndex = 0
        hasBaseline = false
        prevLiveTexture = nil

        let lowW = current.width / downsampleFactor
        let lowH = current.height / downsampleFactor
        curLumaTexture = makeTex(lowW, lowH, .r16Float)
        prevLumaTexture = makeTex(lowW, lowH, .r16Float)

        let blocksX = (lowW + lowResBlockSize - 1) / lowResBlockSize
        let blocksY = (lowH + lowResBlockSize - 1) / lowResBlockSize
        motionVectorTexture = makeTex(blocksX, blocksY, .rg16Float)
    }

    private func copy(_ src: MTLTexture, into dst: MTLTexture, using commandBuffer: MTLCommandBuffer) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, to: dst)
        blit.endEncoding()
    }

    private func dispatchThreads(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, size: MTLSize) {
        let w = pso.threadExecutionWidth
        let h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let groups = MTLSize(
            width: (size.width + w - 1) / w,
            height: (size.height + h - 1) / h,
            depth: 1
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}

private extension MTLTexture {
    var size2D: MTLSize { MTLSize(width: width, height: height, depth: 1) }
}

/// Must mirror `MotionSearchParams` in MoshShaders.metal exactly (field
/// order and types) — this is copied byte-for-byte via `setBytes`.
private struct MotionSearchParams {
    var lowResSize: SIMD2<UInt32>
    var searchRadius: Int32
    var blockSize: Int32
}

/// Must mirror `CompositeParams` in MoshShaders.metal exactly.
private struct CompositeParams {
    var fullResSize: SIMD2<UInt32>
    var lowResSize: SIMD2<UInt32>
    var blockSize: Int32
    var fullResPerBlock: Int32
    var residualGain: Float
}

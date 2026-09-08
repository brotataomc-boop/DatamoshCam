import VideoToolbox
import CoreMedia
import AVFoundation

/// EXPERIMENTAL — literal H.264 bitstream corruption, the same technique
/// used by desktop datamoshing workflows (deleting the leading IDR NAL unit
/// of a clip before it gets decoded onto another clip's reference picture).
///
/// Included because it's the technically "authentic" way to produce the
/// effect, and it's worth understanding even if you ship with the GPU
/// engine (`MetalMoshRenderer`) as your primary real-time path. Two things
/// make it a poor default for a live, on-device viewfinder:
///
/// 1. iOS's hardware H.264 decoder is not documented or guaranteed to
///    "gracefully" apply P-frame deltas to a stale reference picture the
///    way desktop software decoders (e.g. ffmpeg's libavcodec) do.
///    Behavior can vary by SoC/OS version: some frames decode with exactly
///    the classic smear, others may be dropped, and a decode error can
///    force tearing down and rebuilding the whole VTDecompressionSession —
///    which itself reintroduces a clean reference frame, undoing the
///    effect you wanted.
/// 2. It requires a full encode -> corrupt -> decode round trip in real
///    time on top of camera capture, which costs meaningfully more
///    CPU/GPU/thermal budget than the direct GPU simulation.
///
/// Treat this as a toggleable "authentic mode" for advanced users on
/// capable hardware, with `MetalMoshRenderer` as the reliable fallback.
/// Some exact VideoToolbox call signatures below may need small
/// adjustments depending on your SDK version — this is genuinely the area
/// of iOS multimedia development most prone to that kind of drift.
final class BitstreamMosher {

    enum MosherError: Error { case sessionCreationFailed }

    private var compressionSession: VTCompressionSession?
    private var decompressionSession: VTDecompressionSession?
    private var sharedFormatDescription: CMFormatDescription?

    /// True only for the very first frame of the very first clip in the
    /// whole session — the one genuine IDR the decoder ever receives.
    private var isFirstFrameEver = true

    var onDecodedFrame: ((CVPixelBuffer, CMTime) -> Void)?

    // MARK: - Encoder lifecycle (one fresh session per clip)

    /// A fresh VTCompressionSession per clip mirrors two independently
    /// encoded source clips in the classic desktop workflow: clip B's
    /// encoder has no idea clip A ever existed, so its first frame is a
    /// real IDR and its motion vectors are relative to its OWN history.
    func startNewClip(width: Int32, height: Int32) throws {
        if let existing = compressionSession {
            VTCompressionSessionCompleteFrames(existing, untilPresentationTimeStamp: .invalid)
            VTCompressionSessionInvalidate(existing)
        }

        var session: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: width, height: height,
            codecType: kCMVideoCodecType_H264,
            encoderSpecification: nil,
            imageBufferAttributes: nil,
            compressedDataAllocator: nil,
            outputCallback: nil, // using the block-based encode API below instead
            refcon: nil,
            compressionSessionOut: &session
        )
        guard status == noErr, let session else { throw MosherError.sessionCreationFailed }

        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_H264_Baseline_AutoLevel)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
        // Effectively "never insert an automatic keyframe" — the only IDR
        // each clip's own session will emit is its first frame.
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, value: NSNumber(value: Int32.max))

        VTCompressionSessionPrepareToEncodeFrames(session)
        compressionSession = session
    }

    /// Feeds one live frame to the CURRENT clip's own encoder. `completion`
    /// receives the raw AVCC NAL payload for that frame, whether it was a
    /// keyframe, and the format description needed to build a decodable
    /// sample buffer later.
    func encode(
        pixelBuffer: CVPixelBuffer,
        presentationTime: CMTime,
        completion: @escaping (Data, _ isKeyframe: Bool, CMFormatDescription) -> Void
    ) {
        guard let session = compressionSession else { return }
        let forceKeyframe = isFirstFrameEver
        let frameProperties: CFDictionary = [
            kVTEncodeFrameOptionKey_ForceKeyFrame: forceKeyframe
        ] as CFDictionary

        VTCompressionSessionEncodeFrame(
            session,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTime,
            duration: .invalid,
            frameProperties: frameProperties,
            infoFlagsOut: nil
        ) { [weak self] status, _, sampleBuffer in
            guard status == noErr, let sampleBuffer, let self else { return }
            guard
                let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer),
                let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer)
            else { return }

            var length = 0
            var dataPointer: UnsafeMutablePointer<Int8>?
            CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)
            guard let dataPointer else { return }
            let data = Data(bytes: dataPointer, count: length)

            let isKeyframe = AVCCNALUnits.containsIDR(avccData: data)
            self.isFirstFrameEver = false
            completion(data, isKeyframe, formatDescription)
        }
    }

    // MARK: - Decoding a (possibly corrupted) NAL payload

    /// Creates the shared decoder once, from clip A's format description,
    /// and never tears it down at a clip boundary — its internal reference
    /// picture buffer is exactly what we're deliberately leaving stale.
    func ensureDecoder(formatDescription: CMFormatDescription) {
        guard decompressionSession == nil else { return }
        sharedFormatDescription = formatDescription

        var callback = VTDecompressionOutputCallbackRecord(
            decompressionOutputCallback: decompressionCallback,
            decompressionOutputRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        var session: VTDecompressionSession?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: formatDescription,
            decoderSpecification: nil,
            imageBufferAttributes: attrs as CFDictionary,
            outputCallback: &callback,
            decompressionSessionOut: &session
        )
        decompressionSession = session
    }

    /// Strips the leading IDR/SPS/PPS NAL units from `avccData` when
    /// `stripKeyframe` is true (i.e. every clip after the very first), then
    /// hands whatever remains to the same long-lived decompression session.
    /// Because that session is never torn down at the clip boundary, its
    /// reference picture is still clip A's last decoded frame — so a
    /// stripped clip-B P-frame decodes as clip B's motion applied on top of
    /// clip A's pixels.
    func decode(avccData: Data, presentationTime: CMTime, stripKeyframe: Bool) {
        guard let decompressionSession, let formatDescription = sharedFormatDescription else { return }

        let payload = stripKeyframe ? AVCCNALUnits.removingIDRAndParameterSets(from: avccData) : avccData
        guard !payload.isEmpty else { return } // e.g. clip B's very first frame was pure IDR — nothing left to decode

        var blockBuffer: CMBlockBuffer?
        var status = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil, // let CMBlockBuffer allocate and own its storage
            blockLength: payload.count,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: payload.count,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        guard status == kCMBlockBufferNoErr, let blockBuffer else { return }

        status = payload.withUnsafeBytes { raw in
            CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: blockBuffer, offsetIntoDestination: 0, dataLength: payload.count)
        }
        guard status == kCMBlockBufferNoErr else { return }

        var sampleBuffer: CMSampleBuffer?
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: presentationTime, decodeTimeStamp: .invalid)
        var sampleSize = payload.count
        CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault, dataBuffer: blockBuffer,
            formatDescription: formatDescription, sampleCount: 1,
            sampleTimingEntryCount: 1, sampleTimingArray: &timing,
            sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize,
            sampleBufferOut: &sampleBuffer
        )
        guard let sampleBuffer else { return }

        // Synchronous decode keeps this demo simple to reason about; pass
        // `.enableAsynchronousDecompression` in `flags` if you need it off
        // the calling thread.
        let flags = VTDecodeFrameFlags()
        VTDecompressionSessionDecodeFrame(decompressionSession, sampleBuffer: sampleBuffer, flags: flags, frameRefcon: nil, infoFlagsOut: nil)
    }
}

private func decompressionCallback(
    refCon: UnsafeMutableRawPointer?,
    sourceFrameRefCon: UnsafeMutableRawPointer?,
    status: OSStatus,
    infoFlags: VTDecodeInfoFlags,
    imageBuffer: CVImageBuffer?,
    presentationTimeStamp: CMTime,
    presentationDuration: CMTime
) {
    guard status == noErr, let imageBuffer, let refCon else { return }
    let mosher = Unmanaged<BitstreamMosher>.fromOpaque(refCon).takeUnretainedValue()
    mosher.onDecodedFrame?(imageBuffer, presentationTimeStamp)
}

/// Minimal AVCC (4-byte length-prefixed) NAL unit helpers for H.264.
enum AVCCNALUnits {
    static func containsIDR(avccData: Data) -> Bool {
        forEachNALUnit(in: avccData) { type, _ in type == 5 } != nil
    }

    /// Returns a copy of `avccData` with any NAL unit of type 5 (IDR
    /// slice), 7 (SPS), or 8 (PPS) removed, preserving the 4-byte length
    /// prefixes of everything that remains.
    static func removingIDRAndParameterSets(from avccData: Data) -> Data {
        var result = Data()
        _ = forEachNALUnit(in: avccData) { type, range in
            if type != 5 && type != 7 && type != 8 {
                var length = UInt32(range.count).bigEndian
                result.append(Data(bytes: &length, count: 4))
                result.append(avccData.subdata(in: range))
            }
            return false // keep scanning
        }
        return result
    }

    /// Walks 4-byte-length-prefixed NAL units, calling `body(type, range)`
    /// for each. `range` covers just the NAL payload, not its length
    /// prefix. Returns `true` and stops early the first time `body`
    /// returns `true`; returns `nil` if the scan completes without that.
    @discardableResult
    private static func forEachNALUnit(in data: Data, body: (UInt8, Range<Data.Index>) -> Bool) -> Bool? {
        var offset = data.startIndex
        while offset + 4 <= data.endIndex {
            let lengthBytes = data.subdata(in: offset..<offset + 4)
            let length = Int(lengthBytes.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
            let nalStart = offset + 4
            guard length > 0, nalStart + length <= data.endIndex else { break }
            let nalType = data[nalStart] & 0x1F
            if body(nalType, nalStart..<(nalStart + length)) { return true }
            offset = nalStart + length
        }
        return nil
    }
}

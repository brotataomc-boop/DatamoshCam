import CoreImage
import UIKit
import Combine
import Metal

/// Central state machine tying the camera to the Metal mosh renderer.
///
/// `@Published` properties are only ever mutated on the main thread (see
/// `setPhase` and `freezeGhost` below), so this plays nicely as an
/// `ObservableObject` without requiring the whole type to be actor-isolated.
/// This reference project targets Swift 5 concurrency mode; adopting full
/// Swift 6 strict checking is a worthwhile follow-up but a bigger lift than
/// this architecture needs to demonstrate.
final class DatamoshEngine: ObservableObject {

    enum Phase: Equatable {
        case idle       // showing the ghost overlay, waiting for the next hold
        case recording  // actively feeding frames through the mosh pipeline
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var ghostImage: UIImage?

    let camera = CameraManager()
    let renderer: MetalMoshRenderer

    /// A single shared CIContext, created once. Creating a new CIContext per
    /// frame is a well-known source of jank and memory churn — this
    /// instance is reused for the life of the app.
    private let ciContext: CIContext

    private var clipIndex = 0

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.ciContext = CIContext(mtlDevice: device)
        self.renderer = MetalMoshRenderer(device: device)

        // Captures `renderer` directly (not `self`) so the hot per-frame
        // path never has to touch this object at all.
        camera.onFrame = { [weak renderer] pixelBuffer, timestamp in
            renderer?.enqueue(pixelBuffer: pixelBuffer, timestamp: timestamp)
        }

        camera.onClipEnded = { [weak self] pixelBuffer in
            self?.freezeGhost(from: pixelBuffer)
        }
    }

    func start() async {
        await camera.requestAccessAndConfigure()
    }

    func beginHold() {
        guard phase == .idle else { return }
        clipIndex += 1
        // Tell the renderer this is a new clip: from now on incoming frames
        // are clip B's motion/residual data, and the canvas texture (clip
        // A's frozen pixels) is deliberately NOT reset with a fresh
        // I-frame. This is the actual "moshing" moment.
        renderer.beginNewClip(isFirstClipEver: clipIndex == 1)
        camera.startClip()
        ghostImage = nil
        setPhase(.recording)
    }

    func endHold() {
        guard phase == .recording else { return }
        camera.stopClip()
        setPhase(.idle)
    }

    private func setPhase(_ newPhase: Phase) {
        if Thread.isMainThread {
            phase = newPhase
        } else {
            DispatchQueue.main.async { [weak self] in self?.phase = newPhase }
        }
    }

    /// Called on the camera's background queue. Rendering a single frame
    /// through CIContext here — once per clip stop, never per live frame —
    /// is cheap enough to do off the main thread; only the resulting
    /// UIImage needs to hop back to publish.
    private func freezeGhost(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)
        DispatchQueue.main.async { [weak self] in
            self?.ghostImage = image
        }
    }
}

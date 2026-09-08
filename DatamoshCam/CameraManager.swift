import AVFoundation
import CoreVideo

/// Owns the AVCaptureSession and hands off every camera frame to the mosh
/// engine in real time. All frame bookkeeping (including the "last live
/// pixel buffer" used for the ghost freeze) is serialized on a single
/// queue to avoid data races — see `videoDataOutputQueue`.
final class CameraManager: NSObject, ObservableObject {

    enum ClipState: Equatable {
        case idle
        case recording
    }

    @Published private(set) var clipState: ClipState = .idle
    @Published private(set) var isAuthorized = false

    /// Fired on `videoDataOutputQueue` for every camera frame. Consumers
    /// must not retain the CVPixelBuffer beyond what they need — it comes
    /// from a small, reused pool, and holding references longer than
    /// necessary is the fastest way to stall that pool and drop frames.
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?

    /// Fired on `videoDataOutputQueue` the moment a clip stops, with the
    /// last live pixel buffer — used to freeze the ghost overlay.
    var onClipEnded: ((CVPixelBuffer) -> Void)?

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "com.datamoshcam.session")
    // autoreleaseFrequency: .workItem releases camera buffer wrappers at the
    // end of each block, not whenever the queue feels like it — important
    // for keeping memory flat during long recording sessions.
    private let videoDataOutputQueue = DispatchQueue(
        label: "com.datamoshcam.videoData",
        qos: .userInitiated,
        autoreleaseFrequency: .workItem
    )

    private let videoOutput = AVCaptureVideoDataOutput()
    private var lastPixelBuffer: CVPixelBuffer?

    func requestAccessAndConfigure() async {
        let granted = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { continuation.resume(returning: $0) }
        }
        await MainActor.run { self.isAuthorized = granted }
        guard granted else { return }

        sessionQueue.async { [weak self] in
            self?.configureSession()
            self?.session.startRunning()
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1920x1080

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            assertionFailure("Could not configure camera input")
            return
        }
        session.addInput(input)

        // Pin the frame rate to a firm 30 FPS. The mosh engine's motion
        // search assumes a consistent inter-frame interval.
        if device.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= 30 && $0.maxFrameRate >= 30 }) {
            do {
                try device.lockForConfiguration()
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
                device.unlockForConfiguration()
            } catch {
                print("Could not lock device for configuration: \(error)")
            }
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // The single most important line for memory stability: never let
        // frames queue up behind a busy Metal pipeline. Drop instead of
        // accumulate.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)

        guard session.canAddOutput(videoOutput) else {
            assertionFailure("Could not add video output")
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90 // iOS 17+, portrait
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait // pre-iOS 17 fallback
            }
        }
    }

    func startClip() {
        videoDataOutputQueue.async { [weak self] in
            DispatchQueue.main.async { self?.clipState = .recording }
        }
    }

    func stopClip() {
        videoDataOutputQueue.async { [weak self] in
            guard let self else { return }
            if let last = self.lastPixelBuffer {
                self.onClipEnded?(last)
            }
            DispatchQueue.main.async { self.clipState = .idle }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // autoreleasepool ensures the Objective-C bridged wrappers Core
        // Media hands us here get torn down promptly. Without it, a 30 FPS
        // capture queue reliably grows memory over a multi-minute session.
        autoreleasepool {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

            // Swift's CF toll-free bridging keeps this buffer alive for as
            // long as `lastPixelBuffer` holds a strong reference — no manual
            // CFRetain/CFRelease required.
            lastPixelBuffer = pixelBuffer
            onFrame?(pixelBuffer, timestamp)
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Expected under load because of alwaysDiscardsLateVideoFrames —
        // this is the pressure valve working as intended, not a bug.
    }
}

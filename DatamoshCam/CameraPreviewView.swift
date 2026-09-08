import SwiftUI
import MetalKit

/// Hosts the MTKView that displays whatever `MetalMoshRenderer` currently
/// has as its composited output. The view's own display link runs at a
/// steady 30 FPS and simply presents the latest processed texture — this
/// decouples camera capture timing from display timing, which keeps the
/// UI smooth even if a given frame's GPU work takes a little longer.
struct CameraPreviewView: UIViewRepresentable {
    let device: MTLDevice
    let renderer: MTKViewDelegate

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: device)
        view.delegate = renderer
        view.framebufferOnly = true
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 30
        view.colorPixelFormat = .bgra8Unorm
        view.contentMode = .scaleAspectFill
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

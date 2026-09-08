import SwiftUI

/// A single hold-to-record shutter. Uses a zero-distance DragGesture rather
/// than a long-press recognizer because we need clean, immediate press-down
/// and press-up edges — recording must start the instant the finger lands
/// and stop the instant it lifts, with no minimum duration.
struct ShutterButton: View {
    let isRecording: Bool
    let onPress: () -> Void
    let onRelease: () -> Void

    @State private var isPressed = false

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Color.white, lineWidth: 4)
                .frame(width: 84, height: 84)

            Circle()
                .fill(isRecording ? Color.red : Color.white)
                .frame(width: isRecording ? 40 : 68, height: isRecording ? 40 : 68)
                .animation(.spring(response: 0.25, dampingFraction: 0.6), value: isRecording)
        }
        .scaleEffect(isPressed ? 0.92 : 1.0)
        .animation(.easeOut(duration: 0.1), value: isPressed)
        .contentShape(Circle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !isPressed else { return }
                    isPressed = true
                    onPress()
                }
                .onEnded { _ in
                    isPressed = false
                    onRelease()
                }
        )
        .accessibilityLabel(isRecording ? "Recording, release to stop" : "Hold to record")
    }
}

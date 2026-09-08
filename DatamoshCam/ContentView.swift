import SwiftUI

struct ContentView: View {
    @StateObject private var engine = DatamoshEngine()

    var body: some View {
        ZStack {
            CameraPreviewView(device: engine.renderer.device, renderer: engine.renderer)
                .ignoresSafeArea()

            if let ghost = engine.ghostImage, engine.phase == .idle {
                Image(uiImage: ghost)
                    .resizable()
                    .scaledToFill()
                    .opacity(0.35)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }

            VStack {
                Spacer()
                ShutterButton(isRecording: engine.phase == .recording) {
                    engine.beginHold()
                } onRelease: {
                    engine.endHold()
                }
                .padding(.bottom, 40)
            }
        }
        .background(Color.black)
        .task {
            await engine.start()
        }
        .animation(.easeInOut(duration: 0.25), value: engine.ghostImage)
    }
}

#Preview {
    ContentView()
}

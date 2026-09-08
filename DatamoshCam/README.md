# DatamoshCam — Reference Architecture

A real-time datamoshing camera app. Two independent "codec engines" are
included:

- **`MetalMoshRenderer`** (primary, recommended) — simulates P-frame
  decoding on the GPU. Reliable, real-time, works identically across every
  supported device.
- **`BitstreamMosher`** (experimental) — literally encodes, corrupts, and
  decodes an H.264 stream. More "authentic," but hardware-decoder behavior
  under intentionally malformed input isn't guaranteed by Apple, so treat
  it as an advanced/opt-in mode rather than the default.

## File structure

```
DatamoshCam/
├── App/
│   └── DatamoshCamApp.swift        Entry point
├── Views/
│   ├── ContentView.swift           Fullscreen viewfinder + ghost + shutter
│   ├── CameraPreviewView.swift     UIViewRepresentable MTKView wrapper
│   └── ShutterButton.swift         Hold-to-record button
├── Camera/
│   └── CameraManager.swift         AVCaptureSession, 1080p30, frame delivery
├── Engine/
│   ├── DatamoshEngine.swift        State machine, ghost image publishing
│   ├── MetalMoshRenderer.swift     GPU motion-compensation codec engine
│   ├── BitstreamMosher.swift       Experimental literal NAL-stripping engine
│   └── Shaders/
│       └── MoshShaders.metal       Motion search + composite kernels
└── README.md
```

## Setup

1. Create a new iOS App project in Xcode (SwiftUI lifecycle), iOS 17+
   deployment target (for `videoRotationAngle`; drop to iOS 14+ by using
   the `videoOrientation` fallback path exclusively if you need wider reach).
2. Drag the folders above into the project, keeping the group structure.
   Make sure `MoshShaders.metal` is added to the app target's "Compile
   Sources" — Xcode does this automatically for `.metal` files but it's
   worth checking after a drag-and-drop import.
3. Add to `Info.plist`:
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>DatamoshCam needs camera access to record video.</string>
   ```
4. Run on a **physical device**. The simulator has no camera, and Metal +
   `CVMetalTextureCache` camera interop is not meaningfully testable there.

## How the effect actually works

The whole app hinges on one substitution, implemented in the third pass of
`MoshShaders.metal` (`motionCompensateComposite`):

A real H.264 decoder reconstructs a P-frame as:

```
decoded = motionCompensate(reference, motionVector) + residual
```

`residual` was computed by the *encoder* as the difference between the true
current frame and what its own motion vectors predict from *its own*
reference frame. Datamoshing happens when a decoder is hijacked into
running that formula with the right residual and vectors, but the wrong
reference — because the real keyframe that should have refreshed its
reference picture buffer never arrived.

`MetalMoshRenderer` reproduces this without ever building or corrupting a
real NAL unit:

1. On every live frame of the current clip, motion vectors are estimated
   between it and the *previous live frame of the same clip* — genuine
   motion data.
2. `residual = currentFrame − motionCompensate(previousLiveFrame, mv)` —
   the detail that motion alone doesn't explain.
3. `moshed = motionCompensate(canvas, mv) + residual`, where `canvas` is
   the frozen last frame of the *previous* clip (or an already-corrupted
   descendant of it).
4. `canvas` is then replaced by `moshed` for the next frame — so corruption
   compounds over time exactly like a real GOP with a missing keyframe.

`beginNewClip(isFirstClipEver:)` is the "I-frame stripped from the
bitstream" moment: it deliberately leaves `canvas` untouched at every clip
boundary except the very first, so the composite kernel has no choice but
to keep motion-compensating clip A's pixels with clip B's motion.

`BitstreamMosher.swift` does the same thing for real: a fresh
`VTCompressionSession` per clip (so clip B's first frame is a genuine IDR
encoded relative to clip B's own history), NAL-level stripping of that IDR
(plus its SPS/PPS) via `AVCCNALUnits`, and feeding the remainder into a
`VTDecompressionSession` that is *never* torn down across the clip
boundary — so its internal reference picture is still clip A's last
decoded frame when clip B's orphaned P-frames arrive.

## Memory & performance notes

- `alwaysDiscardsLateVideoFrames = true` on the `AVCaptureVideoDataOutput`
  is load-bearing: it's the pressure valve that prevents a slow frame from
  causing an unbounded backlog of retained `CVPixelBuffer`s.
- The video data output delegate queue uses `autoreleaseFrequency: .workItem`
  and wraps its body in `autoreleasepool { }` — both are necessary; either
  one alone still leaks under sustained 30 FPS capture.
- `CVMetalTextureCacheFlush` is called once per enqueued frame. Skipping
  this is a very common, very subtle source of slow memory growth in any
  camera + Metal pipeline.
- All GPU scratch textures (`canvasTextures`, luma buffers, motion vector
  texture) are allocated once in `ensureScratchTextures` and reused for the
  life of the session — no per-frame allocation on the hot path.
- The CIContext used for the ghost-image freeze is created exactly once
  (in `DatamoshEngine.init`) and reused; it's only ever invoked once per
  clip stop, never in the per-frame path.

### Performance tuning

`blockMotionSearch` is a naive full-search block matcher — the simplest
correct implementation, not the fastest. At the default settings (¼-res
motion estimation, 8×8 low-res blocks, ±6 px search radius) it's tuned for
clarity. If you see thermal throttling or dropped frames on older devices:

- Reduce `searchRadius` (biggest single lever — cost grows with its square).
- Increase `downsampleFactor` (e.g. 4 → 8) to shrink the search space further.
- Replace the full search with a three-step or diamond search pattern —
  a well-known ~90% reduction in candidate evaluations for a small quality
  tradeoff.

## Extending: saving a file

Everything above drives the live viewfinder. To also save a moshed video
file, allocate the canvas textures as CVPixelBuffer-backed (via a
`CVPixelBufferPool` + `CVMetalTextureCache`, the same technique used for
camera input) instead of plain `.private` Metal textures, then append each
completed `canvasOut` frame straight to an `AVAssetWriterInputPixelBufferAdaptor`
with a monotonically increasing `CMTime`. Because the canvas texture *is*
the moshed output, no extra copy or re-encode is needed — you're writing
exactly what's on screen.

## Other extension ideas

- Bind `renderer.residualGain` to a slider for a live "corruption amount" control.
- A double-tap could call `beginNewClip(isFirstClipEver: true)` to force a
  hard reset (a real, clean I-frame) when the user wants an intentional
  clean cut instead of a mosh transition.
- Haptic feedback (`UIImpactFeedbackGenerator`) on press/release.
- Front-camera toggle — note the motion search assumes consistent framing,
  so mirror the vectors appropriately if you add this.

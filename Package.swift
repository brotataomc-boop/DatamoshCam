// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DatamoshCam",
    platforms: [
        .iOS(.v17) // Locks the deployment environment to iOS 17 framework APIs
    ],
    products: [
        .executable(name: "DatamoshCamApp", targets: ["DatamoshCamTarget"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "DatamoshCamTarget",
            dependencies: [],
            path: "DatamoshCam",
            resources: [
                // Embeds your metal shader source file natively into the runtime module bundle
                .process("Engine/Shaders/MoshShaders.metal")
            ]
        )
    ]
)

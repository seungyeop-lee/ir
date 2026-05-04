{
  description = "ir - local markdown semantic search with hybrid BM25+vector retrieval and LLM reranking";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          pkgs,
          lib,
          system,
          config,
          ...
        }:
        let
          isDarwin = pkgs.stdenv.isDarwin;

          # GPU backend selection for Linux.
          # Default: CPU + OpenMP only (safe, no external GPU deps).
          # Override: nix build .#cuda / .#rocm / .#vulkan
          linuxGpuVariants = {
            cpu = {
              features = "llama-openmp";
              extraBuildInputs = [ pkgs.llvmPackages.openmp ];
              ldflags = "-lomp";
            };
            cuda = {
              # Requires nixpkgs config.allowUnfree = true and CUDA drivers on host.
              features = "llama-openmp,llama-cuda";
              extraBuildInputs = [
                pkgs.llvmPackages.openmp
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cuda_cudart
                pkgs.cudaPackages.libcublas
              ];
              ldflags = "-lomp";
            };
            rocm = {
              # Requires ROCm stack on host (/opt/rocm or ROCM_PATH set).
              features = "llama-openmp,llama-rocm";
              extraBuildInputs = [
                pkgs.llvmPackages.openmp
                pkgs.rocmPackages.clr
                pkgs.rocmPackages.rocblas
                pkgs.rocmPackages.hipblas
              ];
              ldflags = "-lomp";
            };
            vulkan = {
              features = "llama-openmp,llama-vulkan";
              extraBuildInputs = [
                pkgs.llvmPackages.openmp
                pkgs.vulkan-loader
                pkgs.vulkan-headers
              ];
              ldflags = "-lomp -lvulkan";
            };
          };

          mkPackage =
            { variant ? null }:
            let
              linux = linuxGpuVariants.${if variant != null then variant else "cpu"};
              cargoFlags = lib.optionals (!isDarwin) [
                "--no-default-features"
                "--features"
                linux.features
              ];
            in
            pkgs.rustPlatform.buildRustPackage {
              pname = "ir";
              version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;
              src = ./.;
              cargoLock.lockFile = ./Cargo.lock;

              nativeBuildInputs = [
                pkgs.cmake
                pkgs.python3
              ];

              buildInputs =
                lib.optionals isDarwin [
                  pkgs.darwin.apple_sdk.frameworks.Accelerate
                  pkgs.darwin.apple_sdk.frameworks.Foundation
                  pkgs.darwin.apple_sdk.frameworks.Metal
                ]
                ++ lib.optionals (!isDarwin) linux.extraBuildInputs;

              cargoBuildFlags = cargoFlags;
              cargoTestFlags = cargoFlags;

              # ggml is statically linked; cargo's final link step needs explicit
              # GPU library flags that cmake's detection only set in the static archive.
              env.NIX_LDFLAGS = lib.optionalString (!isDarwin) linux.ldflags;

              CMAKE_GENERATOR = "Unix Makefiles";

              meta = {
                description = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.description;
                homepage = "https://github.com/vlwkaos/ir";
                license = lib.licenses.mit;
                mainProgram = "ir";
                platforms = lib.platforms.unix;
              };
            };
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ (import inputs.rust-overlay) ];
            config = { };
          };

          packages = {
            default = mkPackage { };
            cuda = mkPackage { variant = "cuda"; };
            rocm = mkPackage { variant = "rocm"; };
            vulkan = mkPackage { variant = "vulkan"; };
          };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ config.packages.default ];
            nativeBuildInputs = [
              (pkgs.rust-bin.stable."1.95.0".default.override {
                extensions = [
                  "rust-src"
                  "rust-analyzer"
                ];
              })
            ];
          };

          formatter = pkgs.nixfmt-rfc-style;
        };
    };
}

  {
    pkgs ? import <nixpkgs> {}
  }:
  
  pkgs.mkShell {
    buildInputs = [
      pkgs.gcc
      pkgs.llvmPackages.openmp
      pkgs.cudatoolkit_11
      pkgs.onnxruntime
    ];
    
    shellHook = ''
      echo "Setting up C++ environment for ONNX Inference with CUDA..."
      echo "Environment is ready."
    '';
  }
   {
     pkgs ? import <nixpkgs> {}
   }:
   
   pkgs.mkShell {
     buildInputs = [
       pkgs.gcc
       pkgs.llvmPackages.openmp
       pkgs.onnxruntime
     ];
   
     shellHook = ''
       echo "Setting up C++ environment for ONNX Inference..."
       echo "Environment is ready."
     '';
 }
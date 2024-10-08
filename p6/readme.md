 1. Enter the development environment by running

```
export NIXPKGS_ALLOW_UNFREE=1
```

```
nix-shell
```

 2. Compile the Code
- Compile the C++ code using:

  ```sh
  g++ -fopenmp -I /nix/store/...onnxruntime/include -L /nix/store/...onnxruntime/lib -lonnxruntime -L /nix/store/...cuda/lib64 -lcudart minimal_inference.cpp -o minimal_inference
  ```

  Replace `/nix/store/...onnxruntime` and `/nix/store/...cuda` with the appropriate paths.

 3. Run the Executable
- Execute the compiled binary:

  ```sh
  ./minimal_inference
  ```

 Example Input
- The program runs inference on an example ONNX model (`example_model.onnx`) using generated input tensors.
- Adjust the input data in the code to match your model's requirements.

 Troubleshooting
- Ensure the ONNX model path is correct.
- Verify all dependencies are installed correctly in the Nix shell environment.
- Ensure the CUDA library paths are accessible during compilation and execution.

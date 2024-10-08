1. Enter the development environment by running:

```
export NIXPKGS_ALLOW_UNFREE=1
```

```
nix-shell
```


2. Compile the code using:

```
g++ -fopenmp -I /nix/store/...onnxruntime/include -L /nix/store/...onnxruntime/lib -lonnxruntime p5.cpp -o minimal_inference
```


3. Run the executable:

```
./minimal_inference
```

- Modify the input in the `RunInference` function to match the input requirements of your model.

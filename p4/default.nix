{
  pkgs ? import <nixpkgs> {}
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.numpy
    pkgs.python3Packages.onnxruntime
  ];

  shellHook = ''
    echo "Setting up Python environment..."
    python -m venv venv
    source venv/bin/activate
    pip install onnxruntime numpy
    echo "Environment is ready."
  '';
}
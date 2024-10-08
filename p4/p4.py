# Minimalist Inference Framework
# Requirements:
# - Python 3
# - ONNX Runtime (can be installed via: pip install onnxruntime)

import onnxruntime as ort
import numpy as np

class MinimalInferenceFramework:
    def __init__(self, model_path):
        """
        Initialize the inference session with the given model.
        
        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.model_path = model_path
        self.session = None
        self.load_model()

    def load_model(self):
        """
        Loads the ONNX model into an inference session.
        """
        try:
            self.session = ort.InferenceSession(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def run_inference(self, input_data):
        """
        Runs inference on the given input data.
        
        Args:
            input_data (dict): A dictionary where keys are input names and values are numpy arrays.
        
        Returns:
            dict: A dictionary with output names and the corresponding results as numpy arrays.
        """
        if self.session is None:
            raise RuntimeError("Inference session is not initialized.")
        
        input_names = [input.name for input in self.session.get_inputs()]
        prepared_input = {input_name: input_data[input_name] for input_name in input_names}
        
        output_names = [output.name for output in self.session.get_outputs()]
        result = self.session.run(output_names, prepared_input)
        
        return dict(zip(output_names, result))

# Example usage
def main():
    model_path = "example_model.onnx"  # Path to your ONNX model file
    framework = MinimalInferenceFramework(model_path)

    # Example input assuming a model that takes a single input named 'input'.
    # Adjust input name and shape accordingly for your own model.
    input_data = {
        "input": np.random.randn(1, 3, 224, 224).astype(np.float32)  # Example input tensor
    }
    
    try:
        output = framework.run_inference(input_data)
        print("Inference output:")
        for name, value in output.items():
            print(f"{name}: {value}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
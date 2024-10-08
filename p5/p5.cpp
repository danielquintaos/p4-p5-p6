// Minimalist Inference Framework with Parallelism using OpenMP
// Requirements:
// - Implemented in C++
// - Uses ONNX Runtime for inference
// - Implements CPU parallelism using OpenMP

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <stdexcept>

class MinimalInferenceFramework {
public:
    MinimalInferenceFramework(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "MinimalInferenceFramework"), session_(nullptr) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);  // Let OpenMP handle parallelism
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            std::cout << "Model loaded successfully from " << model_path << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<float> run_inference(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape) {
        if (!session_) {
            throw std::runtime_error("Inference session is not initialized.");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const char* input_name = session_->GetInputNameAllocated(0, allocator).get();
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size(), input_type);

        const char* output_name = session_->GetOutputNameAllocated(0, allocator).get();
        std::vector<const char*> output_names{ output_name };

        std::vector<Ort::Value> output_tensors;

        #pragma omp parallel shared(output_tensors)
        {
            #pragma omp single
            {
                output_tensors = session_->Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, output_names.data(), output_names.size());
            }
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        return std::vector<float>(output_data, output_data + output_size);
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
};

int main() {
    std::string model_path = "./example_model.onnx";  // Path to your ONNX model
    MinimalInferenceFramework framework(model_path);

    // Example input assuming a model that takes a single input of size (1, 3, 224, 224)
    std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);  // Example input tensor filled with 1.0
    std::vector<int64_t> input_shape = { 1, 3, 224, 224 };

    try {
        std::vector<float> output = framework.run_inference(input_data, input_shape);
        std::cout << "Inference output (first 10 elements):" << std::endl;
        for (size_t i = 0; i < std::min(output.size(), size_t(10)); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

    return 0;
}
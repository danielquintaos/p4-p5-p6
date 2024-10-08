#include "MinimalInferenceFramework.h"
#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

MinimalInferenceFramework::MinimalInferenceFramework(const std::string& model_path)
    : model_path_(model_path), env_(ORT_LOGGING_LEVEL_WARNING, "MinimalInference") {
    // Create the ONNX Runtime session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(omp_get_max_threads());  // Set max threads for parallelism
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Enable CUDA GPU execution provider
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options);
        std::cout << "Model loaded successfully from " << model_path_ << "\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load the model: " << e.what() << "\n";
    }
}

std::vector<std::vector<float>> MinimalInferenceFramework::RunInference(const std::vector<std::vector<float>>& input_data) {
    if (!session_) {
        throw std::runtime_error("Model is not loaded.");
    }

    // Prepare input/output metadata
    const size_t input_tensor_size = input_data[0].size();
    const size_t num_inputs = input_data.size();

    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session_->GetInputName(0, allocator);
    std::vector<int64_t> input_shape{1, static_cast<int64_t>(input_tensor_size)};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Prepare outputs
    const char* output_name = session_->GetOutputName(0, allocator);
    std::vector<std::vector<float>> results(num_inputs);

    // Run inference in parallel using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < num_inputs; ++i) {
        std::vector<float> input = input_data[i];
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input_tensor_size, input_shape.data(), input_shape.size());

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> output(output_data, output_data + input_tensor_size);
        results[i] = output;
    }

    return results;
}
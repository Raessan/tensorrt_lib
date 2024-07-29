#ifndef TENSORRT_ENGINE_HPP_
#define TENSORRT_ENGINE_HPP_


#include <fstream>
#include <chrono>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

#include <memory>
#include <atomic>
#include <vector>
#include <iostream>
#include <cassert>


// Utility methods
namespace Util {

    // Check if file exists
    inline bool doesFileExist(const std::string& filepath) {
        std::ifstream f(filepath.c_str());
        return f.good();
    }

    // Check if CUDA command is correct
    inline void checkCudaErrorCode(cudaError_t code) {
        if (code != 0) {
            std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
            std::cout << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
    }

    // Fcn to get files in a directory
    std::vector<std::string> getFilesInDirectory(const std::string& dirPath);
}

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
};

// Options for the network
struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // The batch size to be used when computing calibration data for INT8 inference.
    // Should be set to as large a batch number as your GPU will support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 1;
    // GPU device index
    int deviceIndex = 0;
    // DLA acceleration
    int dlaCore = 0;
    // Out file path
    std::string out_file_path = "";
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class TensorRTEngine {
public:
    TensorRTEngine(const Options& options);
    ~TensorRTEngine();
    // Build the network fron ONNX path
    bool build(std::string onnxModelPath);
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference with the input and output with dims: [n_inputs/n_outputs, batch_size, input_dims/output_dims]. input_dims is the product of all remaining dims of input. Example: if the input is an image, input_dims = C*H*W. Same for output_dim
    bool runInference(const std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>>& outputs);
    // Runs inference considering that the input is stored in a doublevector of pointers, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general [n_outputs, batch_size, output_dims]
    // The first dimension of the inputs is the number of inputs, the second is the batch size, and pointer has a valid range equal to the dimensionality of the input
    bool runInference(const std::vector<std::vector<float *>> &inputs, std::vector<std::vector<std::vector<float>>>& outputs, bool from_device);
    
    // bool runInference(const float * input, float * output, bool from_device=false);

    [[nodiscard]] const std::vector<nvinfer1::Dims>& getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims ;};
    [[nodiscard]] const std::vector<std::string>& getTensorNames() const { return m_IOTensorNames ;};

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when there is just one output, but the batch_size may be >1
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output);

    // Utility method for transforming triple nested output array into single array
    // Should be used when the output batch size is 1, and there is only a single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);

    // Function that checks the input, accepting both std::vector<std::vector<std::vector<float>>> and std::vector<std::vector< float *>> for both versions of runInference
    template<typename T>
    bool check_input(std::vector<std::vector<T>> inputs){
        // First we do some error checking
        if (inputs.empty() || inputs[0].empty()) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Provided input vector is empty!" << std::endl;
            return false;
        }

        // Ensure we have the corect number of outputs
        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Incorrect number of inputs provided!" << std::endl;
            return false;
        }

        // Ensure the batch size does not exceed the max
        if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size is larger than the model expects!" << std::endl;
            std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
            std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        // Make sure the same batch size was provided for all inputs
        for (size_t i = 1; i < inputs.size(); ++i) {
            if (inputs[i].size() != static_cast<size_t>(batchSize)) {
                std::cout << "===== Error =====" << std::endl;
                std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
                return false;
            }
        }
        return true;
    }
    
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options, const std::string& onnxModelPath);

    // Gets the device names
    void getDeviceNames(std::vector<std::string>& deviceNames);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
    std::string m_engineName;
};

#endif // TENSORRT_HPP_
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <iterator>
#include "tensorrt_cpp_lib/tensorrt_engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace Util;

// This function gets the number of files in a directory
std::vector<std::string> Util::getFilesInDirectory(const std::string& dirPath) {
    std::vector<std::string> filepaths;
    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path().string());
    }
    return filepaths;
}

// Would advise using a proper logging utility such as https://github.com/gabime/spdlog
void Logger::log(Severity severity, const char *msg) noexcept {
    
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

// Constructor to store options
TensorRTEngine::TensorRTEngine(const Options &options)
    : m_options(options) {}

// Build the engine from ONNX model path
bool TensorRTEngine::build(std::string onnxModelPath) {

    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options, onnxModelPath);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // Create the network
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    if (input0Batch == -1) {
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else if (input0Batch == 1) {
        std::cout << "Model only supports fixed batch size of 1" << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of 1. Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    } else {
        throw std::runtime_error("Implementation currently only supports dynamic batch sizes or a fixed batch size of 1 (your batch size is fixed to "
        + std::to_string(input0Batch) + ")");
    }

    // Configuration profile
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();

        nvinfer1::Dims dims_kmin, dims_kopt, dims_kmax;
        dims_kmin.nbDims = inputDims.nbDims; dims_kopt.nbDims = inputDims.nbDims; dims_kmax.nbDims = inputDims.nbDims;

        // To work with dynamic batch sizes, the min, opt and max batch sizes should match
        dims_kmin.d[0] = m_options.optBatchSize; dims_kopt.d[0] = m_options.optBatchSize; dims_kmax.d[0] = m_options.maxBatchSize;
        for (int i=1; i<inputDims.nbDims; i++){
            dims_kmin.d[i] = inputDims.d[i];
            dims_kopt.d[i] = inputDims.d[i];
            dims_kmax.d[i] = inputDims.d[i];
        }

        // Set profiles
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, dims_kmin); //Dims(m_options.optBatchSize, inputDims.d[1], inputDims.d[2], inputDims.d[3], inputDims.d[4]));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, dims_kopt); //Dims(m_options.optBatchSize, inputDims.d[1], inputDims.d[2], inputDims.d[3], inputDims.d[4]));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, dims_kmax); //Dims(m_options.maxBatchSize, inputDims.d[1], inputDims.d[2], inputDims.d[3], inputDims.d[4]));
        
    }
    // Add profiles to the config variable
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
    } 

    // Set the precision level
    if (m_options.precision == Precision::INT8) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }
        config->setFlag(BuilderFlag::kINT8);
    } 

    // Enabled DLA
    if (m_options.dlaCore >= 0){
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << m_options.dlaCore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platform that doesn't have any DLA cores" && false);
        }
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(m_options.dlaCore);
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

TensorRTEngine::~TensorRTEngine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool TensorRTEngine::loadNetwork() {
    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Add the following line if you encounter the error: Error Code 1: Serialization (Serialization assertion creator failed.Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)
    // In this case, also #include "NvInferPlugin.h"
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }


    // The execution context contains all of the state associated with a particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);

        
        if (tensorType == TensorIOMode::kINPUT) {

            // Fill the dimension of input without considering batch dimension, and the number of elements
            uint32_t inputLenFloat = 1;
            m_inputDims.push_back(tensorShape);
            
            for (int j=1; j<tensorShape.nbDims; j++){
                inputLenFloat *= tensorShape.d[j];
            }

            // Allocate memory for the input
            // Allocate enough to fit the batch size
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_options.maxBatchSize * inputLenFloat * sizeof(float), stream));


        } else if (tensorType == TensorIOMode::kOUTPUT) {

            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

// Run inference with the input and output with dims: [n_inputs/n_outputs, batch_size, input_dims/output_dims]. input_dims is the product of all remaining dims of input. Example: if the input is an image, input_dims = C*H*W. Same for output_dim
bool TensorRTEngine::runInference(const std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>>& outputs) {

    // Check that the input is correct
    if (check_input(inputs) == false) return false;

    // Get number of inputs and batch size
    const auto numInputs = m_inputDims.size();
    const auto batchSize = static_cast<int32_t>(inputs[0].size());

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto& dims = m_inputDims[i];

        nvinfer1::Dims inputDims;
        inputDims.nbDims = dims.nbDims;
        inputDims.d[0] = batchSize;
        int inputLenFloat = 1;
        for (int j=1; j< dims.nbDims; j++){
            inputDims.d[j] = dims.d[j];
            inputLenFloat *= dims.d[j];
        }

        
        bool success = m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size
        if (!success) {
            throw std::runtime_error("The expected dimension does not match with the input dimension!");
        }

        // Fill the buffers with the input
        for (int j=0; j<batchSize; j++){
            checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j].data(),
                            inputLenFloat * sizeof(float),
                            cudaMemcpyHostToDevice, inferenceCudaStream));
        }
        

    }


    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    outputs.clear();

    // Extract data per output and per batch
    for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
        
        for (int batch = 0; batch < batchSize; ++batch) {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
            std::vector<float> output;
            
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            
            batchOutputs.emplace_back(std::move(output));
        }
        outputs.emplace_back(std::move(batchOutputs));
    }
    
    
    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}

// Runs inference considering that the input is stored in a doublevector of pointers, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general [n_outputs, batch_size, output_dims]
// The first dimension of the inputs is the number of inputs, the second is the batch size, and pointer has a valid range equal to the dimensionality of the input
bool TensorRTEngine::runInference(const std::vector<std::vector<float *>> &inputs, std::vector<std::vector<std::vector<float>>>& outputs, bool from_device) {

    // Check that the input is correct
    if (check_input(inputs) == false) return false;

    // Get number of inputs and batch size
    const auto numInputs = m_inputDims.size();
    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto& dims = m_inputDims[i];

        nvinfer1::Dims inputDims;
        inputDims.nbDims = dims.nbDims;
        inputDims.d[0] = batchSize;
        int inputLenFloat = 1;
        for (int j=1; j< dims.nbDims; j++){
            inputDims.d[j] = dims.d[j];
            inputLenFloat *= dims.d[j];
        }
        bool success = m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size
        if (!success) {
            throw std::runtime_error("The expected dimension does not match with the input dimension!");
        }

        // Check if the provided data is on the GPU
        if (from_device){
            
            // Fill the buffers with the input from the device
            for (int j=0; j<batchSize; j++){
                checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j],
                                inputLenFloat * sizeof(float),
                                cudaMemcpyDeviceToDevice, inferenceCudaStream));
            }
        }
        else{
            
            // Fill the buffers with the input from the CPU
            for (int j=0; j<batchSize; j++){
                checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j],
                                inputLenFloat * sizeof(float),
                                cudaMemcpyHostToDevice, inferenceCudaStream));
            }
        }
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

   // Copy the outputs back to CPU
    outputs.clear();

    // Extract data per output and per batch
    for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
        
        for (int batch = 0; batch < batchSize; ++batch) {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
            std::vector<float> output;
            
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            
            batchOutputs.emplace_back(std::move(output));
        }
        outputs.emplace_back(std::move(batchOutputs));
    }
    
    
    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}

// Run inference with the triple vector input with dims: [n_inputs/n_outputs, batch_size, input_dims]. input_dims is the product of all remaining dims of input
// The outputs will be stored in a double vector of pointers, which can be in CPU (to_device=false) or GPU (to_device=true)
// The first dimension of the output is the number of outputs, the second is the batch size, and pointer has a valid range equal to the dimensionality of the output
bool TensorRTEngine::runInference(const std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<float *>> & outputs, bool to_device) {

    // Check that the input is correct
    if (check_input(inputs) == false) return false;

    // Get number of inputs and batch size
    const auto numInputs = m_inputDims.size();
    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto& dims = m_inputDims[i];

        nvinfer1::Dims inputDims;
        inputDims.nbDims = dims.nbDims;
        inputDims.d[0] = batchSize;
        int inputLenFloat = 1;
        for (int j=1; j< dims.nbDims; j++){
            inputDims.d[j] = dims.d[j];
            inputLenFloat *= dims.d[j];
        }

        bool success = m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size
        if (!success) {
            throw std::runtime_error("The expected dimension does not match with the input dimension!");
        }

        // Fill the buffers with the input
        for (int j=0; j<batchSize; j++){
            checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j].data(),
                            inputLenFloat * sizeof(float),
                            cudaMemcpyHostToDevice, inferenceCudaStream));
        }
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Extract data per output and per batch
    for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {

        // Batch
        auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
        
        for (int batch = 0; batch < batchSize; ++batch) {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
        
            // Copy the output
            if (to_device)
                checkCudaErrorCode(cudaMemcpyAsync(outputs[outputBinding - numInputs][batch], 
                    static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), 
                    outputLenFloat * sizeof(float), cudaMemcpyDeviceToDevice, inferenceCudaStream));
            else
                checkCudaErrorCode(cudaMemcpyAsync(outputs[outputBinding - numInputs][batch], 
                    static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), 
                    outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
        }
    }
    
    
    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}

// Runs inference considering that the input is stored in a double vector of pointers, which can be in CPU (from_device=false) or GPU (from_device=true)
// The outputs will also be stored in a double vector of pointers, which can be in CPU (to_device=false) or GPU (to_device=true)
// The first dimension of the inputs and outputs is the number of inputs/outputs, the second is the batch size, and pointer has a valid range equal to the dimensionality of the input/output
bool TensorRTEngine::runInference(const std::vector<std::vector<float *>> &inputs, std::vector<std::vector<float *>> & outputs, bool from_device, bool to_device) {

    // Check that the input is correct
    if (check_input(inputs) == false) return false;

    // Get number of inputs and batch size
    const auto numInputs = m_inputDims.size();
    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto& dims = m_inputDims[i];

        nvinfer1::Dims inputDims;
        inputDims.nbDims = dims.nbDims;
        inputDims.d[0] = batchSize;
        int inputLenFloat = 1;
        for (int j=1; j< dims.nbDims; j++){
            inputDims.d[j] = dims.d[j];
            inputLenFloat *= dims.d[j];
        }
        bool success = m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size
        if (!success) {
            throw std::runtime_error("The expected dimension does not match with the input dimension!");
        }

        // Check if the provided data is on the GPU
        if (from_device){
            
            // Fill the buffers with the input from the device
            for (int j=0; j<batchSize; j++){
                checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j],
                                inputLenFloat * sizeof(float),
                                cudaMemcpyDeviceToDevice, inferenceCudaStream));
            }
        }
        else{
            
            // Fill the buffers with the input from the CPU
            for (int j=0; j<batchSize; j++){
                checkCudaErrorCode(cudaMemcpyAsync((char *)m_buffers[i] + j*inputLenFloat*sizeof(float), batchInput[j],
                                inputLenFloat * sizeof(float),
                                cudaMemcpyHostToDevice, inferenceCudaStream));
            }
        }
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Extract data per output and per batch
    for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {

        // Batch
        auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
        
        for (int batch = 0; batch < batchSize; ++batch) {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
        
            // Copy the output
            if (to_device)
                checkCudaErrorCode(cudaMemcpyAsync(outputs[outputBinding - numInputs][batch], 
                    static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), 
                    outputLenFloat * sizeof(float), cudaMemcpyDeviceToDevice, inferenceCudaStream));
            else
                checkCudaErrorCode(cudaMemcpyAsync(outputs[outputBinding - numInputs][batch], 
                    static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), 
                    outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
        }
    }
    
    
    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}

// This function converts the engine options into a string
std::string TensorRTEngine::serializeEngineOptions(const Options &options, const std::string& onnxModelPath) {
    
    // If the path doesn't end with a /, we add it
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName;
    if (options.out_file_path.empty() ||  options.out_file_path.back()=='/')
        engineName = options.out_file_path + onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";
    else
        engineName = options.out_file_path + "/" + onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the precision to the name
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::FP32){
        engineName += ".fp32";
    } else if (options.precision == Precision::INT8){
        engineName += ".int8";
    } 
    else{
        throw std::runtime_error("Error, the precision must be FP16 or FP32 or INT8!");
    }

    // Check if the GPU is allowed, and add its index to the name
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    engineName += "." + std::to_string(options.deviceIndex);
    
    // Add the batch and DLA 
    engineName += "." + std::to_string(options.optBatchSize);
    engineName += "." + std::to_string(options.dlaCore);

    return engineName;
}

// Gets the device names
void TensorRTEngine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

// Utility method for transforming triple nested output array into 2D array
// Should be used when there is just one output, but the batch_size may be >1
void TensorRTEngine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

// Utility method for transforming triple nested output array into single array
// Should be used when the output batch size is 1, and there is only a single output feature vector
void TensorRTEngine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}
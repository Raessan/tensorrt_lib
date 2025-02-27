#include "nn_handler_lib/nn_handler.hpp"

// Constructor with the filepath to the model .ONNX and the destination folder for the engine. The constructor reads the model and, if it exists, builds the engine in the destination folder.
NNHandler::NNHandler(std::string file_path_onnx, std::string file_path_destination, Precision p, int dla_core, int device_index, int batch_size){
    // Ensure the onnx model exists
    if (!Util::doesFileExist(file_path_onnx)) {
        std::cout << "Error: Unable to find file at path: " << file_path_onnx << std::endl;
        throw std::runtime_error("Error: Unable to find file!");
    }
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = p;
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = batch_size;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = batch_size;
    // Specify the out file path
    options.out_file_path = file_path_destination;
    // Specify the DLA core (-1 does not activate it)
    options.dlaCore = dla_core;
    // Device index (number of GPU)
    options.deviceIndex = device_index;
    // Initialize the engine
    engine = std::make_shared<TensorRTEngine>(options);
    

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine->build(file_path_onnx);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine->loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Get features from the engine
    this->batch_size = batch_size;
    auto m_inputDims = engine->getInputDims();
    auto m_outputDims = engine->getOutputDims();
    auto m_IOTensorNames = engine->getTensorNames();

    // Fill input data
    n_inputs = m_inputDims.size();
    dims_in.resize(n_inputs);
    for (int i=0; i< n_inputs; i++){
        n_dim_in.push_back(m_inputDims[i].nbDims-1);
        int n_elems = 1;
        for (int j=1; j<m_inputDims[i].nbDims; j++){
            dims_in[i].push_back(m_inputDims[i].d[j]);
            n_elems *= m_inputDims[i].d[j];
        }
        n_elems_in.push_back(n_elems);
        input_names.push_back(m_IOTensorNames[i]);
    }

    // Fill output data
    n_outputs = m_outputDims.size();
    dims_out.resize(n_outputs);
    for (int i=0; i< n_outputs; i++){
        n_dim_out.push_back(m_outputDims[i].nbDims-1);
        int n_elems = 1;
        for (int j=1; j<m_outputDims[i].nbDims; j++){
            dims_out[i].push_back(m_outputDims[i].d[j]);
            n_elems *= m_outputDims[i].d[j];
        }
        n_elems_out.push_back(n_elems);
        output_names.push_back(m_IOTensorNames[n_inputs+i]);
    }

    // Allocate data for the input vector
    auxiliar_input_vector.resize(n_inputs);
    auxiliar_input_pointer.resize(n_inputs);
    for (int i=0; i<n_inputs; i++){
        auxiliar_input_vector[i].resize(batch_size);
        auxiliar_input_pointer[i].resize(batch_size);
    }

    // Allocate data for the output pointers
    auxiliar_output_pointer_cpu.resize(n_outputs);
    auxiliar_output_pointer_gpu.resize(n_outputs);
    for (int i=0; i<n_outputs; i++){
        auxiliar_output_pointer_cpu[i].resize(batch_size);
        auxiliar_output_pointer_gpu[i].resize(batch_size);
        for (int j=0; j<batch_size; j++){
            auxiliar_output_pointer_cpu[i][j] = new float[n_elems_out[i]];
            Util::checkCudaErrorCode(cudaMalloc((void**)&auxiliar_output_pointer_gpu[i][j], sizeof(float)*n_elems_out[i]));
        }
    }
    
    
}

NNHandler::~NNHandler(){
    for (int i=0; i<n_outputs; i++){
        for (int j=0; j<batch_size; j++){
            delete[] auxiliar_output_pointer_cpu[i][j];
            Util::checkCudaErrorCode(cudaFree(auxiliar_output_pointer_gpu[i][j]));
        }
    }
}

// Runs inference. Most general case, valid for any number of inputs and outputs
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<std::vector<float>>> &output){
    assert(input.size() == n_inputs && input[0].size() == batch_size);
    auxiliar_output_vector.clear();
    bool succ = engine->runInference(input, output);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
}
// Runs inference considering that the input is general, and the output has 1 variable
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input is general, and the output has 1 variable and 1 batch
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}

// Runs inference considering that the input has 1 variable, and the output is general
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, std::vector<std::vector<std::vector<float>>> &output){
    assert(n_inputs==1 && input.size() == batch_size);
    for (int i=0; i<batch_size; i++){
        auxiliar_input_vector[0][i] = input[i];
    }
    run_inference(auxiliar_input_vector, output);
}
// Runs inference considering that the input has 1 variable, and the output has 1 variable
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input has 1 variable, and the output has 1 variable and 1 batch
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, std::vector<float> &output){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}

// Runs inference considering that the input has 1 variable and 1 batch, and the output is general
void NNHandler::run_inference(const std::vector<float> &input, std::vector<std::vector<std::vector<float>>> &output){
    assert(n_inputs==1 && batch_size == 1);
    auxiliar_input_vector[0][0] = input;
    run_inference(auxiliar_input_vector, output);
}
// Runs inference considering that the input has 1 variable and 1 batch, and the output has 1 variable
void NNHandler::run_inference(const std::vector<float> &input, std::vector<std::vector<float>> &output){
    assert(n_inputs==1 && batch_size == 1 && n_outputs==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input has 1 variable and 1 batch, and the output has 1 variable and 1 batch
void NNHandler::run_inference(const std::vector<float> &input, std::vector<float> &output){
    assert(n_inputs==1 && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector);
    engine->transformOutput(auxiliar_output_vector, output);
}

// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size);
    auxiliar_output_vector.clear();
    bool succ = engine->runInference(input, output, from_device);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
}
// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<float>> &output, bool from_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<float> &output, bool from_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}

// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
void NNHandler::run_inference(const std::vector<float *> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device){
    assert(n_inputs==1 && input.size() == batch_size);
    for (int i=0; i< batch_size; i++){
        auxiliar_input_pointer[0][i] = const_cast<float *>(input[i]);
    }
    run_inference(auxiliar_input_pointer, output, from_device);
}
// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
void NNHandler::run_inference(const std::vector<float *> &input, std::vector<std::vector<float>> &output, bool from_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
void NNHandler::run_inference(const std::vector<float *> &input, std::vector<float> &output, bool from_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}

// Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
void NNHandler::run_inference(const float * input, std::vector<std::vector<std::vector<float>>> &output, bool from_device){
    assert(n_inputs==1 && batch_size == 1);
    auxiliar_input_pointer[0][0] = const_cast<float *>(input);
    run_inference(auxiliar_input_pointer, output, from_device);
}
// Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
void NNHandler::run_inference(const float * input, std::vector<std::vector<float>> &output, bool from_device){
    assert(n_inputs==1 && batch_size == 1 && n_outputs==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}
// Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
void NNHandler::run_inference(const float * input, std::vector<float> &output, bool from_device){
    assert(n_inputs==1 && n_outputs==1 && batch_size==1);
    run_inference(input, auxiliar_output_vector, from_device);
    engine->transformOutput(auxiliar_output_vector, output);
}


// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<float *>> &output, bool from_device, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size);
    if (to_device){
        bool succ = engine->runInference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
        output = auxiliar_output_pointer_gpu;
    }
    else{
        bool succ = engine->runInference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
        output = auxiliar_output_pointer_cpu;
    }
}
// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, std::vector<float *> &output, bool from_device, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
// Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float *>> &input, float * &output, bool from_device, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}
// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float *> &input, std::vector<std::vector<float *>> &output, bool from_device, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size);
    for (int i=0; i<input.size(); i++){
        auxiliar_input_pointer[0][i] = input[i];
    }
    run_inference(auxiliar_input_pointer, output, from_device, to_device);
}
// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float *> &input, std::vector<float *> &output, bool from_device, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
// Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float *> &input, float * &output, bool from_device, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}

// Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const float * input, std::vector<std::vector<float *>> &output, bool from_device, bool to_device){
    assert(n_inputs==1 && batch_size == 1);
    auxiliar_input_pointer[0][0] = const_cast<float *>(input);
    run_inference(auxiliar_input_pointer, output, from_device, to_device);
}
// Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const float * input, std::vector<float *> &output, bool from_device, bool to_device){
    assert(n_inputs==1 && batch_size == 1 && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
//Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const float * input, float * &output, bool from_device, bool to_device){
    assert(n_inputs==1 && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, from_device, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, from_device, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}


// Runs inference considering that the input is general (on CPU), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float *>> &output, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size);
    if (to_device){
        bool succ = engine->runInference(input, auxiliar_output_pointer_gpu, to_device);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
        output = auxiliar_output_pointer_gpu;
    }
    else{
        bool succ = engine->runInference(input, auxiliar_output_pointer_cpu, to_device);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
        output = auxiliar_output_pointer_cpu;
    }
}
// Runs inference considering that the input is general (on CPU), and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<float *> &output, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
// Runs inference considering that the input is general (on CPU), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<std::vector<float>>> &input, float * &output, bool to_device){
    assert(input.size() == n_inputs && input[0].size() == batch_size && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}
// Runs inference considering that the input has 1 variable (on CPU), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, std::vector<std::vector<float *>> &output, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size);
    for (int i=0; i<input.size(); i++){
        auxiliar_input_vector[0][i] = input[i];
    }
    run_inference(auxiliar_input_vector, output, to_device);
}
// Runs inference considering that the input has 1 variable (on CPU) and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, std::vector<float *> &output, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
// Runs inference considering that the input has 1 variable (on CPU), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<std::vector<float>> &input, float * &output, bool to_device){
    assert(n_inputs==1 && input.size() == batch_size && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}
// Runs inference considering that the input has 1 variable and 1 batch (on CPU), and the output is general, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float> &input, std::vector<std::vector<float *>> &output, bool to_device){
    assert(n_inputs==1 && batch_size == 1);
    auxiliar_input_vector[0][0] = input;
    run_inference(auxiliar_input_vector, output, to_device);
}
// Runs inference considering that the input has 1 variable and 1 batch (on CPU), and the output has 1 variable, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float> &input, std::vector<float *> &output, bool to_device){
    assert(n_inputs==1 && batch_size == 1 && n_outputs==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0];
    }
}
// Runs inference considering that the input has 1 variable and 1 batch (on CPU), and the output has 1 variable and batch 1, in the CPU (to_device=false) or the GPU (to_device=true)
void NNHandler::run_inference(const std::vector<float> &input, float * &output, bool to_device){
    assert(n_inputs==1 && n_outputs==1 && batch_size==1);
    if (to_device){
        run_inference(input, auxiliar_output_pointer_gpu, to_device);
        output = auxiliar_output_pointer_gpu[0][0];
    }
    else{
        run_inference(input, auxiliar_output_pointer_cpu, to_device);
        output = auxiliar_output_pointer_cpu[0][0];
    }
}

// Prints the data of the handler related to inputs and outputs and their dimensions
void NNHandler::print_data(){
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Num inputs: " << n_inputs << std::endl;
    for (int i=0; i< n_inputs; i++){
        std::cout << "Name input " << i+1 << ": " << input_names[i] << std::endl;
        std::cout << "Number of dimensions: " << n_dim_in[i] << std::endl;
        for (int j=0; j<n_dim_in[i]; j++){
            std::cout << "Dimension " << j+1 << ": " << dims_in[i][j] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

    std::cout << "Num outputs: " << n_outputs << std::endl;
    for (int i=0; i< n_outputs; i++){
        std::cout << "Name output " << i+1 << ": " << output_names[i] << std::endl;
        std::cout << "Number of dimensions: " << n_dim_out[i] << std::endl;
        for (int j=0; j<n_dim_out[i]; j++){
            std::cout << "Dimension " << j+1 << ": " << dims_out[i][j] << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }

}
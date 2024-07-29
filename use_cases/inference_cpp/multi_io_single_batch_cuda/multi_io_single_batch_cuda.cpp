#include <iostream>
#include <set>
#include <memory>
#include <chrono>
#include <iomanip>

#include "nn_handler_lib/nn_handler.hpp"
#include "aux_fcn.hpp"

// PARAMETERS
/** \brief Number of inferences for warmup */
constexpr int n_inferences_warmup = 10;
/** \brief Number of inferences to calculate the average time */
constexpr int n_inferences = 100;
/** \brief Path of the ONNX model*/
const std::string path_model_onnx = "../../models/multi_io/model_single_batch.onnx";
/** \brief Path to save the TensorRT engine for inference*/
const std::string path_engine_save = "../../models/multi_io";
// Precision of the NN. Either FP16 or FP32
Precision precision = Precision::FP16;
// DLA core to use. If -1, it is not used
int dla_core = -1;
// GPU index (ORIN only has the 0 index)
int device_index=0;
// Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1.
int batch_size = 1;
         
// Input and output files
const std::string input1_path = "../../test_data/multi_io/x1.txt";
const std::string input2_path = "../../test_data/multi_io/x2.txt";
const std::string input3_path = "../../test_data/multi_io/x3.txt";
const std::string output1_path = "../../test_data/multi_io/y1.txt";
const std::string output2_path = "../../test_data/multi_io/y2.txt";

int main(){

    // Create variable for inference
    NNHandler nn_handler(path_model_onnx, path_engine_save, precision, dla_core, device_index, batch_size);
    // Print the data of the handler
    nn_handler.print_data();

    // Load input and output ground truth. 
    std::vector<std::vector<float>> input1_file = read_file(input1_path, batch_size);
    std::vector<std::vector<float>> input2_file = read_file(input2_path, batch_size);
    std::vector<std::vector<float>> input3_file = read_file(input3_path, batch_size);
    std::vector<std::vector<float>> output1_gt_file = read_file(output1_path, batch_size);
    std::vector<std::vector<float>> output2_gt_file = read_file(output2_path, batch_size);

    // This is the vector that will have the first batch of each input on the GPU
    std::vector<std::vector<float *>> d_input;
    // We have to resize it to contain all the data
    d_input.resize(3);
    for (int i=0; i<3; i++){
        d_input[i].resize(1);
    }
    
    // Now, we allocate data on the GPU and fill there the data
    checkCuda(cudaMalloc((void**)&d_input[0][0], sizeof(float)*input1_file[0].size()));
    checkCuda(cudaMalloc((void**)&d_input[1][0], sizeof(float)*input2_file[0].size()));
    checkCuda(cudaMalloc((void**)&d_input[2][0], sizeof(float)*input3_file[0].size()));
    checkCuda(cudaMemcpy(d_input[0][0], input1_file[0].data(), sizeof(float)*input1_file[0].size(),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_input[1][0], input2_file[0].data(), sizeof(float)*input2_file[0].size(),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_input[2][0], input3_file[0].data(), sizeof(float)*input3_file[0].size(),cudaMemcpyHostToDevice));
    
    // We take the inner std::vector of the outputs
    std::vector<float> output1_gt = output1_gt_file[0];
    std::vector<float> output2_gt = output2_gt_file[0];

    // Predicted output
    std::vector<std::vector<std::vector<float>>> output_pred;
    
    // Perform WARMUP inference (only with the first batch)
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(d_input, output_pred, true);
    }
    
    // Get the current time before inference
    auto start = std::chrono::high_resolution_clock::now();
    // Measure time of inference
    for (int i=0; i<n_inferences; i++){
        nn_handler.run_inference(d_input, output_pred, true);
    }
    // Get the current time after inference
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to perform inference: " << duration.count()/n_inferences << " milliseconds" << std::endl;

    // Show the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Ground truth output 1: " << std::endl;
    for (int i=0; i<10; i++){
        std::cout << output1_gt[i] << " ";
    }
    std::cout << "\nPredicted output 1: " << std::endl;
    for (int i=0; i<10; i++){
        std::cout << output_pred[0][0][i] << " ";
    }
    std::cout << "\nGround truth output 2: " << std::endl;
    for (int i=0; i<5; i++){
        std::cout << output2_gt[i] << " ";
    }
    std::cout << "\nPredicted output 2: " << std::endl;
    for (int i=0; i<5; i++){
        std::cout << output_pred[1][0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Mean square error output 1: " << calculate_mae(output1_gt, output_pred[0][0], 10) << std::endl;
    std::cout << "Mean square error output 2: " << calculate_mae(output2_gt, output_pred[1][0], 5) << std::endl;

    // Free all the CUDA pointers
    for (int i=0; i<d_input.size(); i++){
        checkCuda(cudaFree(d_input[i][0]));
    }

    return 0;
}
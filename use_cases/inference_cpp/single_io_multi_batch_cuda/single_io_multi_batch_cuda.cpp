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
const std::string path_model_onnx = "../../models/single_io/model_multi_batch.onnx";
/** \brief Path to save the TensorRT engine for inference*/
const std::string path_engine_save = "../../models/single_io";
// Precision of the NN
Precision precision = Precision::FP16;
// DLA core to use. If -1, it is not used
int dla_core = -1;
// GPU index (ORIN only has the 0 index)
int device_index=0;
// Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1.
int batch_size = 3;
         
// Input and output files
const std::string input_path = "../../test_data/single_io/x.txt";
const std::string output_path = "../../test_data/single_io/y.txt";

int main(){

    // Create variable for inference
    NNHandler nn_handler(path_model_onnx, path_engine_save, precision, dla_core, device_index, batch_size);
    // Print the data of the handler
    nn_handler.print_data();

    // Load input and output ground truth. Since the input is single, we can pass the std::vector<std::vector<float>> directly to the network
    std::vector<std::vector<float>> input = read_file(input_path, batch_size);

    // Ground truth output
    std::vector<std::vector<float>> output_gt = read_file(output_path, batch_size);

    // We create the input pointers to upload to the device. We only need one outer vector because the number of inputs is 1, so we only need the batch dimension
    std::vector<float *> d_input(batch_size);
    // Fill with the data from file
    for (int i=0; i<batch_size; i++){
        checkCuda(cudaMalloc((void**)&d_input[i], sizeof(float)*input[i].size()));
        checkCuda(cudaMemcpy(d_input[i], input[i].data(), sizeof(float)*input[i].size(),cudaMemcpyHostToDevice));
    }

    // Predicted output. Since the input is single, we can use std::vector<std::vector<float>>, where the first dimension is the batch size
    std::vector<std::vector<float>> output_pred;

    // Perform WARMUP inference (only with the first batch) 
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(d_input, output_pred, true);
    }
    
    // Get the current time before inference
    auto start = std::chrono::high_resolution_clock::now();
    // Measure time of inference
    for (int i=0; i<n_inferences; i++){
        // Inference in GPU memory
        nn_handler.run_inference(d_input, output_pred, true);
    }
    // Get the current time after inference
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to perform inference: " << duration.count()/n_inferences << " milliseconds" << std::endl;

    // Show the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Ground truth output: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<10; j++){
            std::cout << output_gt[i][j] << " ";
        }
        std::cout << std::endl;
    }   
    std::cout << "Predicted output: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<10; j++){
            std::cout << output_pred[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Mean square error: " << calculate_mae(output_gt, output_pred, 10) << std::endl;

    // Free all the CUDA pointers
    for (int i=0; i<d_input.size(); i++){
        checkCuda(cudaFree(d_input[i]));
    }

    return 0;
}
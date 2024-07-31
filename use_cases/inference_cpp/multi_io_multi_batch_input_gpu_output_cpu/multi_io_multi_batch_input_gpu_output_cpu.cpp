#include <iostream>
#include <set>
#include <memory>
#include <chrono>
#include <iomanip>

#include "nn_handler_lib/nn_handler.hpp"
#include "aux_fcn.hpp"

// PARAMETERS
/** \brief Number of inferences for warmup */
constexpr int n_inferences_warmup = 100;
/** \brief Number of inferences to calculate the average time */
constexpr int n_inferences = 1000;
/** \brief Path of the ONNX model*/
const std::string path_model_onnx = "../../models/multi_io/model_multi_batch.onnx";
/** \brief Path to save the TensorRT engine for inference*/
const std::string path_engine_save = "../../models/multi_io";
// Precision of the NN. Either FP16 or FP32
Precision precision = Precision::FP16;
// DLA core to use. If -1, it is not used
int dla_core = -1;
// GPU index (ORIN only has the 0 index)
int device_index=0;
// Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1.
int batch_size = 3;
         
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
    std::vector<std::vector<float>> input1 = read_file(input1_path, batch_size);
    std::vector<std::vector<float>> input2 = read_file(input2_path, batch_size);
    std::vector<std::vector<float>> input3 = read_file(input3_path, batch_size);
    std::vector<std::vector<float>> output1_gt = read_file(output1_path, batch_size);
    std::vector<std::vector<float>> output2_gt = read_file(output2_path, batch_size);

    // This is the vector that will contain the info of all inputs
    std::vector<std::vector<float *>> input_gpu;

    // We have to resize it to contain all the data
    input_gpu.resize(3);
    for (int i=0; i<3; i++){
        input_gpu[i].resize(batch_size);
    }
    // Now, we allocate data on the GPU and fill there the data
    for (int i=0; i<batch_size; i++){
        checkCuda(cudaMalloc((void**)&input_gpu[0][i], sizeof(float)*input1[i].size()));
        checkCuda(cudaMalloc((void**)&input_gpu[1][i], sizeof(float)*input2[i].size()));
        checkCuda(cudaMalloc((void**)&input_gpu[2][i], sizeof(float)*input3[i].size()));
        checkCuda(cudaMemcpy(input_gpu[0][i], input1[i].data(), sizeof(float)*input1[i].size(),cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(input_gpu[1][i], input2[i].data(), sizeof(float)*input2[i].size(),cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(input_gpu[2][i], input3[i].data(), sizeof(float)*input3[i].size(),cudaMemcpyHostToDevice));
    }
    

    // Predicted output
    std::vector<std::vector<std::vector<float>>> output_pred;
    
    // Perform WARMUP inference (only with the first batch)
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(input_gpu, output_pred, true);
    }
    
    // Get the current time before inference
    auto start = std::chrono::high_resolution_clock::now();
    // Measure time of inference
    for (int i=0; i<n_inferences; i++){
        nn_handler.run_inference(input_gpu, output_pred, true);
    }
    // Get the current time after inference
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to perform inference: " << duration.count()/n_inferences << " milliseconds" << std::endl;

    // Show the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Ground truth output 1: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<10; j++){
            std::cout << output1_gt[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Predicted output 1: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<10; j++){
            std::cout << output_pred[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Ground truth output 2: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<5; j++){
            std::cout << output2_gt[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Predicted output 2: " << std::endl;
    for (int i=0; i<batch_size; i++){
        for (int j=0; j<5; j++){
            std::cout << output_pred[1][i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Mean square error output 1: " << calculate_mae(output1_gt, output_pred[0], 10) << std::endl;
    std::cout << "Mean square error output 2: " << calculate_mae(output2_gt, output_pred[1], 5) << std::endl;

    // Free all the CUDA pointers
    for (int i=0; i<input_gpu.size(); i++){
        for (int j=0; j<input_gpu[0].size(); j++){
            checkCuda(cudaFree(input_gpu[i][j]));
        }
    }

    return 0;
}
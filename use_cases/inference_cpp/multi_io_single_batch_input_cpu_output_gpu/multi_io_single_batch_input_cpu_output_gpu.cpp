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

    // This is the vector that will have the first batch of each input
    std::vector<std::vector<std::vector<float>>> input;
    // We have to resize it to contain all the data
    input.resize(3);
    for (int i=0; i<3; i++){
        input[i].resize(1);
    }
    
    // Now copy the data from each input
    input[0][0] = input1_file[0];
    input[1][0] = input2_file[0];
    input[2][0] = input3_file[0];
    
    // We take the inner std::vector of the outputs
    std::vector<float> output1_gt = output1_gt_file[0];
    std::vector<float> output2_gt = output2_gt_file[0];

    // Predicted output in the GPU
    std::vector<std::vector<float *>> output_pred_gpu;
    
    // Perform WARMUP inference. Set to_device to true
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(input, output_pred_gpu, true);
    }
    
    // Get the current time before inference. Set to_device to true
    auto start = std::chrono::high_resolution_clock::now();
    // Measure time of inference
    for (int i=0; i<n_inferences; i++){
        nn_handler.run_inference(input, output_pred_gpu, true);
    }
    // Get the current time after inference
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to perform inference: " << duration.count()/n_inferences << " milliseconds" << std::endl;

    // Now, we get the output on the CPU for comparison
    std::vector<std::vector<float *>> output_pred_cpu;
    output_pred_cpu.resize(2);
    output_pred_cpu[0].resize(batch_size);
    output_pred_cpu[1].resize(batch_size);
    for (int i=0; i<2; i++){
        for (int j=0; j<batch_size; j++){
            // We have to allocate data to copy from GPU
            output_pred_cpu[i][j] = new float[nn_handler.get_n_elems_out()[i]];
            // Copy data from GPU to CPU
            checkCuda(cudaMemcpy(output_pred_cpu[i][j], output_pred_gpu[i][j], sizeof(float)*nn_handler.get_n_elems_out()[i],cudaMemcpyDeviceToHost));
        }
    }

    // Show the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Ground truth output 1: " << std::endl;
    for (int i=0; i<10; i++){
        std::cout << output1_gt[i] << " ";
    }
    std::cout << "\nPredicted output 1: " << std::endl;
    for (int i=0; i<10; i++){
        std::cout << output_pred_cpu[0][0][i] << " ";
    }
    std::cout << "\nGround truth output 2: " << std::endl;
    for (int i=0; i<5; i++){
        std::cout << output2_gt[i] << " ";
    }
    std::cout << "\nPredicted output 2: " << std::endl;
    for (int i=0; i<5; i++){
        std::cout << output_pred_cpu[1][0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Mean square error output 1: " << calculate_mae(output1_gt, output_pred_cpu[0][0], 10) << std::endl;
    std::cout << "Mean square error output 2: " << calculate_mae(output2_gt, output_pred_cpu[1][0], 5) << std::endl;

    // Free the output pointer on the CPU (the GPU pointer is managed automatically by the handler)
    for (int i=0; i<output_pred_cpu.size(); i++){
        for (int j=0; j<output_pred_cpu[0].size(); j++){
            delete[] output_pred_cpu[i][j];
        }
    }

    return 0;
}
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

    // Load input and output ground truth. We will load all the batches but will only use the first one
    std::vector<std::vector<float>> input1 = read_file(input1_path);
    std::vector<std::vector<float>> input2 = read_file(input2_path);
    std::vector<std::vector<float>> input3 = read_file(input3_path);
    std::vector<std::vector<float>> output1_gt = read_file(output1_path);
    std::vector<std::vector<float>> output2_gt = read_file(output2_path);

    // This is the vector that will have the first batch of each input
    std::vector<std::vector<std::vector<float>>> input;
    // We have to resize it to contain all the data
    input.resize(3);
    for (int i=0; i<3; i++){
        input[i].resize(batch_size);
    }
    for (int i=0; i<batch_size; i++){
        input[0][i].resize(input1[i].size());
        std::copy(input1[i].begin(), input1[i].end(), input[0][i].begin());
    }
    for (int i=0; i<batch_size; i++){
        input[1][i].resize(input2[i].size());
        std::copy(input2[i].begin(), input2[i].end(), input[1][i].begin());
    }
    for (int i=0; i<batch_size; i++){
        input[2][i].resize(input3[i].size());
        std::copy(input3[i].begin(), input3[i].end(), input[2][i].begin());
    }


    // Predicted output
    std::vector<std::vector<std::vector<float>>> output_pred;
    
    // Perform WARMUP inference (only with the first batch)
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(input, output_pred);
    }
    
    // Get the current time before inference
    auto start = std::chrono::high_resolution_clock::now();
    // Measure time of inference
    for (int i=0; i<n_inferences; i++){
        nn_handler.run_inference(input, output_pred);
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

    return 0;
}
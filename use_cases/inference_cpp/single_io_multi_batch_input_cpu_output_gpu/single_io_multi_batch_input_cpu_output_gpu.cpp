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

    // Predicted output. Pointers of vectors on the GPU
    std::vector<float *> output_pred_gpu;

    // Perform WARMUP inference (only with the first batch). We set to_device=true
    for (int i=0; i< n_inferences_warmup; i++){
        nn_handler.run_inference(input, output_pred_gpu, true);
    }
    
    // Get the current time before inference. We set to_device=true
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
    std::vector<float *> output_pred_cpu;
    output_pred_cpu.resize(batch_size);
    for (int i=0; i<batch_size; i++){
        // We have to allocate data to copy from GPU
        output_pred_cpu[i] = new float[nn_handler.get_n_elems_out()[0]];
        // Copy data from GPU to CPU
        checkCuda(cudaMemcpy(output_pred_cpu[i], output_pred_gpu[i], sizeof(float)*nn_handler.get_n_elems_out()[0],cudaMemcpyDeviceToHost));
    }

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
            std::cout << output_pred_cpu[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Mean square error: " << calculate_mae(output_gt, output_pred_cpu, 10) << std::endl;

     // Free the output pointer on the CPU (the GPU pointer is managed automatically by the handler)
    for (int i=0; i<output_pred_cpu.size(); i++){
        delete[] output_pred_cpu[i];
    }

    return 0;
}
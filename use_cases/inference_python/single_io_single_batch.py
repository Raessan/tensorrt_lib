
import numpy as np
import sys
import os
import tensorrt as trt
import time

engine_path = os.path.abspath(os.path.join("..", "..", "tensorrt_python" ))
sys.path.insert(0, engine_path)
from tensorrt_engine import TensorRTEngine, Options

# PARAMETERS
# Number of inferences for warmup
n_inferences_warmup = 10
# Number of inferences to calculate the average time */
n_inferences = 100
# Path of the ONNX model*/
path_model_onnx = "../models/single_io/model_single_batch.onnx"
# Path to save the TensorRT engine for inference*/
path_engine_save = "../models/single_io"
# Precision of the NN. Either FP16 or FP32
precision = trt.float16
# DLA core to use. If -1, it is not used
dla_core = -1
# GPU index (ORIN only has the 0 index)
device_index=0
# Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1.
batch_size = 1

# Input and output files
input_path = "../test_data/single_io/x.txt"
output_path = "../test_data/single_io/y.txt"

if __name__ == "__main__":

    # We create the options variable, and only modify the out_file_path. We leave the rest of options by default
    trt_options = Options()
    trt_options.out_file_path = path_engine_save
    trt_options.precision = precision
    trt_options.batch_size = batch_size
    trt_options.deviceIndex = device_index
    trt_options.dlaCore = dla_core
    trt_options.out_file_path = path_engine_save

    # Create the engine, build and load the network
    engine = TensorRTEngine(trt_options)
    engine.build(path_model_onnx)
    engine.loadNetwork()

    # Print engine data
    engine.print_data()

    # Get the dimensions from the engine
    input_dims = engine.input_dims[0]
    output_dims = engine.output_dims[0]


    # Load data
    input_all_batches = np.loadtxt(input_path, delimiter=',')
    output_gt_all_batches = np.loadtxt(output_path, delimiter=',')

    # Take first batch
    input = input_all_batches[0,:]
    output_gt = output_gt_all_batches[0,:]

    # Reshape to meet dimensions
    input = input.reshape(input_dims)
    output_gt = output_gt.reshape(output_dims)

    # The function expects a list of inputs
    inputs = [input]

    # Warmup inference
    for i in range(n_inferences_warmup):
        _ = engine.do_inference_v2(inputs)

    # Measure inference time
    start_time = time.time()
    for i in range(n_inferences):
        output_pred = engine.do_inference_v2(inputs)
    end_time = time.time()
    avg_inference_time = (end_time - start_time)*1000.0 / n_inferences
    print(f"Average inference time: {avg_inference_time:.6f} miliseconds")

    # Show results
    print("Ground truth output:\n", output_gt)
    print("Predicted output:\n", output_pred[0])

    # Show errors
    mae_output = np.mean(np.abs(output_gt - output_pred[0]))
    print("Mean square error: ", mae_output)

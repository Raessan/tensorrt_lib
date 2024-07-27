
import numpy as np
import sys
import os
import tensorrt as trt
import time

engine_path = os.path.abspath(os.path.join("..", "..", "tensorrt_python" ))
sys.path.insert(0, engine_path)
from engine import Engine, Options

# PARAMETERS
# Number of inferences for warmup
n_inferences_warmup = 10
# Number of inferences to calculate the average time */
n_inferences = 100
# Path of the ONNX model*/
path_model_onnx = "../models/multi_io/model_single_batch.onnx"
# Path to save the TensorRT engine for inference*/
path_engine_save = "../models/multi_io"
# Precision of the NN. Either FP16 or FP32
precision = trt.float16
# DLA core to use. If -1, it is not used
dla_core = -1
# GPU index (ORIN only has the 0 index)
device_index=0
# Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1.
batch_size = 1

# Input and output files
input1_path = "../test_data/multi_io/x1.txt"
input2_path = "../test_data/multi_io/x2.txt"
input3_path = "../test_data/multi_io/x3.txt"
output1_path = "../test_data/multi_io/y1.txt"
output2_path = "../test_data/multi_io/y2.txt"

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
    engine = Engine(trt_options)
    engine.build(path_model_onnx)
    engine.loadNetwork()

    # Print engine data
    engine.print_data()

    # # Load data
    input1_all_batches = np.loadtxt(input1_path, delimiter=',')
    input2_all_batches = np.loadtxt(input2_path, delimiter=',')
    input3_all_batches = np.loadtxt(input3_path, delimiter=',')

    output1_gt_all_batches = np.loadtxt(output1_path, delimiter=',')
    output2_gt_all_batches = np.loadtxt(output2_path, delimiter=',')

    # Take first batch
    input1 = input1_all_batches[0,:]
    input2 = input2_all_batches[0,:]
    input3 = input3_all_batches[0,:]
    output1_gt = output1_gt_all_batches[0,:]
    output2_gt = output2_gt_all_batches[0,:]

    # Reshape to meet dimensions
    input1 = input1.reshape(engine.input_dims[0])
    input2 = input2.reshape(engine.input_dims[1])
    input3 = input3.reshape(engine.input_dims[2])
    output1_gt = output1_gt.reshape(engine.output_dims[0])
    output2_gt = output2_gt.reshape(engine.output_dims[1])

    # The function expects a list of inputs
    inputs = [input1, input2, input3]

    # Warmup inference
    for i in range(n_inferences_warmup):
        _ = engine.do_inference_v2(inputs)

    # Measure inference time
    start_time = time.time()
    for i in range(n_inferences):
        output_pred = engine.do_inference_v2(inputs)
    end_time = time.time()

    # Show results
    print("Ground truth output 1:\n", output1_gt)
    print("Predicted output 1:\n", output_pred[0])

    print("Ground truth output 2:\n", output2_gt)
    print("Predicted output 2:\n", output_pred[1])

    # Show errors
    mae_output1 = np.mean(np.abs(output1_gt - output_pred[0]))
    mae_output2 = np.mean(np.abs(output2_gt - output_pred[1]))

    print("Mean square error 1: ", mae_output1)
    print("Mean square error 2: ", mae_output2)

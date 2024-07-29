#ifndef NN_HANDLER_HPP_
#define NN_HANDLER_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

#include "tensorrt_cpp_lib/tensorrt_engine.h"

/** 
 * @brief Class that creates a NN handler with TensorRT to build and run inference with an ONNX model, and to keep information about inputs and outputs. This class assumes that the batch size is always 1
 * 
 */
class NNHandler{
    
    public:
        /**
         * @brief Constructor with the filepath to the model .ONNX and the destination folder for the engine. The constructor reads the model and, if it exists, builds the engine in the destination folder.
         * @param file_path_onnx Path to the onnx file
         * @param file_path_destination Path to the destination file
         * @param p Precision of the NN. Default: FP16
         * @param dla_core DLA core to use. By default, it does not use it (by setting it to -1)
         * @param device_index GPU index (ORIN only has the 0 index)
         * @param batch_size Batch size. If the model has fixed batch, this has to be 1. If the model has dynamic batch, this can be >1. All the inferences with this handler will be performed with that batch size
        */
        NNHandler(std::string file_path_onnx, std::string file_path_destination, Precision p=Precision::FP16, int dla_core = -1, int device_index=0, int batch_size=1);

        /**
         * @brief Runs inference. Most general case, valid for any number of inputs and outputs
         * @param input Triple vector of float pointer with the input of the NN, in order: [n_inputs, batch_size, input_dims]. input_dims is the product of all remaining dims of input. Example: if the input is an image, input_dims = C*H*W
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]. output_dims follows the same rule as input_dims
        */
        void run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<std::vector<float>>> &output);
        /**
         * @brief Runs inference considering that the input is general, and the output has 1 variable
         * @param input Triple vector of float pointer with the input of the NN, in order: [n_inputs, batch_size, input_dims]
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
        */
        void run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<std::vector<float>> &output);
        /**
         * @brief Runs inference considering that the input is general, and the output has 1 variable and 1 batch
         * @param input Triple vector of float pointer with the input of the NN, in order: [n_inputs, batch_size, input_dims].
         * @param output Single vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
        */
        void run_inference(const std::vector<std::vector<std::vector<float>>> &input, std::vector<float> &output);

        /**
         * @brief Runs inference considering that the input has 1 variable, and the output is general
         * @param input Double vector of float with the input of the NN, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]
        */
        void run_inference(const std::vector<std::vector<float>> &input, std::vector<std::vector<std::vector<float>>> &output);
        /**
         * @brief Runs inference considering that the input has 1 variable, and the output has 1 variable
         * @param input Double vector of float with the input of the NN, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
        */
        void run_inference(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output);
        /**
         * @brief Runs inference considering that the input has 1 variable, and the output has 1 variable and 1 batch
         * @param input Double vector of float with the input of the NN, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Single vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
        */
        void run_inference(const std::vector<std::vector<float>> &input, std::vector<float> &output);

        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, and the output is general
         * @param input Single vector of float with the input of the NN, only valid if n_inputs = 1 and batch_size = 1. The length of the vector is "input_dims"
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]
        */
        void run_inference(const std::vector<float> &input, std::vector<std::vector<std::vector<float>>> &output);
        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, and the output has 1 variable
         * @param input Single vector of float with the input of the NN, only valid if n_inputs = 1 and batch_size = 1. The length of the vector is "input_dims"
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
        */
        void run_inference(const std::vector<float> &input, std::vector<std::vector<float>> &output);
        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, and the output has 1 variable and 1 batch
         * @param input Single vector of float with the input of the NN, only valid if n_inputs = 1 and batch_size = 1. The length of the vector is "input_dims"
         * @param output Vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
        */
        void run_inference(const std::vector<float> &input, std::vector<float> &output);

        /**
         * @brief Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
         * @param input Double vector of input pointers. The dimensions are [n_inputs, batch_size, input_dims]
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
         * @param input Double vector of input pointers. The dimensions are [n_inputs, batch_size, input_dims]
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<std::vector<float>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input is general, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
         * @param input Double vector of input pointers. The dimensions are [n_inputs, batch_size, input_dims]
         * @param output Vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<std::vector<float *>> &input, std::vector<float> &output, bool from_device);

        /**
         * @brief Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
         * @param input Input vector of pointers, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<float *> &input, std::vector<std::vector<std::vector<float>>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
         * @param input Input vector of pointers, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<float *> &input, std::vector<std::vector<float>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input has 1 variable, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
         * @param input Input vector of pointers, only valid if n_inputs = 1. The dimensions are [batch_size, input_dims]
         * @param output Vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const std::vector<float *> &input, std::vector<float> &output, bool from_device);

        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output is general
         * @param input Input pointer, only valid if n_inputs = 1 and batch_size = 1. The valid range of the pointer is "input_dims"
         * @param output Triple vector of float with the output of the NN, in order: [n_outputs, batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const float * input, std::vector<std::vector<std::vector<float>>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable
         * @param input Input pointer, only valid if n_inputs = 1 and batch_size = 1. The valid range of the pointer is "input_dims"
         * @param output Double vector of float with the output of the NN, only valid if n_outputs = 1. The dimensions are [batch_size, output_dims]
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const float * input, std::vector<std::vector<float>> &output, bool from_device);
        /**
         * @brief Runs inference considering that the input has 1 variable and 1 batch, which can be in CPU (from_device=false) or GPU (from_device=true), and the output has 1 variable and 1 batch
         * @param input Input pointer, only valid if n_inputs = 1 and batch_size = 1. The valid range of the pointer is "input_dims"
         * @param output Vector of float with the output of the NN, only valid if n_outputs = 1 and batch_size = 1. The length of the vector is "output_dims"
         * @param from_device If true, the input is provided on the GPU. If false, it is provided on the GPU
        */
        void run_inference(const float * input, std::vector<float> &output, bool from_device);


        /**
         * @brief Returns the number of inputs to the NN
         * @return n_inputs
        */
        inline int get_n_inputs(){return n_inputs;}
        /**
         * @brief Returns the number of outputs to the NN
         * @return n_outputs
        */
        inline int get_n_outputs(){return n_outputs;}
        /**
         * @brief Returns the sizes of the inputs of the NN
         * @return n_dim_in
        */
        inline std::vector<std::vector<int>> get_dimensions_in(){return dims_in;}
        /**
         * @brief Returns the sizes of the outputs of the NN
         * @return n_dim_out
        */
        inline std::vector<std::vector<int>> get_dimensions_out(){return dims_out;}
        /**
         * @brief Returns the number of elems of the input to the NN
         * @return n_elems_in
        */
        inline std::vector<int> get_n_elems_in(){return n_elems_in;}
        /**
         * @brief Returns the number of elems of the output to the NN
         * @return n_elems_out
        */
        inline std::vector<int> get_n_elems_out(){return n_elems_out;}
        /**
         * @brief Returns the numbers of the inputs
         * @return input_names
        */
        inline std::vector<std::string> get_input_names(){return input_names;}
        /**
         * @brief Returns the numbers of the outputs
         * @return output_names
        */
        inline std::vector<std::string> get_output_names(){return output_names;}
        /**
         * @brief Returns the batch_size
         * @return batch_size
        */
        inline int get_batch_size(){return batch_size;}

        /**
         * @brief Prints the data of the handler related to inputs and outputs and their dimensions
        */
        void print_data();

        
    private:
        /** \brief Number of inputs */
        int n_inputs;
        /** \brief Sizes  of each input */
        std::vector<int> n_dim_in, n_elems_in;
        /** \brief Dimension sizes of each input */
        std::vector<std::vector<int>> dims_in;
        /** \brief Names of each input */
        std::vector<std::string> input_names;
        /** \brief Number of outputs */
        int n_outputs;
        /** \brief Sizes of each output */
        std::vector<int> n_dim_out, n_elems_out;
        /** \brief Dimension sizes of each output */
        std::vector<std::vector<int>> dims_out;
        /** \brief Names of each output */
        std::vector<std::string> output_names;
        /** \brief Batch size */
        int batch_size;

        /** \brief Options of the NN */
        Options options;
        

        /** \brief Auxiliar output */
        std::vector<std::vector<std::vector<float>>> auxiliar_output;
        /** \brief Auxiliar input vector */
        std::vector<std::vector<std::vector<float>>> auxiliar_input_vector;
        /** \brief Auxiliar input pointer */
        std::vector<std::vector<float *>> auxiliar_input_pointer;
        /** \brief Engine of the NN */
        std::shared_ptr<TensorRTEngine> engine;

};

#endif // NN_HANDLER_HPP_
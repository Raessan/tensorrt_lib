import tensorrt as trt
import numpy as np
from cuda import cuda, cudart
import os.path
import ctypes

# Loggers for errors
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#  Define an explicit batch size and then create the network (implicit batch size is deprecated).
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Check if CUDA command is correct
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

# Performs a cuda call and process result and error
def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

# Struct with options. All have default valid values except out_file_path, which must be provided
class Options:
    def __init__(self):
        self.precision = trt.float16
        self.batch_size = 1
        self.deviceIndex = 0
        self.dlaCore = -1
        self.out_file_path = ""

# Pair of host and device memory, where the host memory is wrapped in a numpy array"""
class HostDeviceMem:

    # Creation of pointer and allocation of memory, both in host and device
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self.memory_released = False

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    # This function frees the memory of host and device
    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))
        self.memory_released = True
        
# Main engine class
class TensorRTEngine:

    # Constructor that initializes variables
    def __init__(self, options):
        # Options class
        self.options = options
        # The execution context contains all of the state associated with a particular invocation
        self.context = None
        # Path to store the engine
        self.engine_path = None
        # Main engine variable
        self.engine = None
        # Inputs of the NN
        self.inputs = []
        # Output of the NN
        self.outputs = []
        # These contain the inputs and outputs on the GPU
        self.bindings = []
        # Configuration profile used to define the inputs and outputs sizes
        self.profile_idx = None
        # Input dimensions, which is a list of shapes. The list has n_input numbers, and the shapes start with the batch dimension
        self.input_dims = []
        # Output dimensions, which is a list of shapes. The list has n_output numbers, and the shapes start with the batch dimension
        self.output_dims = []
        # Names of the inputs
        self.input_names = []
        # Names of the outputs
        self.output_names = []
        # CUDA stream
        self.stream = None
        # Boolean to check if the stream is destroyed
        self.destroyed_stream = False

    # Destructor that frees memory
    def __del__(self):
        # Free memory from declared pointers
        for mem in self.inputs + self.outputs:
            if not mem.memory_released:
                mem.free()
        # Free stream
        if self.stream is not None and self.destroyed_stream is False:
            cuda_call(cudart.cudaStreamDestroy(self.stream))
            self.destroyed_stream = True

    # Build NN from path
    def build(self, onnxModelPath):

        # Get full name of path
        self.engine_path = self.serializeEngineOptions(onnxModelPath)
        
        # If the path exists, we don't re-build it
        if os.path.isfile(self.engine_path):
            print("The engine already exists. Aborting build")
            return

        # If the ONNX path does not exists, we can't proceed
        elif not os.path.isfile(onnxModelPath):
            raise Exception("The provided ONNX path does not exist!!")

        # This function converts the engine options into a string
        self.serializeEngineOptions(onnxModelPath)

        # Create our engine builder
        builder = trt.Builder(TRT_LOGGER)
        # Create the network
        network = builder.create_network(EXPLICIT_BATCH)
        # Create a parser for reading the onnx file
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Load the ONNX model and parse it in order to populate the TensorRT network.
        with open(onnxModelPath, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            
        # Get info from network
        numInputs = network.num_inputs
        if (numInputs < 1):
            raise Exception("Error, model needs at least 1 input!");
        input0Batch = network.get_input(0).shape[0]
        for i in range(numInputs):
            if network.get_input(i).shape[0] is not input0Batch:
                raise Exception("Error, the model has multiple inputs, each with differing batch sizes!")
        
        # Check to see if the model supports dynamic batch size or not
        if input0Batch == -1:
            print("Model supports dynamic batch size")
        elif input0Batch == 1:
            print("Model only supports fixed batch size of 1")
            if (self.options.batch_size is not input0Batch):
                raise Exception("Error model only supports a fixed batch size of 1.")
        
        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = 1<<30

        # Set the precision level
        if self.options.precision == trt.float16:
            if not builder.platform_has_fast_fp16:
                raise Exception("Error: GPU does not support FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.options.precision == trt.int8:
            if not builder.platform_has_fast_int8:
                raise Exception("Error: GPU does not support INT8 precision")
            config.set_flag(trt.BuilderFlag.INT8)

        # Enabled DLA
        if self.options.dlaCore >= 0:
            if builder.num_DLA_cores == 0:
                raise Exception("Trying to use DLA core on a platform that doesn't have any DLA cores")
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.options.dlaCore

        # Register a single optimization profile for the config
        optProfile = builder.create_optimization_profile()
        # Must specify dimensions for all the inputs the model expects.
        for i in range(numInputs):
            input = network.get_input(i)
            inputName = input.name
            inputDims = input.shape
            dims_kmin = []
            dims_kopt = []
            dims_kmax = []
            # To work with dynamic batch sizes, the min, opt and max batch sizes should match
            dims_kmin.append(self.options.batch_size)
            dims_kopt.append(self.options.batch_size)
            dims_kmax.append(self.options.batch_size)
            for dim in inputDims[1:]:
                dims_kmin.append(dim)
                dims_kopt.append(dim)
                dims_kmax.append(dim)
            # Set profiles
            optProfile.set_shape(inputName, dims_kmin, dims_kopt, dims_kmax)
        
        # Add profile to the config variable      
        config.add_optimization_profile(optProfile)

        # Write the engine to disk
        self.engine = builder.build_engine(network, config)
        with open(self.engine_path, 'wb') as f:
            f.write(self.engine.serialize())

    def loadNetwork(self):
        # Set the device index
        res = cudart.cudaSetDevice(self.options.deviceIndex)
        if (res[0].value != 0):
            raise Exception("The device Index for the GPU is invalid")
        
        # Read the serialized model from disk
        with open(self.engine_path, "rb") as f:
            serialized_engine = f.read()
        # Create a runtime to deserialize the engine file
        runtime = trt.Runtime(TRT_LOGGER)
        # Create an engine, a representation of the optimized model
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine == None:
            raise Exception("Engine could not be loaded")
        
        # The execution context contains all of the state associated with a particular invocation
        self.context = self.engine.create_execution_context()
        # Create a cuda stream
        self.stream = cuda_call(cudart.cudaStreamCreate())
        # Get tensor names
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = self.engine.get_tensor_shape(binding) if self.profile_idx is None else self.engine.get_tensor_profile_shape(binding, self.profile_idx)[-1]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and self.profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            
            # Size (in number of elements) to allocate
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.options.batch_size
            
            # Type of data we will allocate
            if (self.engine.get_tensor_dtype(binding)) == trt.float32:
                dtype = np.dtype('float32')
            elif (self.engine.get_tensor_dtype(binding)) == trt.float16:
                dtype = np.dtype('float16') # FLOAT 16 IS NOT IMPLEMENTED IN CTYPES, SO THIS WILL PROBABLY RAISE AN EXCEPTION
            elif (self.engine.get_tensor_dtype(binding)) == trt.int32:
                dtype = np.dtype('int32')
            elif (self.engine.get_tensor_dtype(binding)) == trt.int8:
                dtype = np.dtype('int8')
            else:
                dtype = np.dtype('bool')

            # Allocate host and device buffers
            bindingMemory = HostDeviceMem(size, dtype)

            # Append the device buffer to device bindings.
            self.bindings.append(int(bindingMemory.device))

            # Append to the appropriate list.
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append(bindingMemory)
                self.input_dims.append(shape)
                self.input_names.append(binding)
            else:
                self.outputs.append(bindingMemory)
                self.output_dims.append(shape)
                self.output_names.append(binding)
                
    def _do_inference_base(self, execute_async):
        # Transfer input data to the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, self.stream)) for inp in self.inputs]
        # Run inference.
        execute_async()
        # Transfer predictions back from the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, self.stream)) for out in self.outputs]
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        # Return only the host outputs.
        return [out.host.reshape(out_shape) for out, out_shape in zip(self.outputs, self.output_dims)]


    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, inputs, batch_size=1):
        for i in range(len(inputs)):
            np.copyto(self.inputs[i].host, inputs[i].ravel())
        def execute_async():
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream)
        return self._do_inference_base(execute_async)


    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self, inputs):
        for i in range(len(inputs)):
            np.copyto(self.inputs[i].host, inputs[i].ravel())
        def execute_async():
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        return self._do_inference_base(execute_async)
    
    # This function converts the engine options into a string
    def serializeEngineOptions(self, onnxModelPath):

        # If the path doesn't end with a /, we add it
        filenamePos = onnxModelPath.rfind('/') + 1
        if (self.options.out_file_path == "" or self.options.out_file_path[-1] == '/'):
            engineName = self.options.out_file_path + onnxModelPath[filenamePos: onnxModelPath.rfind('.')] + ".engine"
        else:
            engineName = self.options.out_file_path + "/" + onnxModelPath[filenamePos: onnxModelPath.rfind('.')] + ".engine"
        
        # Add the precision to the name
        if (self.options.precision == trt.float32):
            engineName += ".fp32"
        elif (self.options.precision == trt.float16):
            engineName += ".fp16"
        elif (self.options.precision == trt.int8):
            engineName += ".int8"
        else:
            raise Exception("Precision has to be float32 or float16 or int8 (if the model is explicitly quantized)")
        
        # Check if the GPU is allowed, and add its index to the name
        n_gpus = cudart.cudaGetDeviceCount()[1]
        if (self.options.deviceIndex > n_gpus - 1):
            raise Exception("Error, provided device index is out of range!")
        engineName+= "." + str(self.options.deviceIndex)

        # Add batch and DLA
        engineName += "." + str(self.options.batch_size)
        engineName += "." + str(self.options.dlaCore)
        return engineName
    
    # This function prints all the dimensions of inputs and outputs
    def print_data(self):
        for i in range(len(self.inputs)):
            print("Input " + str(i+1) + ": " + self.input_names[i])
            print(self.input_dims[i])
        for i in range(len(self.outputs)):
            print("Output " + str(i+1) + ": " + self.output_names[i])
            print(self.output_dims[i])
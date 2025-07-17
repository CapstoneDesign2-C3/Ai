import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path: str):
    """
    Load a TensorRT engine from file and create an execution context.
    Returns:
        engine (trt.ICudaEngine), context (trt.IExecutionContext)
    """
    # Create runtime and deserialize engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    # Create execution context
    context = engine.create_execution_context()
    return engine, context


def infer_with_trt(context: trt.IExecutionContext, input_tensor: np.ndarray) -> np.ndarray:
    """
    Perform inference with a TensorRT engine execution context.
    Args:
        context: execution context (with engine)
        input_tensor: numpy array matching input binding shape
    Returns:
        output_tensor: numpy array matching output binding shape
    """
    engine = context.engine  # type: trt.ICudaEngine

    # If dynamic shapes, set the binding shape for input (binding 0)
    context.set_binding_shape(0, tuple(input_tensor.shape))

    # Prepare buffers
    bindings = []
    inputs = []
    outputs = []
    for binding_idx in range(engine.num_bindings):
        binding_shape = context.get_binding_shape(binding_idx)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding_idx):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    # Create CUDA stream
    stream = cuda.Stream()

    # Copy input data to host buffer, then to device
    np.copyto(inputs[0][0], input_tensor.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute inference asynchronously
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy predictions back to host
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    # Reshape output to expected shape and return
    output_shape = context.get_binding_shape(engine.num_bindings - 1)
    output = outputs[0][0].reshape(output_shape)
    return output

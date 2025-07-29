import tensorrt as trt
import pycuda.autoinit  # initializes CUDA context

# Logger & builder setup
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX model
with open("yolo11m.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# BuilderConfig: workspace + FP16
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GiB
config.set_flag(trt.BuilderFlag.FP16)

# **FIX 1: Add optimization profile for dynamic shapes**
profile = builder.create_optimization_profile()

# Get input tensor info
input_tensor = network.get_input(0)
input_name = input_tensor.name
print(f"Input tensor name: {input_name}")
print(f"Input tensor shape: {input_tensor.shape}")

# Set optimization profile for dynamic batch size and/or image size
# Adjust these ranges based on your YOLO model's expected inputs
# Common YOLO input formats: [batch, 3, 640, 640] or [batch, 3, height, width]

# For dynamic batch size (1 to 8) with fixed image size 640x640:
profile.set_shape(input_name, 
                 (1, 3, 640, 640),    # min shape
                 (4, 3, 640, 640),    # opt shape  
                 (8, 3, 640, 640))    # max shape

# Alternative: For dynamic image size with fixed batch=1:
# profile.set_shape(input_name,
#                  (1, 3, 320, 320),    # min: 320x320
#                  (1, 3, 640, 640),    # opt: 640x640
#                  (1, 3, 1280, 1280))  # max: 1280x1280

config.add_optimization_profile(profile)

# Build serialized engine
print("Building TensorRT engine... This may take a few minutes.")
serialized = builder.build_serialized_network(network, config)

# **FIX 2: Check if serialization was successful before deserializing**
if serialized is None:
    raise RuntimeError("Failed to build TensorRT engine")

print("Engine built successfully, now deserializing...")

# Write to file first
with open("yolo11m_fp16.engine", "wb") as f:
    f.write(serialized)

print("✅ Built yolo11m_fp16.engine successfully")

# Deserialize to check (optional verification)
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized)

if engine is None:
    raise RuntimeError("Failed to deserialize engine")

# Use newer TensorRT API (num_io_tensors instead of num_bindings)
try:
    num_tensors = engine.num_io_tensors
    print(f"Engine has {num_tensors} I/O tensors")
    for i in range(num_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        print(f"  Tensor {i}: {tensor_name} -> {tensor_shape} ({tensor_mode})")
except AttributeError:
    # Fallback for older TensorRT versions
    num_bindings = engine.num_bindings
    print(f"Engine has {num_bindings} bindings")
    for i in range(num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        print(f"  Binding {i}: {binding_name} -> {binding_shape} ({'input' if is_input else 'output'})")

# Clean up to avoid CUDA errors
del engine
del runtime
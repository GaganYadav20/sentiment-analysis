import tensorflow as tf

# Check if TensorFlow was built with CUDA support
print("Built with CUDA:", tf.test.is_built_with_cuda())

# List all available physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

if gpus:
    for gpu in gpus:
        print("Device details:", gpu)
else:
    print("CUDA/GPU not detected. TensorFlow is using the CPU.")
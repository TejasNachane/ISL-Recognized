import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.config.list_physical_devices('GPU'))  # Should list your GPU

# Ensure GPU memory growth is enabled
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")

        # CUDA version
        cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
        print(f"CUDA version: {cuda_version}")

        # cuDNN version
        cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
        print(f"cuDNN version: {cudnn_version}")

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")
else:
    print("No GPUs detected. Training will use CPU.")

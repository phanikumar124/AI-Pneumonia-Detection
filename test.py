try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is not installed. Install it with 'pip install tensorflow'.")
else:
    print("TensorFlow Version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices("GPU"))
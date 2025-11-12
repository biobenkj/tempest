import os
import sys
import logging
import warnings

def suppress_tensorflow_logging():
    """
    Suppress TensorFlow's excessive logging unless debugging is explicitly enabled
    via '--debug' in sys.argv or TEMPEST_DEBUG=1.
    """
    # Respect global debug flags
    if "--debug" in sys.argv or os.getenv("TEMPEST_DEBUG", "0") == "1":
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        return  # Don't suppress in debug mode

    # Otherwise, suppress TF noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ.setdefault("TF_DISABLE_PLUGIN_REGISTRATION", "1")
    os.environ.setdefault("TF_ENABLE_DEPRECATION_WARNINGS", "0")

    warnings.filterwarnings("ignore")

    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").propagate = False
    except ImportError:
        pass
# tempest/utils/logging_utils.py
import os
import sys
import logging
import warnings

def suppress_tensorflow_logging():
    """
    Fully suppress TensorFlow logging (both Python and C++ backends),
    unless '--debug' in sys.argv or TEMPEST_DEBUG=1.
    """
    # Respect debug flags
    if "--debug" in sys.argv or os.getenv("TEMPEST_DEBUG", "0") == "1":
        return

    # MUST set these before importing tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ.setdefault("TF_DISABLE_PLUGIN_REGISTRATION", "1")
    os.environ.setdefault("TF_ENABLE_DEPRECATION_WARNINGS", "0")

    # Silence Python warnings/loggers
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Try to catch Python-level TF logs *if imported later*
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").propagate = False
    except Exception:
        pass
# models package
try:
    from .cnn_model import build_cnn, load_model, MODEL_PATH
except ImportError:
    # Fallback to PyTorch version if TensorFlow is not available
    from .cnn_model_pytorch import build_cnn, load_model, MODEL_PATH

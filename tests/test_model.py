import os
import tensorflow as tf

def test_model_file_exists():
    assert os.path.exists("model/model.h5") or os.path.exists("model.h5"), \
        "Model file not found!"

def test_model_loading():
    try:
        model = tf.keras.models.load_model("model.h5")
    except Exception as e:
        assert False, f"Model failed to load: {e}"

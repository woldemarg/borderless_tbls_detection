import tensorflow as tf
import config

# %%

TF_MODELS_BASE_PATH = 'http://download.tensorflow.org/models/object_detection/tf2/'


# %%

def download_tf_model(base_path,
                      model_date,
                      model_name,
                      cache_folder):
    model_url = (base_path +
                 model_date +
                 '/' +
                 model_name +
                 '.tar.gz')
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_url,
        untar=True,
        cache_dir=cache_folder
    )
    return model_dir


# %%

# Start the model download.
model_path = download_tf_model(TF_MODELS_BASE_PATH,
                               config.MODEL_DATE,
                               config.MODEL_NAME,
                               config.PRETRAINED_MODELS)

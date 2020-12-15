### sources:
* *models* - https://github.com/tensorflow/models
* *labelImg* - https://github.com/tzutalin/labelImg
* *protoc* - https://github.com/protocolbuffers/protobuf/releases

```
# from <root>
conda create -n <name> python=3.7 tensorflow=2.3 spyder=4.2 tf_slim cython git
conda activate <name>
git clone https://github.com/tensorflow/models.git
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
cd models\research
# from <root>\models\research
protoc object_detection\protos\*.proto --python_out=.
copy object_detection\packages\tf2\setup.py .
python setup.py install
python object_detection\builders\model_builder_tf2_test.py
conda install imutils pdf2image beautifulsoup4 typeguard
pip install tf-image
copy object_detection\model_main_tf2.py ..\..\workspace\.
copy object_detection\exporter_main_v2.py ..\..\workspace\.
cd ..\..
# from <root>
tensorboard --logdir=<log>
set NUM_TRAIN_STEPS=1000
set CHECKPOINT_EVERY_N=1000
set PIPELINE_CONFIG_PATH=<pipeline>
set MODEL_DIR=<log>
set SAMPLE_1_OF_N_EVAL_EXAMPLES=1
set NUM_WORKERS=1
python workspace\model_main_tf2.py \
    --pipeline_config_path=%PIPELINE_CONFIG_PATH% \
    --model_dir=%MODEL_DIR% \
    --checkpoint_every_n=%CHECKPOINT_EVERY_N% \
    --num_workers=%NUM_WORKERS% \
    --num_train_steps=%NUM_TRAIN_STEPS% \
    --sample_1_of_n_eval_examples=%SAMPLE_1_OF_N_EVAL_EXAMPLES% \
    --alsologtostderr
# downgrade numpy to avoid TypeError
# https://github.com/tensorflow/models/issues/2961#issuecomment-663870239
conda install numpy=1.17.4  
python workspace\model_main_tf2.py \
    --pipeline_config_path=%PIPELINE_CONFIG_PATH% \
    --model_dir=%MODEL_DIR% \
    --checkpoint_dir=%MODEL_DIR%
python workspace\exporter_main_v2.py \
    --input_type=image_tensor \
    --pipeline_config_path=%PIPELINE_CONFIG_PATH% \
    --trained_checkpoint_dir=%MODEL_DIR% \
    --output_directory=saved_models\efficientdet_d1_coco17_tpu-32

```

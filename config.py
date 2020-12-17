import os

# %%

MAX_IMG_WIDTH = 768
NUM_AUG = 3
INF_TH = 0.3
PROJECT_FOLDER = r'D:\holomb_learn\tbl_detection'
DEMO_IMG = os.path.sep.join([PROJECT_FOLDER, 'demo', 'images'])
IMG_FOLDER = os.path.sep.join([PROJECT_FOLDER, 'images'])
IMG_UNPROCESSED = os.path.sep.join([IMG_FOLDER, 'unprocessed'])
IMG_PROCESSED = os.path.sep.join([IMG_FOLDER, 'processed'])
PDF_IMAGES = os.path.sep.join([IMG_PROCESSED, 'pdf_images'])
INV_IMAGES = os.path.sep.join([IMG_PROCESSED, 'inv_images'])
ALL_IMAGES = os.path.sep.join([IMG_PROCESSED, 'all_images'])
ALL_ANNOTS = os.path.sep.join([IMG_PROCESSED, 'all_annots'])
TRAIN_SET = os.path.sep.join([IMG_FOLDER, 'splitted', 'train_set'])
TEST_SET = os.path.sep.join([IMG_FOLDER, 'splitted', 'test_set'])
VAL_SET = os.path.sep.join([IMG_FOLDER, 'splitted', 'val_set'])
XML_STYLE = os.path.sep.join([IMG_FOLDER, 'xml_style.xml'])
MODEL_ATTR = os.path.sep.join([PROJECT_FOLDER, 'workspace'])
MODEL_DATA = os.path.sep.join([MODEL_ATTR, 'data'])
LABEL_MAP = os.path.sep.join([MODEL_DATA, 'label_map.pbtxt'])
TRAIN_RECORD = os.path.sep.join([MODEL_DATA, 'train.record'])
TEST_RECORD = os.path.sep.join([MODEL_DATA, 'test.record'])
VAL_RECORD = os.path.sep.join([MODEL_DATA, 'val.record'])
TRAIN_CSV = os.path.sep.join([MODEL_DATA, 'train.csv'])
TEST_CSV = os.path.sep.join([MODEL_DATA, 'test.csv'])
VAL_CSV = os.path.sep.join([MODEL_DATA, 'val.csv'])
MODEL_DATE = '20200711'
MODEL_NAME = 'efficientdet_d1_coco17_tpu-32'
PRETRAINED_MODELS = os.path.sep.join([MODEL_ATTR, 'pretrained_models'])
SAVED_MODEL = os.path.sep.join([PROJECT_FOLDER,
                                'saved_models',
                                MODEL_NAME,
                                'saved_model'])

import os
import lxml.etree as et
import numpy as np
from imutils import paths
from bs4 import BeautifulSoup
import tensorflow as tf
from tf_image.application.augmentation_config import (
    AugmentationConfig,
    AspectRatioAugmentation,
    ColorAugmentation)
from tf_image.application.tools import random_augmentations
import config

# %%

aug_config = AugmentationConfig()
aug_config.color = ColorAugmentation.NONE
aug_config.crop = True
aug_config.distort_aspect_ratio = AspectRatioAugmentation.NONE
aug_config.quality = True
aug_config.erasing = False
aug_config.rotate90 = False
aug_config.rotate_max = 5
aug_config.flip_vertical = False
aug_config.flip_horizontal = False
aug_config.padding_square = False

# %%

image_paths = list(paths.list_images(config.TRAIN_SET))

# %%

for i, image_path in enumerate(image_paths):
    filename = image_path.split(os.path.sep)[-1]
    filename = filename[:filename.rfind('.')]
    ANNOT_PATH = os.path.sep.join([config.TRAIN_SET,
                                  "{}.xml".format(filename)])

    CONTENTS = str(open(ANNOT_PATH).read())
    soup = BeautifulSoup(CONTENTS, 'xml')

    w = int(soup.width.string)
    h = int(soup.height.string)

    image_encoded = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_encoded)

    bboxes = []
    for o in soup.find_all("object"):
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)
        bboxes.append([yMin, xMin, yMax, xMax])

    bboxes = np.array(bboxes, dtype=np.float32)

    bboxes /= np.stack([h, w, h, w])

    for t in range(config.NUM_AUG):

        image_augmented, bboxes_augmented = random_augmentations(image,
                                                                 aug_config,
                                                                 bboxes=tf.constant(bboxes))

        image_augmented_encoded = tf.image.encode_png(image_augmented)

        filename_aug = '{}_aug_{}.jpg'.format(filename, t)        

        tf.io.write_file(os.path.sep.join([config.TRAIN_SET,
                                           filename_aug]),
                         image_augmented_encoded)

        bboxes_abs = ((bboxes_augmented.numpy() *
                       np.stack([tf.shape(image_augmented)[0],
                                 tf.shape(image_augmented)[1],
                                 tf.shape(image_augmented)[0],
                                 tf.shape(image_augmented)[1]]))
                      .astype(int))

        bboxes_abs = bboxes_abs[:, [1, 0, 3, 2]]

        for i, o in enumerate(soup.find_all("object")):
            strings = [s for s in o.bndbox.strings if s.isdigit()]
            for j, s in enumerate(strings):
                s.replace_with(str(bboxes_abs[i, j]))

        soup.filename.string = filename_aug

        xml_string = et.fromstring(soup.decode_contents())
        xml_styles = et.fromstring(str(open(config.XML_STYLE).read()))

        transformer = et.XSLT(xml_styles)
        xml_prettified = transformer(xml_string)



        with open(os.path.sep.join([config.TRAIN_SET,
                                    '{}_aug_{}.xml'.format(filename, t)]),
                  'w') as f:
            f.write(str(xml_prettified))

import os
import re
import cv2
from imutils import paths, resize
import config

# %%

inv_images = paths.list_images(os.path.sep.join([config.IMG_UNPROCESSED, 'inv']))

for image_path in inv_images:
    resized = resize(cv2.imread(image_path), width=config.MAX_IMG_WIDTH)
    fname = re.sub(r'\s+', '_', image_path.split(os.path.sep)[-1])

    if not os.path.exists(config.INV_IMAGES):
        os.makedirs(config.INV_IMAGES)

    cv2.imwrite(os.path.sep.join([config.INV_IMAGES,
                                  fname]),
                resized)

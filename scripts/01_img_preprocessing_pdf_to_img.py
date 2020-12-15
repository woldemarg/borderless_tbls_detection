import os
import re
from imutils import resize
from pdf2image import convert_from_path
import numpy as np
import cv2
import config


# %%

def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


# %%

raw_pdf_dir = os.path.sep.join([config.IMG_UNPROCESSED, 'pdf'])

# %%

kernal = np.ones((2, 2), np.uint8)

for img_path in absolute_file_paths(raw_pdf_dir):
    fname = img_path.split(os.path.sep)[-1]
    fname = fname[:fname.rfind('.')]
    pil_img = convert_from_path(img_path, dpi=300)[0]
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    resized = resize(cv2_img, width=config.MAX_IMG_WIDTH)
    fname_new = re.sub(r'\s+', '_', fname)

    if not os.path.exists(config.PDF_IMAGES):
        os.makedirs(config.PDF_IMAGES)

    cv2.imwrite(os.path.sep.join([config.PDF_IMAGES,
                                  "{}.jpg".format(fname_new)]),
                resized)

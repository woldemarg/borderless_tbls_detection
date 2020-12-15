import os
import random
import cv2
import matplotlib.pyplot as plt
import imutils
import config

# %%

image_paths = list(paths.list_images(config.ORIG_IMAGES))
sample_ids = sorted(random.sample(range(0, len(image_paths)), 25))
sample_data = {}

fig, ax = plt.subplots(5, 5, figsize=(25, 25))
for i, (k, v) in enumerate(sample_data.items()):
    img = cv2.imread(os.path.sep.join([config.TBLS_IMAGES,
                                       '{}.jpg'.format(k)])).copy()
    for bbox in v:
        (xMin, yMin, xMax, yMax) = bbox
        cv2.rectangle(img,
                      (xMin, yMin),
                      (xMax, yMax),
                      (0, 0, 255),
                      2)
    ax.flat[i].axis('off')
    ax.flat[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.tight_layout()
from collections import namedtuple
import os
from bs4 import BeautifulSoup
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths, resize
from imutils.object_detection import non_max_suppression
import config

# %%

detect_fn = tf.saved_model.load(config.SAVED_MODEL)

# %%

demo_paths = list(paths.list_images(config.DEMO_IMG))

Demo = namedtuple('Demo', ['img_path', 'gt_boxes', 'in_boxes'])

demo_data = []
demo_imgs = []

# img_path = demo_paths[2]

for i, img_path in enumerate(demo_paths):

    bboxes = []
    img_array = cv2.imread(img_path)
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = img_tensor[tf.newaxis, ]

    detections = detect_fn(img_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value
                  in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = (detections['detection_classes']
                                       .astype(np.int64))

    h = img_array.shape[0]
    w = img_array.shape[1]

    boxes = detections['detection_boxes']
    proba = detections['detection_scores']

    idxs = np.where(proba >= config.INF_TH)
    boxes = boxes[idxs]
    proba = proba[idxs]

    boxes *= (np.stack([h, w, h, w])).astype(int)
    bboxes.append(boxes)

    boxes = non_max_suppression(boxes, proba)
    bboxes.append(boxes)

    fname = img_path.split(os.path.sep)[-1]
    fname = fname[:fname.rfind('.')]
    ANNOT_PATH = os.path.sep.join([config.TEST_SET,
                                   "{}.xml".format(fname)])

    CONTENTS = str(open(ANNOT_PATH).read())
    soup = BeautifulSoup(CONTENTS, 'xml')

    gt_boxes = []
    for o in soup.find_all("object"):
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)
        gt_boxes.append((yMin, xMin, yMax, xMax))

    demo_data.append(Demo(img_path, np.array(gt_boxes), boxes))

    for j in range(2):
        img_copy = img_array.copy()
        for box in bboxes[j]:
            startY, startX, endY, endX = box
            cv2.rectangle(img_copy,
                          (startX, startY),
                          (endX, endY),
                          (0, 0, 255),
                          2)
        demo_imgs.append(img_copy)

# %%

fig, ax = plt.subplots(5, 2, figsize=(5, 15))
for i, im in enumerate(demo_imgs):
    ax.flat[i].axis('off')
    ax.flat[i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.tight_layout()


# %%

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou


# %%


fig, ax = plt.subplots(1, 5, figsize=(15, 5))

for i, d in enumerate(demo_data):
    image = cv2.imread(d.img_path)

    cv2.rectangle(image,
                  tuple(d.gt_boxes[0][[1, 0]]),
                  tuple(d.gt_boxes[0][[3, 2]]),
                  (0, 255, 0),
                  2)
    cv2.rectangle(image,
                  tuple(d.in_boxes[0][[1, 0]]),
                  tuple(d.in_boxes[0][[3, 2]]),
                  (0, 0, 255),
                  2)

    iou_val = bb_intersection_over_union(d.gt_boxes[0], d.in_boxes[0])

    y = d.in_boxes[0][0] - 10 if startY - 10 > 10 else d.in_boxes[0][0] + 10

    cv2.putText(image,
                'IoU: {:.2f}'.format(iou_val),
                (d.in_boxes[0][1], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2)
    ax[i].axis('off')
    ax[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.tight_layout()

# %%

dem = demo_data[3]
dem_wdth = dem.in_boxes[0][3] - dem.in_boxes[0][1]
dem_hght = dem.in_boxes[0][2] - dem.in_boxes[0][0]
dem_xmin = dem.in_boxes[0][1]
dem_ymin = dem.in_boxes[0][0]

dem_image = cv2.imread(dem.img_path)
tbl_image = dem_image[dem_ymin: dem_ymin + dem_hght,
                      dem_xmin: dem_xmin + dem_wdth]

plt.imshow(tbl_image)

tbl_gray = cv2.cvtColor(tbl_image, cv2.COLOR_BGR2GRAY)
tbl_thresh_bin = cv2.threshold(tbl_gray, 127, 255, cv2.THRESH_BINARY)[1]

plt.imshow(tbl_thresh_bin, cmap='gray')

R = 2.5
tbl_resized = resize(tbl_thresh_bin, width=int(tbl_image.shape[1] // R))
plt.imshow(tbl_resized, cmap='gray')

# %%


def get_dividers(image, axis):
    blank_lines = np.where(np.all(image == 255, axis=axis))[0]
    filtered_idx = np.where(np.diff(blank_lines) != 1)[0]
    return blank_lines[filtered_idx]

# %%


img_copy = cv2.merge([tbl_resized, tbl_resized, tbl_resized])
img_copy = tbl_image.copy()

for i in column_dividers:
    i = i * R
    start_point = (i, 0)
    end_point = (i, tbl_image.shape[1])
    cv2.line(img_copy, start_point, end_point, (255, 0, 0), 1)

plt.figure(figsize=(10, 10))
plt.imshow(img_copy)

whitespace_сols = np.where(np.all(tbl_resized == 255, axis=1))[0]
rightmost_whitespace_cols = np.where(np.diff(whitespace_сols) != 1)[0]
column_dividers = np.append(whitespace_сols[rightmost_whitespace_cols],
                            whitespace_сols[-1])


img_copy = cv2.merge([tbl_resized, tbl_resized, tbl_resized])
# img_copy = tbl_image.copy()

for i in column_dividers:
    start_point = (0, i)
    end_point = (tbl_resized.shape[1], i)
    cv2.line(img_copy, start_point, end_point, (255, 0, 0), 1)


plt.figure(figsize=(10, 10))
plt.imshow(img_copy)

tbl_bw = cv2.adaptiveThreshold(tbl_gray,
                               255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               21,
                               15)

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models.research.object_detection.utils import (
    label_map_util)
from models.research.object_detection.utils import (
    visualization_utils)
from models.research.object_detection.data_decoders.tf_example_decoder import (
    TfExampleDecoder)
from imutils.object_detection import non_max_suppression
import config

# %%

detect_fn = tf.saved_model.load(config.SAVED_MODEL)

# %%

inference_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=r'demo\test_selected',
    # image_size=(640, 640),
    batch_size=1,
    shuffle=False,
    label_mode=None
  )
  # Numpy version of the dataset.
  inference_ds_numpy = list(inference_ds.as_numpy_iterator())


def tensors_from_tfrecord(
    tfrecords_filename,
    tfrecords_num,
    dtype=tf.float32
):
    decoder = TfExampleDecoder()
    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename)
    images = []

    for raw_record in raw_dataset.take(tfrecords_num):
        example = decoder.decode(raw_record)
        image = example['image']
        image = tf.cast(image, dtype=dtype)
        images.append(image)

    return images


def test_detection(tfrecords_filename, tfrecords_num, detect_fn):
    image_tensors = tensors_from_tfrecord(
        tfrecords_filename,
        tfrecords_num,
        dtype=tf.uint8
    )

    for image_tensor in image_tensors:
        image_np = image_tensor.numpy()
        input_tensor = tf.expand_dims(image_tensor, 0)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value
                      in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = (detections['detection_classes']
                                           .astype(np.int64))
        image_np_with_detections = image_np.astype(int).copy()
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=.3,
            agnostic_mode=False
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np_with_detections)
    plt.show()


test_detection(
    tfrecords_filename=config.TEST_RECORD,
    tfrecords_num=5,
    detect_fn=detect_fn
)


image_tensors = tensors_from_tfrecord(
    config.TEST_RECORD,
    15,
    dtype=tf.uint8
)

image_tensor = image_tensors[13]
image_np = image_tensor.numpy()

# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = tf.expand_dims(image_tensor, 0)

detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))

detections = {key: value[0, :num_detections].numpy() for key, value
              in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = (detections['detection_classes']
                                   .astype(np.int64))

h = image_np.shape[0]
w = image_np.shape[1]

boxes = detections['detection_boxes']
proba = detections['detection_scores']

# filter indexes by enforcing a minimum prediction
# probability be met
idxs = np.where(proba >= 0.6)
boxes = boxes[idxs]
proba = proba[idxs]

# clone the original image so that we can draw on it
clone = image_np.copy()

# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
    # draw the bounding box, label, and probability on the image
    box *= (np.stack([h, w, h, w])).astype(int)
    startY, startX, endY, endX = box
    cv2.rectangle(clone,
                  (startX, startY),
                  (endX, endY),
                  (0, 255, 0),
                  2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "borderless: {:.2f}%".format(prob * 100)
    cv2.putText(clone, text, (int(startX), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# run non-maxima suppression on the bounding boxes
boxIdxs = non_max_suppression(boxes, proba)
clone2 = image_np.copy()

# loop over the bounding box indexes
for row in boxIdxs:
    # draw the bounding box, label, and probability on the image
    (startY, startX, endY, endX) = row
    cv2.rectangle(clone2, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)

fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(clone)
ax[1].imshow(clone2)

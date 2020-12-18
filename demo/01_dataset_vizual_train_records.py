import random
import tensorflow as tf
import numpy as np
from google.protobuf import text_format
import matplotlib.pyplot as plt
from models.research.object_detection.utils import (
    visualization_utils)
from models.research.object_detection.protos import (
    string_int_label_map_pb2)
from models.research.object_detection.data_decoders.tf_example_decoder import (
    TfExampleDecoder)
import config

# %%

decoder = TfExampleDecoder(
    label_map_proto_file=config.LABEL_MAP,
    use_display_name=False)

label_map_proto = string_int_label_map_pb2.StringIntLabelMap()

with tf.io.gfile.GFile(config.LABEL_MAP, 'r') as f:
    text_format.Merge(f.read(), label_map_proto)
    class_dict = {}

for entry in label_map_proto.item:
    class_dict[entry.id] = {'name': entry.name}

raw_dataset = list(tf.data.TFRecordDataset(config.TRAIN_RECORD))
random.shuffle(raw_dataset)

fig, ax = plt.subplots(2, 5, figsize=(15, 5))

for i, raw_record in enumerate(raw_dataset[:10]):

    example = decoder.decode(raw_record)

    image = example['image'].numpy()
    boxes = example['groundtruth_boxes'].numpy()
    classes = example['groundtruth_classes'].numpy()

    scores = np.ones(boxes.shape[0])

    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        class_dict,
        max_boxes_to_draw=None,
        use_normalized_coordinates=True
    )

    ax.flat[i].axis('off')
    ax.flat[i].imshow(image)

fig.tight_layout()

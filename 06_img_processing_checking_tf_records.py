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

# Count the number of examples in the dataset.
def count_tfrecords(tfrecords_filename):
    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename)
    # Keep in mind that the list() operation might be
    # a performance bottleneck for large datasets.
    return len(list(raw_dataset))

TRAIN_RECORDS_NUM = count_tfrecords(config.TRAIN_RECORD)
TEST_RECORDS_NUM = count_tfrecords(config.TEST_RECORD)

print('[INFO] TRAIN_RECORDS_NUM: ', TRAIN_RECORDS_NUM)
print('[INFO] TEST_RECORDS_NUM:  ', TEST_RECORDS_NUM)

# %%

# Visualize the TFRecord dataset.
def visualize_tfrecords(tfrecords_filename,
                        label_map=None,
                        print_num=1):

    decoder = TfExampleDecoder(
        label_map_proto_file=label_map,
        use_display_name=False
    )

    if label_map is not None:
        label_map_proto = string_int_label_map_pb2.StringIntLabelMap()

        with tf.io.gfile.GFile(label_map, 'r') as f:
            text_format.Merge(f.read(), label_map_proto)
            class_dict = {}

            for entry in label_map_proto.item:
                class_dict[entry.id] = {'name': entry.name}

    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename)

    for raw_record in raw_dataset.take(print_num):
        example = decoder.decode(raw_record)

        image = example['image'].numpy()
        boxes = example['groundtruth_boxes'].numpy()
        # filename = example['filename']
        # confidences = example['groundtruth_image_confidences']
        # area = example['groundtruth_area']
        # image_classes = example['groundtruth_image_classes']
        # weights = example['groundtruth_weights']
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

        plt.figure(figsize=(8, 8))
        plt.imshow(image)

    plt.show()


# %%

# %matplotlib inline

# Visualizing the training TFRecord dataset.
visualize_tfrecords(
    tfrecords_filename=config.TRAIN_RECORD,
    label_map=config.LABEL_MAP,
    print_num=15
)

import os
import io
import glob
from collections import namedtuple
import xml.etree.ElementTree as ET
import tensorflow as tf
import pandas as pd
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
import config


# %%

# Convers labels from XML format to CSV.
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename',
                   'width',
                   'height',
                   'class',
                   'xmin',
                   'ymin',
                   'xmax',
                   'ymax']

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x
            in zip(gb.groups.keys(), gb.groups)]


# Creates a TFRecord.
def create_tf_example(group, path, label_map_dict):
    with (tf.io.gfile.GFile(os.path.join(path,
                                         '{}'.format(group.filename)),
                            'rb')) as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def dataset_to_tfrecord(
    images_dir,
    xmls_dir,
    label_map_path,
    output_path,
    csv_path=None
):

    if not os.path.exists(output_path.rsplit(os.path.sep, 1)[0]):
        os.makedirs(config.TF_RECORDS)

    if not os.path.exists(csv_path.rsplit(os.path.sep, 1)[0]):
        os.makedirs(config.LABELS_CSV)

    label_map = label_map_util.load_labelmap(label_map_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    # tfrecord_writer = tf1.python_io.TFRecordWriter(output_path)
    tfrecord_writer = tf.io.TFRecordWriter(output_path)
    images_path = os.path.join(images_dir)
    csv_examples = xml_to_csv(xmls_dir)
    grouped_examples = split(csv_examples, 'filename')

    for group in grouped_examples:
        tf_example = create_tf_example(group, images_path, label_map_dict)
        tfrecord_writer.write(tf_example.SerializeToString())

    tfrecord_writer.close()

    print('[INFO] Successfully created the TFRecord file: {}'
          .format(output_path))

    if csv_path is not None:
        csv_examples.to_csv(csv_path, index=None)
        print('[INFO] Successfully created the CSV file: {}'.format(csv_path))


# %%

# Generate a TFRecord for train dataset.
dataset_to_tfrecord(
    images_dir=config.TRAIN_SET,
    xmls_dir=config.TRAIN_SET,
    label_map_path=config.LABEL_MAP,
    output_path=config.TRAIN_RECORD,
    csv_path=config.TRAIN_CSV
)

# Generate a TFRecord for test dataset.
dataset_to_tfrecord(
    images_dir=config.TEST_SET,
    xmls_dir=config.TEST_SET,
    label_map_path=config.LABEL_MAP,
    output_path=config.TEST_RECORD,
    csv_path=config.TEST_CSV
)

# # Generate a TFRecord for val dataset.
# dataset_to_tfrecord(
#     images_dir=config.VAL_SET,
#     xmls_dir=config.VAL_SET,
#     label_map_path=config.LABEL_MAP,
#     output_path=config.VAL_RECORD,
#     csv_path=config.VAL_CSV
# )

import re
import os
import math
import shutil
import random
import config


# %%

def partition_dataset(
        images_dir,
        xml_labels_dir,
        train_dir,
        test_dir,
        val_dir,
        train_ratio,
        test_ratio,
        val_ratio,
        copy_xml):

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    images = [f for f in os.listdir(images_dir)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$',
                           f,
                           re.IGNORECASE)]

    num_images = len(images)

    num_train_images = math.ceil(train_ratio * num_images)
    num_test_images = math.ceil(test_ratio * num_images)
    num_val_images = math.ceil(val_ratio * num_images)

    print('[INFO] Intended split')
    print(f'[INFO]  train: {num_train_images}/{num_images} images')
    print(f'[INFO]  test: {num_test_images}/{num_images} images')
    print(f'[INFO]  val: {num_val_images}/{num_images} images')

    actual_num_train_images = 0
    actual_num_test_images = 0
    actual_num_val_images = 0

    def copy_random_images(num_images, dest_dir):
        copied_num = 0

        if not num_images:
            return copied_num

        # while images:
        for i in range(num_images):
            if not images:
                break

            idx = random.randint(0, len(images)-1)
            filename = images[idx]
            shutil.copyfile(os.path.join(images_dir, filename),
                            os.path.join(dest_dir, filename))

            if copy_xml:
                xml_filename = os.path.splitext(filename)[0]+'.xml'
                shutil.copyfile(os.path.join(xml_labels_dir, xml_filename),
                                os.path.join(dest_dir, xml_filename))

            images.remove(images[idx])
            copied_num += 1

        return copied_num

    actual_num_train_images = copy_random_images(num_train_images, train_dir)
    actual_num_test_images = copy_random_images(num_test_images, test_dir)
    actual_num_val_images = copy_random_images(num_val_images, val_dir)

    print('\n', '[INFO] Actual split')
    print(f'[INFO]  train: {actual_num_train_images}/{num_images} images')
    print(f'[INFO]  test: {actual_num_test_images}/{num_images} images')
    print(f'[INFO]  val: {actual_num_val_images}/{num_images} images')


# %%

partition_dataset(
    images_dir=config.ALL_IMAGES,
    train_dir=config.TRAIN_SET,
    test_dir=config.TEST_SET,
    val_dir=config.VAL_SET,
    xml_labels_dir=config.ALL_ANNOTS,
    train_ratio=0.85,
    test_ratio=0.15,
    val_ratio=0.0,
    copy_xml=True
)

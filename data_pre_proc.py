import os

import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import img_to_array


def resize_bbox(row, img_size):
    """
    Transform bbox coordinates according to resized image
    """
    # compute scaling factor
    x_scale = img_size[0] / row["Width"]
    y_scale = img_size[1] / row["Height"]

    # transform upper left box coordinates
    x0 = int(np.round(row["Roi.X1"] * x_scale))
    y0 = int(np.round(row["Roi.Y1"] * y_scale))
    # transform lower right box coordinates
    x1 = int(np.round(row["Roi.X2"] * x_scale))
    y1 = int(np.round(row["Roi.Y2"] * y_scale))

    return x0, y0, x1, y1


def read_data(df, filename, img_size=(30, 30)):
    """
    Read and preprocess GTSRB images and bboxes and save ndarrays to disk
    """
    images = []
    bbox = []
    labels = []
    for idx, row in df.iterrows():
        # get class label
        labels.append(row["ClassId"])

        # read image and resize it
        img = cv2.imread(os.path.join("data", row["Path"]))
        img = cv2.resize(img, img_size)
        images.append(img_to_array(img))

        # resize bbox according to new image size
        bbox.append(resize_bbox(row, img_size))

    # convert to float ndarrays and normalize image data
    images = np.array(images, dtype="float") / 255.
    bbox = np.array(bbox, dtype="float")
    labels = np.array(labels, dtype="float")

    # save data to disk
    with open(os.path.join("data", filename), "wb") as f:
        np.save(f, images)
        np.save(f, bbox)
        np.save(f, labels)

    return images, bbox, labels


def load_data(path):
    """
    Load ndarrays from disk
    """
    with open(path, "rb") as f:
        images = np.load(f)
        bbox = np.load(f)
        labels = np.load(f)

    return images, bbox, labels


if __name__ == '__main__':
    train = pd.read_csv("data/Train.csv")
    test = pd.read_csv("data/Test.csv")

    read_data(train, "train.npy")
    read_data(test, "test.npy")

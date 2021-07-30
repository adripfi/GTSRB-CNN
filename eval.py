import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from data import load_data
from train import get_model, r2_keras

tf.compat.v1.enable_eager_execution()
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def plot_grid(images, bbox_true, bbox_pred, nrow=5, ncol=5):
    """
    Plot image grid and draw predicted as well ground truth bboxes
    """
    fig, axs = plt.subplots(nrow, ncol)
    for i, ax in enumerate(fig.axes):
        # convert image to RGB with values between 0 and 255
        img = images[i, :] * 255.
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB)
        # draw bboxes
        img = cv2.rectangle(img, tuple(bbox_true[i][:2].astype("int")), tuple(bbox_true[i][2:].astype("int")),
                            color=(255, 0, 0))
        img = cv2.rectangle(img, tuple(bbox_pred[i][:2].astype("int")), tuple(bbox_pred[i][2:].astype("int")),
                            color=(0, 255, 0))
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # load and compile model
    model = get_model()
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss={"classification": loss, "regression": "mse"},
                  metrics={"classification": "acc", "regression": r2_keras},
                  loss_weights={"classification": 5, "regression": 1})

    # load pre trained weights
    model.load_weights("weights.h5")

    # evaluate test data
    img, rois, label = load_data("data/test.npy")
    labels_pred, bbox_pred = model.predict(img)
    labels_pred_bool = np.argmax(labels_pred, axis=1)

    print(classification_report(label, labels_pred_bool))
    mse = np.mean(np.concatenate((bbox_pred - rois) ** 2))
    print("MSE:", mse)

    plot_grid(img, rois, bbox_pred)


if __name__ == '__main__':
    main()

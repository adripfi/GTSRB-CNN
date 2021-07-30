import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import load_data

tf.compat.v1.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import he_normal, zeros, RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as keras_be

# Fix for internal CUDA error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_model(img_size=(30, 30)):
    """
    Construct and return model
    """
    input_layer = Input(shape=(img_size[0], img_size[1], 3,), dtype='float32')
    cv1 = Conv2D(filters=32, kernel_size=5, activation=relu, kernel_initializer=he_normal(), bias_initializer=zeros())(input_layer)
    cv2 = Conv2D(filters=64, kernel_size=3, activation=relu, kernel_initializer=he_normal(), bias_initializer=zeros())(cv1)
    mp1 = MaxPool2D(pool_size=(2, 2))(cv2)
    do1 = Dropout(0.25)(mp1)
    cv3 = Conv2D(filters=64, kernel_size=3, activation=relu, kernel_initializer=he_normal(), bias_initializer=zeros())(do1)
    mp2 = MaxPool2D(pool_size=(2, 2))(cv3)
    do2 = Dropout(0.25)(mp2)

    flat = Flatten()(do2)
    fc1 = Dense(units=256, activation=relu, kernel_initializer=he_normal(), bias_initializer=zeros(), kernel_regularizer=l2(1e-3))(flat)
    do3 = Dropout(0.5)(fc1)
    cf = Dense(units=43, activation=None, name="classification", kernel_regularizer=l2(1e-4))(do3)
    reg = Dense(units=4, activation='linear', name="regression", kernel_initializer=RandomNormal(), kernel_regularizer=l2(0.1))(do3)

    return Model(inputs=input_layer, outputs=[cf, reg])


def r2_keras(y_true, y_pred):
    """
    Coefficient of determination for regression model
    """
    ss_res = keras_be.sum(keras_be.square(y_true - y_pred))
    ss_tot = keras_be.sum(keras_be.square(y_true - keras_be.mean(y_true)))

    return 1 - ss_res / (ss_tot + keras_be.epsilon())



def main():
    img, rois, label = load_data("data/train.npy")

    trainX, testX, trainRoiY, testRoiY, trainLabelY, testLabelY= train_test_split(img, rois, label,
                                                                                  test_size=0.2, random_state=42)
    model = get_model()

    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss={"classification": loss, "regression": "mse"},
                  metrics={"classification": "acc", "regression": r2_keras},
                  loss_weights={"classification": 5, "regression": 1})

    history = model.fit(x=trainX,
                        y={"regression": trainRoiY, "classification": trainLabelY},
                        validation_data=(testX, {"regression": testRoiY, "classification": testLabelY}),
                        epochs=200, batch_size=32, verbose=1)

    model.save_weights("weights.h5")


if __name__ == '__main__':
    main()



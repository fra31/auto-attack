#%%
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.insert(0, '..')

from autoattack import AutoAttack, utils_tf2


#%%
class mnist_loader:
    def __init__(self):

        self.n_class = 10
        self.dim_x   = 28
        self.dim_y   = 28
        self.dim_z   = 1
        self.img_min = 0.0
        self.img_max = 1.0
        self.epsilon = 0.3

    def download(self):
        (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

        trainX = trainX.astype(np.float32)
        testX  = testX.astype(np.float32)

        # ont-hot
        trainY = tf.keras.utils.to_categorical(trainY, self.n_class)
        testY  = tf.keras.utils.to_categorical(testY , self.n_class)

        # get validation sets
        training_size = 55000
        validX = trainX[training_size:,:]
        validY = trainY[training_size:,:]

        trainX = trainX[:training_size,:]
        trainY = trainY[:training_size,:]

        # expand dimesion
        trainX = np.expand_dims(trainX, axis=3)
        validX = np.expand_dims(validX, axis=3)
        testX  = np.expand_dims(testX , axis=3)

        return trainX, trainY, validX, validY, testX, testY

    def get_raw_data(self):
        return self.download()

    def get_normalized_data(self):
        trainX, trainY, validX, validY, testX, testY = self.get_raw_data()
        trainX = trainX / 255.0 * (self.img_max - self.img_min) + self.img_min
        validX = validX / 255.0 * (self.img_max - self.img_min) + self.img_min
        testX  = testX  / 255.0 * (self.img_max - self.img_min) + self.img_min
        trainY = trainY
        validY = validY
        testY  = testY
        return trainX, trainY, validX, validY, testX, testY

#%%
def mnist_model():
    # declare variables
    model_layers = [ tf.keras.layers.Input(shape=(28,28,1), name="model/input"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c1"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c2"),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf/p1"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c3"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c4"),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf/p2"),
                        tf.keras.layers.Flatten(name="clf/f1"),
                        tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', name="clf/d1"),
                        tf.keras.layers.Dense(10 , activation=None  , kernel_initializer='he_normal', name="clf/d2"),
                        tf.keras.layers.Activation('softmax', name="clf_output")
                    ]

    # clf_model
    clf_model = tf.keras.Sequential()
    for ii in model_layers:
        clf_model.add(ii)

    clf_model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    clf_model.summary()

    return clf_model

#%%
def arg_parser(parser):

    parser.add_argument("--path" , dest ="path", type=str, default='./autoattack/examples/tf_model.weight.h5', help="path of tf.keras model's wieghts")
    args, unknown = parser.parse_known_args()
    if unknown:
        msg = " ".join(unknown)
        print('[Warning] Unrecognized arguments: {:s}'.format(msg) )

    return args

#%%
if __name__ == '__main__':

    # get arguments
    parser = ArgumentParser()
    args = arg_parser(parser)

    # MODEL PATH
    MODEL_PATH = args.path

    # init tf/keras
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # load data
    batch_size = 1000
    epsilon = mnist_loader().epsilon
    _, _, _, _, testX, testY = mnist_loader().get_normalized_data()

    # convert to pytorch format
    testY = np.argmax(testY, axis=1)
    torch_testX = torch.from_numpy( np.transpose(testX, (0, 3, 1, 2)) ).float().cuda()
    torch_testY = torch.from_numpy( testY ).float()

    # load model from saved weights
    print('[INFO] MODEL_PATH: {:s}'.format(MODEL_PATH) )
    tf_model = mnist_model()
    tf_model.load_weights(MODEL_PATH)

    # remove 'softmax layer' and put it into adapter
    atk_model = tf.keras.models.Model(inputs=tf_model.input, outputs=tf_model.get_layer(index=-2).output) 
    atk_model.summary()
    model_adapted = utils_tf2.ModelAdapter(atk_model)

    # run attack
    adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, version='standard', is_tf_model=True)
    x_adv, y_adv = adversary.run_standard_evaluation(torch_testX, torch_testY, bs=batch_size, return_labels=True)
    np_x_adv = np.moveaxis(x_adv.cpu().numpy(), 1, 3)
    np.save("./output/mnist_adv.npy", np_x_adv)

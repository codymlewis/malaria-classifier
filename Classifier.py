#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Use a deep neural network to identify malaria infections from images, and show some hyper-parameter tuning

Author: Cody Lewis
Date: 2019-05-19
'''

import os
import argparse

from tensorflow import keras


def create_model(rep_cap=1, learning_rate=0.0002):
    '''
    Create a DNN based on Mobile Net.
    '''
    rep_cap /= 3  # have a third of the neurons

    input_shape = (3, 224, 224) if keras.backend.image_data_format() == "channels_first" else (224, 224, 3)
    input_img = keras.layers.Input(shape=input_shape)
    cur_layer = keras.layers.Convolution2D(int(32 * rep_cap), (3, 3), strides=(2, 2), padding="same")(input_img)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(32 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(64 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(64 * rep_cap), (3, 3), strides=(2, 2), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(128 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(128 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(128 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(128 * rep_cap), (3, 3), strides=(2, 2), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(256 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(256 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(256 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(256 * rep_cap), (3, 3), strides=(2, 2), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(512 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)

    for _ in range(5):
        cur_layer = keras.layers.SeparableConvolution2D(
            int(512 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
        cur_layer = keras.layers.BatchNormalization()(cur_layer)
        cur_layer = keras.layers.Activation("relu")(cur_layer)
        cur_layer = keras.layers.Convolution2D(int(512 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
        cur_layer = keras.layers.BatchNormalization()(cur_layer)
        cur_layer = keras.layers.Activation("relu")(cur_layer)

    cur_layer = keras.layers.SeparableConvolution2D(int(512 * rep_cap), (3, 3), strides=(2, 2), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(1024 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.SeparableConvolution2D(int(1024 * rep_cap), (3, 3), strides=(2, 2), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)
    cur_layer = keras.layers.Convolution2D(int(1024 * rep_cap), (3, 3), strides=(1, 1), padding="same")(cur_layer)
    cur_layer = keras.layers.BatchNormalization()(cur_layer)
    cur_layer = keras.layers.Activation("relu")(cur_layer)

    cur_layer = keras.layers.GlobalAveragePooling2D()(cur_layer)
    out_layer = keras.layers.Dense(2, activation="softmax")(cur_layer)
    model = keras.Model(input_img, out_layer, name="mobile-net-modded")

    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

    return model


def train(model, train_dir="cell_images", model_name="DNN", epochs=10_000, batch_size=50,
          nb_train_samples=1000, nb_test_samples=200):
    '''
    Train the neural network using the specified data.
    '''
    tensorboard = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = keras.callbacks.ModelCheckpoint("data/DNN/" + model_name + ".{epoch:03d}.h5", monitor='val_loss',
                                                   verbose=1, save_best_only=False, save_weights_only=False,
                                                   mode='auto', period=1)
    if not os.path.exists("data/DNN"):
        os.makedirs("data/DNN")
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical"
    )

    model.fit_generator(train_generator, steps_per_epoch=nb_train_samples, epochs=epochs,
                        validation_steps=nb_test_samples, verbose=1, callbacks=[tensorboard, checkpointer])


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Use a deep neural network to determine malaria infections from images")
    PARSER.add_argument("-t", "--train", dest="train", action="store_const", const=True, default=False,
                        help="Train the model")
    PARSER.add_argument("-e", "--epochs", dest="epochs", type=int, action="store", default=5,
                        help="The number of epochs to train for")
    PARSER.add_argument("-r", "--rep-cap", dest="rep_cap", type=float, action="store", default=1.0,
                        help="Value for the representative capacity")
    PARSER.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, action="store", default=0.0002,
                        help="Value for the learning rate")
    PARSER.add_argument("-l", "--load", dest="load_model", metavar="MODEL_FILE", action="store", default=None, nargs=1,
                        help="Load a model")

    ARGS = PARSER.parse_args()

    if ARGS.load_model:
        MODEL = keras.models.load_model(ARGS.load_model[0])
    else:
        MODEL = create_model(ARGS.rep_cap, ARGS.learning_rate)

    if ARGS.train:
        print(f"Training for {ARGS.epochs} epochs, with a representative capacity of {ARGS.rep_cap}, and a learning rate of {ARGS.learning_rate}.")
        MODEL.summary()
        train(MODEL, epochs=ARGS.epochs, model_name=f"DNN-rc{ARGS.rep_cap}-lr{ARGS.learning_rate}")
    else:
        PARSER.print_help()

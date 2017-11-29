# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:17:08 2017

@author: gshai
"""

from os import walk

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras import applications, optimizers

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


IMG_WIDTH, IMG_HEIGHT = 224, 224

PATH_TRAIN = 'dataset/train'
PATH_VAL = 'dataset/val'
PATH_TEST = 'dataset/test'

PATH_PREPROC_TRAIN = 'preprocessed/train_vgg16_224'
PATH_PREPROC_VAL = 'preprocessed/val_vgg16_224'

TOP_MODEL_WEIGHTS_PATH = 'top_weights.h5'
FINAL_MODEL_WEIGHTS_PATH = 'final_weights_72_0.22.h5'


#%%

def get_filenames(path):
    '''get_filenames'''
    _, plant_names, _ = next(walk(path))

    plants = dict()

    for name in plant_names:
        _, _, filenames = next(walk(path + '/' + name))
        plants[name] = np.array(filenames)

    return plants


def get_images(files, input_path):
    '''get_images'''
    img_arrays_list = list()

    for imgfile in files:
        imgpath = input_path + '/' + imgfile
        img = load_img(imgpath, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img) / 255
        img_array = img_array[np.newaxis]

        img_arrays_list.append(img_array)

    img_arrays = np.concatenate(img_arrays_list, axis=0)

    return img_arrays


def save_bottleneck_features(model, plants_files, input_path, output_path):
    '''save_bottleneck_features'''
    print("saving...")

    plant_arrays = list()

    for idx, (plant, files) in enumerate(plants_files.items()):

        img_arrays = get_images(files, input_path + '/' + plant)

        bottleneck_features = model.predict(
            img_arrays, batch_size=1, verbose=1)

        plant_arrays.append((bottleneck_features, idx, plant))

    for (plant_features, plant_label, plant_name) in plant_arrays:
        filename = output_path + \
            '/' + plant_name + '_' + str(plant_label) + '.npy'
        np.save(open(filename, 'wb'), plant_features)


def calculate_features(model):
    '''calculate_features'''

    plants_train = get_filenames(PATH_TRAIN)
    plants_val = get_filenames(PATH_VAL)

    save_bottleneck_features(
        model, plants_train,
        input_path=PATH_TRAIN,
        output_path=PATH_PREPROC_TRAIN)

    save_bottleneck_features(
        model, plants_val,
        input_path=PATH_VAL,
        output_path=PATH_PREPROC_VAL)


#%%

def load_features(path_to_processed):
    '''load_features'''

    _, _, filenames = next(walk(path_to_processed))

    features_array = list()
    labels_array = list()

    plant_coding = dict()

    for idx, name in enumerate(filenames):
        features = np.load(open(path_to_processed + '/' + name, 'rb'))
        labels = idx * np.ones(features.shape[0])

        features_array.append(features)
        labels_array.append(labels)
        plant_coding[str(idx)] = name.split('_')[0]

    features_array = np.concatenate(features_array, axis=0)
    labels_array = np.concatenate(labels_array, axis=0)
    return features_array, labels_array


def train_top_model(model):
    '''train_top_model'''
    epochs = 30
    batch_size = 32

    features_train, labels_train = load_features(PATH_PREPROC_TRAIN)
    features_val, labels_val = load_features(PATH_PREPROC_VAL)

    one_hot = OneHotEncoder(sparse=False)
    labels_train = one_hot.fit_transform(labels_train[:, np.newaxis])
    labels_val = one_hot.fit_transform(labels_val[:, np.newaxis])

    optimizer = optimizers.Adam(lr=1e-4)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    try:
        print("fit model")
        model.fit(
            x=features_train,
            y=labels_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(features_val, labels_val))
    except KeyboardInterrupt:
        print("\nabort fitting")
    finally:
        print("save weights")
        model.save_weights(TOP_MODEL_WEIGHTS_PATH)


#%%

def create_data_generators(batch_size):
    '''create_data_generators'''
    train_datagen = ImageDataGenerator(
        rotation_range=180,
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='reflect')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        PATH_TRAIN,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    val_generator = test_datagen.flow_from_directory(
        PATH_VAL,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    return train_generator, val_generator


def fine_tune_final_model(model):
    '''fine_tune_final_model'''
    print("fine tuning")

    epochs = 200
    batch_size = 8

    train_generator, val_generator = create_data_generators(batch_size)

    # configure the model the model
    for layer in model.layers[:-5]:
        layer.trainable = False

    optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # load previous weights
    model.load_weights(FINAL_MODEL_WEIGHTS_PATH)

    # fine-tune the model
    save_weights_checkpoint = ModelCheckpoint(
        'final_weights_{epoch:02d}_{val_loss:.2f}.h5',
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        period=1)

    try:
        print("fit model")
        model.fit_generator(
            train_generator,
            steps_per_epoch=4042 // batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[save_weights_checkpoint],
            validation_data=val_generator,
            validation_steps=708 // batch_size)
    except KeyboardInterrupt:
        print("\nabort fitting")


#%%

def demo():
    '''demo'''
    print('Demo from validation set')
    _, plant_names, _ = next(walk(PATH_VAL))

    model = load_final_model()
    model.load_weights('final_weights_72_0.22.h5')

    _, val_generator = create_data_generators(1)

    try:
        for _ in range(20):
            input("Press Enter to Predict...")
            data, label = next(val_generator)
            prediction = model.predict(data)
            print("\n")
            fig = plt.figure()
            plt.imshow(data[0])
            plt.show()
            plt.close(fig)
            print("\nThis is   --- {}".format(
                plant_names[np.argmax(label)]))
            print("Predicted --- {}\n".format(
                plant_names[np.argmax(prediction)]))
        print("End of demo")
    except KeyboardInterrupt:
        print("aborting")


def predict_test():
    '''predict_test'''
    _, plant_names, _ = next(walk(PATH_VAL))
    _, _, filenames = next(walk(PATH_TEST))

    img_arrays = get_images(filenames, PATH_TEST)

    model = load_final_model()
    model.load_weights('final_weights_72_0.22.h5')

    predictions = model.predict(img_arrays, batch_size=8, verbose=1)
    pred_plant_labels = np.argmax(predictions, axis=1)

    pred_plant_names = [plant_names[idx] for idx in pred_plant_labels]

    # prediction_pairs = list(zip(filenames, pred_plant_names))

    test_output = pd.DataFrame({'file': filenames,
                                'species': pred_plant_names})

    test_output.to_csv(
        'results/seedlings_test.csv',
        index=False,
    )


#%%

def load_base_model(input_shape):
    '''load_base_model'''
    input_tensor = Input(shape=input_shape)
    model = applications.vgg16.VGG16(
        include_top=False, weights='imagenet', input_tensor=input_tensor)
    # model = load_model('VGG16.h5')
    model.summary()
    return model


def load_top_model(input_shape):
    '''load_top_model'''
    print("load model")

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    model.summary()
    return model


def load_final_model():
    '''load_final_model'''
    base_model = load_base_model((IMG_WIDTH, IMG_HEIGHT, 3))

    base_model_output_shape = base_model.output_shape[1:]
    top_model = load_top_model(base_model_output_shape)

    top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

    model = Model(inputs=base_model.input,
                  outputs=top_model(base_model.output))
    model.summary()
    return model


#%%


if __name__ == '__main__':

    # BASE_MODEL = load_base_model((IMG_WIDTH, IMG_HEIGHT, 3))

    # calculate_features(BASE_MODEL)

    # BOTTLENECK_SHAPE = BASE_MODEL.output_shape[1:]
    # TOP_MODEL = load_top_model(BOTTLENECK_SHAPE)
    # train_top_model(TOP_MODEL)

    FINAL_MODEL = load_final_model()
    fine_tune_final_model(FINAL_MODEL)

    # demo()

    # predict_test()

    pass

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50

import ssl



class DogCatClassifier:
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 64

    def __init__(self, data="data", epochs=20):
        self.epochs = epochs

        # Load data
        self.X = sorted(os.listdir(data))
        self.y = np.empty(len(self.X), dtype=str)
        self.y[np.char.startswith(self.X, "cat")] = "cat"
        self.y[np.char.startswith(self.X, "dog")] = "dog"

        self.model = DogCatClassifier._load_model()

    def train(self):
        train_set, val_set, test_set = self._gen_data()

        # Fit the model
        history = self.model.fit(train_set,
                                 steps_per_epoch=train_set.n // DogCatClassifier.BATCH_SIZE,
                                 epochs=self.epochs,
                                 validation_data=val_set,
                                 validation_steps=val_set.n // DogCatClassifier.BATCH_SIZE)

        # Show the predictions on the testing set
        step_size_test = test_set.n // test_set.batch_size
        result = self.model.evaluate(test_set, steps=step_size_test)
        print("testing set evaluation:", dict(zip(self.model.metrics_names, result)))

        self.model.save('Resnet50')

        # Plot training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        #plt.savefig(os.path.join(SAVE_DIR, "results.png"))

    @classmethod
    def _load_model(cls):

        input = Input(shape=(cls.IMG_HEIGHT,cls.IMG_WIDTH,3))
        res_model = ResNet50(include_top=False, weights="imagenet", input_tensor=input)

        for layer in res_model.layers[:143]:
            layer.trainable = False

        #for i, layer in enumerate(res_model.layers):
        #    print(i, layer.name, "-", layer.trainable)

        model = Sequential()
        model.add(res_model)
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(learning_rate=1e-3),
                      metrics=['accuracy', 'AUC'])
        print(model.summary())

        return model

    def _gen_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        df_train = pd.DataFrame({"filename": X_train,
                                 "class": y_train})
        df_test = pd.DataFrame({"filename": X_test,
                                "class": y_test})

        train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, preprocessing_function=preprocess_input,
                                           horizontal_flip=True, shear_range=0.2, width_shift_range=0.2,
                                           height_shift_range=0.2, zoom_range=0.2, rotation_range=30,
                                           fill_mode="nearest")
        test_datagen = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess_input)

        train_data_generator = train_datagen.flow_from_dataframe(df_train,
                                                                 directory='data',
                                                                 x_col='filename',
                                                                 y_col='class',
                                                                 subset='training',
                                                                 shuffle=True,
                                                                 batch_size=self.BATCH_SIZE,
                                                                 class_mode='binary',
                                                                 target_size=(self.IMG_HEIGHT, self.IMG_WIDTH))
        valid_data_generator = train_datagen.flow_from_dataframe(df_train,
                                                                 directory='data',
                                                                 x_col='filename',
                                                                 y_col='class',
                                                                 subset='validation',
                                                                 shuffle=True,
                                                                 batch_size=self.BATCH_SIZE,
                                                                 class_mode='binary',
                                                                 target_size=(self.IMG_HEIGHT, self.IMG_WIDTH))
        test_data_generator = test_datagen.flow_from_dataframe(df_test,
                                                               directory='data',
                                                               x_col='filename',
                                                               y_col='class',
                                                               shuffle=False,
                                                               batch_size=self.BATCH_SIZE,
                                                               class_mode='binary',
                                                               target_size=(self.IMG_HEIGHT, self.IMG_WIDTH))

        return train_data_generator, valid_data_generator, test_data_generator


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    clf = DogCatClassifier()
    #clf.train()

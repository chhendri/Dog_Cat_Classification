import os
import pickle

import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class DogCatClassifier:
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    BATCH_SIZE = 16

    def __init__(self, data="data", epochs=10):
        self.epochs = epochs

        # Load data
        self.X = sorted(os.listdir(data))
        # self.X = sorted(np.random.choice(os.listdir(data), replace=False, size=500))
        self.y = np.empty(len(self.X), dtype=str)
        self.y[np.char.startswith(self.X, "cat")] = "cat"
        self.y[np.char.startswith(self.X, "dog")] = "dog"

        self.model = DogCatClassifier._load_model()

    def train(self):
        train_set, val_set, test_set = DogCatClassifier._gen_data(self.X, self.y)

        # Fit the model
        self.model.fit(train_set,
                       steps_per_epoch=train_set.n // DogCatClassifier.BATCH_SIZE,
                       epochs=self.epochs,
                       validation_data=val_set,
                       validation_steps=val_set.n // DogCatClassifier.BATCH_SIZE)

        # Show the predictions on the testing set
        step_size_test = test_set.n // test_set.batch_size
        self.model.evaluate(test_set, steps=step_size_test)

        test_set.reset()
        pred = self.model.predict(test_set,
                                  steps=step_size_test,
                                  verbose=1)

        print("1s:", np.multiply(pred >= 0.5, 1).sum())
        print("0s:", np.multiply(pred < 0.5, 1).sum())

    @classmethod
    def _load_model(cls):
        # Build a CNN model for image classification
        model = Sequential()
        model.add(Conv2D(128, (3, 3), input_shape=(cls.IMG_HEIGHT, cls.IMG_WIDTH, 3),
                         activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=1e-4),
                      metrics=['accuracy'])
        print(model.summary())

        return model

    @classmethod
    def _gen_data(cls, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        df_train = pd.DataFrame({"filename": X_train,
                                 "class": y_train})
        df_test = pd.DataFrame({"filename": X_test,
                                "class": y_test})

        train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, preprocessing_function=preprocess_input,
                                           horizontal_flip=True, shear_range=0.2, fill_mode="nearest")
        test_datagen = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess_input)

        train_data_generator = train_datagen.flow_from_dataframe(df_train,
                                                                 directory='data',
                                                                 x_col='filename',
                                                                 y_col='class',
                                                                 subset='training',
                                                                 shuffle=True,
                                                                 batch_size=cls.BATCH_SIZE,
                                                                 class_mode='binary',
                                                                 target_size=(cls.IMG_HEIGHT, cls.IMG_WIDTH))
        valid_data_generator = train_datagen.flow_from_dataframe(df_train,
                                                                 directory='data',
                                                                 x_col='filename',
                                                                 y_col='class',
                                                                 subset='validation',
                                                                 shuffle=True,
                                                                 batch_size=cls.BATCH_SIZE,
                                                                 class_mode='binary',
                                                                 target_size=(cls.IMG_HEIGHT, cls.IMG_WIDTH))
        test_data_generator = test_datagen.flow_from_dataframe(df_test,
                                                               directory='data',
                                                               x_col='filename',
                                                               shuffle=False,
                                                               batch_size=62,
                                                               class_mode=None,
                                                               target_size=(cls.IMG_HEIGHT, cls.IMG_WIDTH))

        return train_data_generator, valid_data_generator, test_data_generator


if __name__ == "__main__":
    clf = DogCatClassifier()
    clf.train()

    with open("test", "wb") as file:
        pickle.dump(clf, file)

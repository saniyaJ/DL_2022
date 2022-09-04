import pandas as pd
import numpy as np
import os
import PIL
from pathlib import Path
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import IPython
from IPython.display import display
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix


ROOT_PATH = Path().resolve()


class SVMClassifier:
    train_data: pd.DataFrame
    train_label: pd.DataFrame
    test_data: pd.DataFrame
    test_labels: pd.DataFrame

    def __init__(
        self,
        train_data,
        train_label,
        test_data,
        test_label
    ):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

    def standard_scaler(self):
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.train_data)
        self.x_test = sc.transform(self.test_data)

    def train_scaled_data(self):
        self.svm_classifier = SVC(kernel = 'rbf', random_state = 0)
        self.svm_classifier.fit(self.x_train, self.train_label)

    def get_confusion_matrix(self):
        y_pred = self.svm_classifier.predict(self.x_test)
        self.conf_matrix = confusion_matrix(self.test_label, y_pred)

        return self


def create_data_generators(
    data_directory,
    size_of_batch,
    max_pixel,
    classes = ['others', 'aeroplane']
):
    train_datagen = ImageDataGenerator(
        rescale = 1./255.,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True
    )
    valid_datagen = ImageDataGenerator(rescale = 1.0/255.)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_directory, 'data/training'),
        batch_size=size_of_batch,
        class_mode = 'binary',
        target_size = (max_pixel, max_pixel),
        classes=classes
    )
    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(data_directory, 'data/validation'),
        batch_size=size_of_batch,
        class_mode = 'binary',
        target_size = (max_pixel, max_pixel),
        classes=classes
    )

    return train_generator, valid_generator


def generate_base_model(
    input_array_shape
):
    base_model = InceptionV3(
        input_shape = input_array_shape,
        include_top = True,
        weights = 'imagenet',
        classes=1000,
    )

    for layer in base_model.layers:
        layer.trainable = False
    
    return base_model


def fit_model(
    model_base,
    train_data_generator,
    validation_data_generator
):
    x = Flatten()(model_base.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = Dense(1000, activation='softmax')(x)

    model = tf.keras.models.Model(model_base.input, x)
    x = Flatten()(model.output)
    model = tf.keras.models.Model(model.input, x)
    # Model compilation
    model.compile(
        optimizer = RMSprop(learning_rate=0.0001),
        loss = 'binary_crossentropy',
        metrics = ['acc']
    )
    # Fit the model
    model.fit(
        train_data_generator, 
        validation_data=validation_data_generator, 
        steps_per_epoch=1,
        epochs=10
    )
    
    return model


def get_features(
    fitted_model,
    data_generator: str,
    feature_dimension: float = 1000.,
    sample_size: int = 83,
    batch_size: int = 200
):
    features = np.zeros(shape=(sample_size, feature_dimension))  # Must be equal to the output of the fitted_model
    labels = np.zeros(shape=(sample_size))
    
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in data_generator:
        features_batch = fitted_model.predict(inputs_batch)
        # print(features_batch.shape)
        if features_batch.shape[0] != batch_size:
            features[i * batch_size:] = features_batch
            labels[i * batch_size:] = labels_batch
        else:
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_size:
            break
    
    return features, labels


if __name__ == '__main__':
    maxPixel = 299
    # Data generators
    train_data, validation_data = create_data_generators(
        data_directory=ROOT_PATH,
        size_of_batch=20,
        max_pixel=maxPixel,
        classes=['others', 'aeroplane']
    )
    # Create base model
    base_mod = generate_base_model(
        input_array_shape=(maxPixel, maxPixel, 3)
    )
    # Add layers to model
    image_classifier_model = fit_model(
        model_base=base_mod,
        train_data_generator=train_data,
        validation_data_generator=validation_data
    )
    # Get train features
    train_features, train_labels = get_features(
        image_classifier_model,
        data_generator=train_data,
        feature_dimension=1000,
        sample_size=83,
        batch_size=20
    )
    # Get validation features
    validation_features, validation_labels = get_features(
        image_classifier_model,
        data_generator=validation_data,
        feature_dimension=1000,
        sample_size=92,
        batch_size=20
    )
    # Train and predict using a SVM Classifier
    svm_object = SVMClassifier(
        train_data=train_features,
        train_label=train_labels,
        test_data=validation_features,
        test_label=validation_labels
    )
    scaled_object = svm_object.standard_scaler()
    trained_model = svm_object.train_scaled_data()
    confusion_matrix_from_svm = svm_object.get_confusion_matrix()
    
    print(confusion_matrix)

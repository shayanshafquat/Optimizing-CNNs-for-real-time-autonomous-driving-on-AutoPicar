import tensorflow as tf
import os
import numpy as np
import random
import cv2
from datetime import datetime
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import SpatialPyramidPooling, GaussianBlurLayer, get_class_weights_for_angle_model, RandomGaussianBlur
from vit_keras import vit
from tensorflow.keras.utils import get_custom_objects

# Configure GPU usage and reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(123)

    
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# def get_sample_weights_for_angle_model(directory_path):
#     class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0']
#     class_counts = {}

#     for class_name in class_names:
#         class_dir = os.path.join(directory_path, class_name)
#         class_counts[class_name] = len([item for item in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, item))])

#     total_samples = sum(class_counts.values())
#     sample_weights = {class_name: total_samples / count for class_name, count in class_counts.items()}
#     return sample_weights


def preprocess_image(image, label):
    angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
    image = tf.image.resize(image, [192, 192])
    label_indices = tf.argmax(label, axis=-1)
    label_unnormalized = tf.gather(angles, label_indices)
    label_normalized = (tf.cast(label_unnormalized, tf.float32) - 50) / 80
    return tf.cast(image, tf.float32), label_normalized

def build_model(model_name):
    model = Sequential(name=model_name)
    model.add(Input(shape=(192, 192, 3)))
    model.add(tf.keras.layers.RandomBrightness(0.2, seed=123))
    model.add(tf.keras.layers.RandomContrast(0.2, seed=123))

    if model_name == 'resnetv2':
        base_model = tf.keras.applications.ResNet50(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50V2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'vgg':
        base_model = tf.keras.applications.VGG16(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    base_model.trainable = False
    model.add(Lambda(preprocess_input))
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def fine_tune_model(model, checkpoint_path, learning_rate, layers_to_tune):
    print("Finetuning ...")
    # learning_rate /= 10
    print(f"Learning Rate:{learning_rate}")
    model.layers[3].trainable = True
    for layer in model.layers[3].layers[:-layers_to_tune]:
        layer.trainable = False
    for layer in model.layers[3].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.summary()
    return learning_rate

if __name__ == "__main__":
    directory = '../../data/angle_class_data'
    finetuning = False
    learning_rate = 0.001

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',  # Changed to int for regression
        class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'],
        batch_size=128,
        image_size=(192, 192),
        shuffle=True,
        seed=123,
        validation_split=0.15,
        subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'],
        batch_size=128,
        image_size=(192, 192),
        shuffle=True,
        seed=123,
        validation_split=0.15,
        subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    ts = datetime.now().strftime('%Y%m%d')

    model_name = 'mobilenetv2'
    model = build_model(model_name)
    model.summary()

    checkpoint_path = f"../../saved_models/angle/{model.name}_{ts}_reg"
    model_checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_mean_squared_error',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch')

    if not finetuning:
        print("Base Learning")
        history = model.fit(train_ds,
                            epochs=10,
                            validation_data=val_ds,
                            callbacks=[model_checkpoint_callback],
                            verbose=1)

    # Assuming we are fine-tuning in three stages
    layers_to_tune = [20, 50]
    lr_tune = [0.00005, 0.00001]

    for i in range(len(layers_to_tune)):
        model = tf.keras.models.load_model(checkpoint_path)
        learning_rate = fine_tune_model(model, checkpoint_path, lr_tune[i], layers_to_tune[i])
        fine_tune_epochs = 10
        total_epochs = fine_tune_epochs
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            callbacks=[model_checkpoint_callback],
            verbose=1)
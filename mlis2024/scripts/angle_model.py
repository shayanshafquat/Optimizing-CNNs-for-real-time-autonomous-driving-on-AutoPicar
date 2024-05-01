import tensorflow as tf
import os
import numpy as np
import random
import cv2
from datetime import datetime
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import SpatialPyramidPooling, GaussianBlurLayer, get_class_weights_for_angle_model, RandomGaussianBlur
from vit_keras import vit
from tensorflow.keras.utils import get_custom_objects



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
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

tf.random.set_seed(123)

def preprocess_image(image, label):
    # Define the angles tensor
    angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)

    # Calculate mirrored angles
    mirrored_angles = 180 - tf.cast(angles, tf.float32)

    image = tf.image.resize(image, [192, 192]) 
    # original_image = tf.identity(image)  # Preserve the original image for comparison

    # if random.random() < 0.2:
    #     image = tf.image.flip_left_right(image)

    # def adjust_label(label):
    #     # Assuming label is a batch of one-hot encoded labels with shape [batch_size, num_classes]
    #     # Find the indices of the max value in each row (highest probability class)
    #     label_indices = tf.argmax(label, axis=-1)
        
    #     # For each label index, find the corresponding angle and then find the mirrored angle's index
    #     angles_batch = tf.gather(angles, label_indices)
        
    #     # This operation needs to be adapted to handle a batch of angles
    #     # Use tf.map_fn to apply the finding operation across the batch
    #     def find_mirrored_index(angle):
    #         # Find the index of this angle in the mirrored_angles list
    #         mirrored_index = tf.where(tf.equal(mirrored_angles, angle))
    #         return tf.reshape(mirrored_index, [-1])[0]

    #     mirrored_indices = tf.map_fn(find_mirrored_index, angles_batch, fn_output_signature=tf.int64)
        
    #     # Create new one-hot encoded labels based on the mirrored indices
    #     new_labels = tf.one_hot(mirrored_indices, depth=tf.size(angles), dtype=label.dtype)
        
    #     return new_labels

    # # Check if flipping occurred
    # is_flipped = tf.reduce_any(tf.not_equal(original_image, image))

    # # Use tf.cond to conditionally adjust the label, ensuring the output matches the input type
    # label = tf.cond(is_flipped, lambda: adjust_label(label), lambda: tf.identity(label))
    
    return tf.cast(image, tf.float32), label

# Model definition with preprocessing included
def build_model(num_classes, model_name):
    # with tf.device('/cpu:0'):
    # Initialize the model as a Sequential model
    model = Sequential(name = model_name)

    # Data augmentation layers
    model.add(Input(shape=(192, 192, 3)))  # Define input shape
    model.add(tf.keras.layers.RandomBrightness(0.2, seed=123))
    model.add(tf.keras.layers.RandomContrast(0.2, seed=123))
    model.add(tf.keras.layers.GaussianNoise(4))

    # Select the base model based on the provided model_name
    if model_name == 'resnetv2':
        base_model = tf.keras.applications.ResNet50(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    if model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50V2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'vgg':
        base_model = tf.keras.applications.VGG16(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Set the base model to non-trainable (freeze layers)
    base_model.trainable = False
    base_model_num_layers = len(base_model.layers)

    # Include Gaussian blur and preprocessing
    # model.add(GaussianBlurLayer(kernel_size=(3, 3), sigma=0))  # Custom Gaussian blur layer
    model.add(RandomGaussianBlur(5,0.1))
    model.add(tf.keras.layers.Lambda(preprocess_input))  # Preprocessing using Lambda layer

    # Add the base model
    model.add(base_model)

    # Additional layer configurations - Uncomment as needed
    model.add(GlobalAveragePooling2D())  # Global average pooling layer
    # model.add(SpatialPyramidPooling(pool_list=[1, 2, 4]))
    # model.add(GlobalMaxPooling2D())
    # model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # Output layer

    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',  # Make sure you have this loss function implemented or available
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
    
    return model, base_model_num_layers

# Fine-tuning function
def fine_tune_model(model, checkpoint_path, learning_rate, layers_to_tune):
    print("Finetuning ...")
    learning_rate /= 10 
    print(f"Learning Rate:{learning_rate}")

    # Unfreeze all layers in base model
    model.layers[5].trainable = True
    for layer in model.layers[5].layers[:-layers_to_tune]:
        layer.trainable = False
    for layer in model.layers[5].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
    model.summary()
    return learning_rate

class_names = ['65.0', '50.0', '75.0', '115.0', '130.0', '85.0', '105.0', '120.0', '95.0', '80.0', '110.0', '125.0', '90.0', '100.0', '60.0', '70.0', '55.0']


if __name__ == "__main__":
    directory = '../../data/angle_class_data'
    finetuning = False
    learning_rate = 0.001
    num_fine_tuning = 2

    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'],
                color_mode='rgb',
                batch_size=128,
                image_size = (192,256),
                shuffle=True,
                seed=123,
                validation_split=0.15,
                subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'],
                color_mode='rgb',
                batch_size=128,
                image_size = (192,256),
                shuffle=True,
                seed=123,
                validation_split=0.15,
                subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    ts = datetime.now().strftime('%Y%m%d')
    
    class_weights = get_class_weights_for_angle_model(directory)
    
    print("Class weights for multi classification:", class_weights)

    model_name = 'mobilenetv2'
    model, base_layers = build_model(17, model_name)
    get_custom_objects().update({'RandomGaussianBlur': RandomGaussianBlur})
    model.summary()

    checkpoint_path = f"../../saved_models/angle/{model.name}_{ts}"
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
                epochs=30,
                validation_data=val_ds,
                callbacks=[model_checkpoint_callback],
                verbose=1, class_weight=class_weights)

    for i in range(num_fine_tuning):
        # Load the best model for fine-tuning
        model = tf.keras.models.load_model(checkpoint_path)
        layers_to_tune = int((base_layers/3)*(i+1))
        # Fine-tuning
        learning_rate = fine_tune_model(model, checkpoint_path, learning_rate, layers_to_tune)
        # Fine-tuning training with a smaller learning rate
        fine_tune_epochs = 15
        total_epochs = fine_tune_epochs  # Initial epochs + fine-tune epochs

        history_finetune = model.fit(
                                train_ds,
                                validation_data=val_ds,
                                epochs=total_epochs,
                                callbacks=[model_checkpoint_callback],
                                verbose=1, class_weight=class_weights)

    # # Evaluate the fine-tuned model
    # evaluate_model(model, valid_generator)

    # # Initialize early stopping
    # early_stopping = EarlyStopping(
    #     monitor='val_mean_squared_error',  # Monitor the model's validation loss
    #     patience=8,         # How many epochs to wait after last time val loss improved
    #     verbose=1,           # Log when training is stopped
    #     mode='min',
    #     restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    # )


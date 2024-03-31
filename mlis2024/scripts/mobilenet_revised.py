import os
import pandas as pd
import numpy as np

import cv2
from datetime import datetime
import tensorflow as tf
# from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

def preprocess_image(image):
    # Note: The input image here is expected to be a numpy array, so no conversion is needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Resize image (if not done by ImageDataGenerator)
    # image = cv2.resize(image, (224, 224))
    # image = (image - 127.5) / 127.5
    return image

# def preprocess_image(image):
#     # Resize images
#     image = tf.image.resize(image, (224, 224))
#     # Normalize images
#     # image = image / 255.0
#     return image

# Prepare data with ImageDataGenerator
def prepare_data(csv_file, img_dir, label_type='angle', batch_size=32):
    df = pd.read_csv(csv_file)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))
    
    if 'angle' in df.columns:
        df['angle'] = df['angle'].apply(lambda x: int(x * 80 + 50))

    # Map angle values to discrete classes (e.g., 50->0, 55->1, ..., 130->16)
    df['angle'] = df['angle'].apply(lambda x: (x - 50) // 5)

    # Ensure label_angle is treated as a string for categorical mode in flow_from_dataframe
    df['angle'] = df['angle'].astype(str)

    # Split data
    train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # # Augmentation configuration for training
    # train_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input,
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    # )
    # train_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input,  # Apply MobileNetV2 preprocessing
    # brightness_range=[0.5, 1.0],  # Adjust brightness to simulate different lighting conditions
    # width_shift_range=0.1,  # Slight horizontal shifts to simulate car position variation
    # height_shift_range=0.1,  # Slight vertical shifts to simulate road elevation changes
    # zoom_range=0.1,  # Slight zoom to simulate distance changes
    # fill_mode='nearest'  # Fill in new pixels after a shift or shear operation
    # )
    train_datagen = ImageDataGenerator(
        # horizontal_flip=True,
        preprocessing_function=preprocess_image, # Apply MobileNetV2 preprocessing
    # shear_range = 0.1,
    # zoom_range=0.1
    )
    
    # No augmentation for validation, only rescaling
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    
    # Train and validation generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col='image_path',
        y_col=label_type,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary' if label_type == 'speed' else 'categorical'
    )
    
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=None,
        x_col='image_path',
        y_col=label_type,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary' if label_type == 'speed' else 'categorical'
    )
    
    return train_generator, valid_generator

# Model definition with preprocessing included
def build_model(num_classes, noise_stddev=0.1):

    # with tf.device('/cpu:0'):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.1),
                                            tf.keras.layers.CenterCrop(224,224),
                                            tf.keras.layers.RandomFlip('horizontal'),
                                            tf.keras.layers.RandomZoom(0.1, fill_mode='reflect'),
                                            tf.keras.layers.RandomRotation(0.05, fill_mode='nearest'),

    ])

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    inputs = Input(shape=(240,320,3))
    # aug_inputs = data_augmentation(inputs)
    x = preprocess_input(aug_inputs)  # Preprocessing for MobileNetV2
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)


    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    model.summary()
    return model

# Fine-tuning function
def fine_tune_model(model):
    # Unfreeze all layers in base model
    model.layers[2].trainable = True  # '2' is the index of base_model within our model
    

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])


# Model evaluation function
def evaluate_model(model, valid_generator):
    results = model.evaluate(valid_generator, verbose=1)
    print(f"Loss: {results[0]}, Accuracy: {results[1]}, Precision: {results[2]}, Recall: {results[3]}")


# Main script
if __name__ == "__main__":
    csv_file = 'training_norm.csv'
    img_dir = 'training_data/training_data'  # Adjust as needed
    label_type = 'angle'  # Or 'speed'
    num_classes = 17 if label_type == 'angle' else 1  # Adjust based on your dataset
    
    tf.random.set_seed(42) 

    train_generator, valid_generator = prepare_data(csv_file, img_dir, label_type, batch_size=64)
    model = build_model(num_classes)
    
    
    # Initial training
    # checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
        # Path to save the best model
    timestamp = datetime.now().strftime('%Y%m%d')
    checkpoint_path = f'models/angle/mobilenet_v1_{timestamp}'

    model_checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',     # Monitor validation loss
    verbose=1,              # Log a message each time the callback saves the model
    save_best_only=True,    # Only save the model if 'val_loss' has improved
    save_weights_only=False, # Only save the weights of the model
    mode='min',             # 'min' means the monitored quantity should decrease
    save_freq='epoch')       # Check every epoch


    history = model.fit(train_generator, epochs=10, validation_data=valid_generator, callbacks=[model_checkpoint_callback], verbose=1)

    # Load the best model for fine-tuning
    model = tf.keras.models.load_model(checkpoint_path)

    # Fine-tuning
    fine_tune_model(model)
    # Fine-tuning training with a smaller learning rate
    fine_tune_epochs = 10
    total_epochs = 10 + fine_tune_epochs  # Initial epochs + fine-tune epochs

    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=total_epochs,
        initial_epoch=10,  # Start fine-tuning from the epoch we left off
        callbacks=[model_checkpoint_callback],
        verbose=1)

    # Evaluate the fine-tuned model
    evaluate_model(model, valid_generator)
    


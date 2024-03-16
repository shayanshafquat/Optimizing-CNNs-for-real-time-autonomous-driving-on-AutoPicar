import os
import datetime
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf

import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

print( f'tf.__version__: {tf.__version__}' )
print( f'keras.__version__: {keras.__version__}' )

import cv2
from PIL import Image

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,200)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def image_data_generator(image_paths, steering_angles, batch_size):
    while True:
        batch_images = []
        batch_steering_angles = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]
              
            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)
            
        yield( np.asarray(batch_images), np.asarray(batch_steering_angles))

def nvidia_model():
    model = Sequential(name='Nvidia_Model')
    model.add(tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False, weights="imagenet", input_shape=(200, 200, 3)))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(1, activation = 'sigmoid'))
    model.layers[0].trainable = False

    model.compile(optimizer=tf.optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics='accuracy')

    return model

data_dir = 'training_data/training_data'
file_list = os.listdir(data_dir)

df = pd.read_csv('training_data/training_norm.csv')

image_id = []
image_name = []
image_path = []
image_array = []
file_size = []
for filename in file_list:
    im = cv2.imread(data_dir + '/' + filename)

    image_id.append(int(filename.split('.')[0]))
    # image_name.append(filename)
    image_array.append(im)
    image_path.append(data_dir + '/' + filename)
    file_size.append(os.path.getsize(data_dir + '/' + filename))

data = {
    'image_id': image_id,
    'image': image_array,
    'image_path': image_path,
    'file_size': file_size
}
df_image = pd.DataFrame(data)

merged_df = pd.merge(df, df_image, how='left', on='image_id')

X_train, X_valid, y_train, y_valid = train_test_split(merged_df['image_path'].to_list(), merged_df['speed'].to_list(), test_size=0.4)




model_output_dir = 'models/angle'

# start Tensorboard before model fit, so we can see the epoch tick in Tensorboard
# Jupyter Notebook embedded Tensorboard is a new feature in TF 2.0!!  

# clean up log folder for tensorboard
log_dir_root = f'{model_output_dir}/logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#!rm -rf $log_dir_root

tensorboard_callback = TensorBoard(log_dir_root, histogram_freq=1)

# Specify the file path where you want to save the model
filepath = 'models/angle/{epoch:02d}-{val_loss:.2f}.hdf5'

# Create the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath,
    monitor='val_loss',     # Monitor validation loss
    verbose=1,              # Log a message each time the callback saves the model
    save_best_only=True,    # Only save the model if 'val_loss' has improved
    save_weights_only=True, # Only save the weights of the model
    mode='min',             # 'min' means the monitored quantity should decrease
    save_freq='epoch'       # Check every epoch
)

history = model.fit(
    image_data_generator(X_train, y_train, batch_size=200),
    steps_per_epoch=500,
    epochs=10,
    validation_data = image_data_generator(X_valid, y_valid, batch_size=200),
    validation_steps=200,
    verbose=1,
    shuffle=1,
    callbacks=[checkpoint_callback, tensorboard_callback]
)
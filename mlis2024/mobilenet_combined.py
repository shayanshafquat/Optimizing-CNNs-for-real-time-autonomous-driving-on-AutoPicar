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
from keras.models import Sequential, Model  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop

print( f'tf.__version__: {tf.__version__}' )
print( f'keras.__version__: {keras.__version__}' )

import cv2
from PIL import Image

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def img_preprocess(image):
    # height, _, _ = image.shape
    # image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (192,192)) # input image size (200,66) Nvidia model
    # image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    # image = (image - 127.5) / 127.5
    return image


def image_data_generator(image_paths, labels_dict, batch_size):
    while True:
        batch_images = []
        batch_angles = []
        batch_speeds = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[random_index])

            # Since labels_dict is a dictionary, access the labels with keys
            angle_label = labels_dict['angle_output'][random_index]
            speed_label = labels_dict['speed_output'][random_index]
            angle_label *= 16
              
            image = img_preprocess(image)
            batch_images.append(image)
            
            # Assuming angle_label needs conversion to class index and one-hot encoding
            # Adjust this part according to how your angle labels are structured
            # Example assumes angle_label is already an integer class label; adjust if it's not
            angle_one_hot = to_categorical(angle_label, num_classes=17)  # Adjust num_classes based on your total classes
            batch_angles.append(angle_one_hot)
            
            # Add speed label as is (assuming it's already 0 or 1 for binary classification)
            batch_speeds.append(speed_label)
            
        batch_angles = np.array(batch_angles)
        batch_angles = batch_angles.reshape((batch_size, 17))
        yield (np.asarray(batch_images), {'angle_output': batch_angles, 'speed_output': np.array(batch_speeds)})

def nvidia_model():
    model = Sequential(name='Nvidia_Model')
    model.add(tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False, weights="imagenet", input_shape=(200, 200, 3)))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(1, activation = 'sigmoid'))
    model.layers[0].trainable = False

    model.compile(optimizer=tf.optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics='accuracy')

    return model

def mobile_net_classification_model():
    inputs = Input(shape=(192, 192, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)

    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    # Common part of the model
    common = Dense(1024, activation='relu')(x)
    common = Dropout(0.5)(common)

    # Branch for the angle prediction (multi-class classification)
    angle_branch = Dense(512, activation='relu')(common)
    angle_branch = Dropout(0.5)(angle_branch)
    angle_output = Dense(17, activation='softmax', name='angle_output')(angle_branch) # 10 classes for angle

    # Branch for the speed prediction (binary classification)
    # speed_branch = Dense(512, activation='relu')(common)
    # speed_branch = Dropout(0.5)(speed_branch)
    speed_output = Dense(1, activation='sigmoid', name='speed_output')(common) # Binary classification for speed


    model = Model(inputs=inputs, outputs=[angle_output, speed_output])
    # Create an RMSprop optimizer with a custom learning rate
    custom_lr = 0.0001  # Example custom learning rate
    optimizer = RMSprop(learning_rate=custom_lr)

    model.compile(optimizer='adam',
                  loss={'angle_output': 'categorical_crossentropy', 'speed_output': 'binary_crossentropy'},
                  metrics={'angle_output': 'accuracy', 'speed_output': 'accuracy'})

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

cleaned_df = merged_df[merged_df['speed'] <= 1]

angle_labels = cleaned_df['angle'].to_list()
speed_labels = cleaned_df['speed'].to_list()
image_paths = cleaned_df['image_path'].to_list()


X_train, X_valid, angle_train, angle_valid, speed_train, speed_valid = train_test_split(image_paths, angle_labels, speed_labels, test_size=0.3)

model = mobile_net_classification_model()

model_output_dir = 'models/combined'

# start Tensorboard before model fit, so we can see the epoch tick in Tensorboard
# Jupyter Notebook embedded Tensorboard is a new feature in TF 2.0!!  

# clean up log folder for tensorboard
log_dir_root = f'{model_output_dir}/logs'
#!rm -rf $log_dir_root

tensorboard_callback = TensorBoard(log_dir_root, histogram_freq=1)

# Specify the file path where you want to save the model
filepath = 'models/combined/{epoch:02d}-{val_loss:.2f}.hdf5'

# Create the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath,
    monitor='val_loss',     # Monitor validation loss
    verbose=1,              # Log a message each time the callback saves the model
    save_best_only=True,    # Only save the model if 'val_loss' has improved
    save_weights_only=False, # Only save the weights of the model
    mode='min',             # 'min' means the monitored quantity should decrease
    save_freq='epoch'       # Check every epoch
)

history = model.fit(
    image_data_generator(X_train, {'angle_output': angle_train, 'speed_output': speed_train}, batch_size=128),
    steps_per_epoch=500,
    epochs=10,
    validation_data = image_data_generator(X_valid, {'angle_output': angle_valid, 'speed_output': speed_valid}, batch_size=128),
    verbose=1,
    shuffle=1,
    callbacks=[model_checkpoint_callback, tensorboard_callback]
)
import tensorflow as tf
import os
import numpy as np
import cv2
from datetime import datetime
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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

import tensorflow.keras.backend as K

class SpatialPyramidPooling(Layer):
	"""Spatial pyramid pooling layer for 2D inputs.
	See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
	K. He, X. Zhang, S. Ren, J. Sun
	# Arguments
		pool_list: list of int
			List of pooling regions to use. The length of the list is the number of pooling regions,
			each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
			regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
	# Input shape
		4D tensor with shape:
		`(samples, channels, rows, cols)` if dim_ordering='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if dim_ordering='channels_last'.
	# Output shape
		2D tensor with shape:
		`(samples, channels * sum([i * i for i in pool_list])`
	"""

	def __init__(self, pool_list, **kwargs):

		self.dim_ordering = K.image_data_format()
		assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

		self.pool_list = pool_list

		self.num_outputs_per_channel = sum([i * i for i in pool_list])

		super(SpatialPyramidPooling, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.dim_ordering == 'channels_first':
			self.nb_channels = input_shape[1]
		elif self.dim_ordering == 'channels_last':
			self.nb_channels = input_shape[3]

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

	def get_config(self):
		config = {'pool_list': self.pool_list}
		base_config = super(SpatialPyramidPooling, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def call(self, x, mask=None):

		input_shape = K.shape(x)

		if self.dim_ordering == 'channels_first':
			num_rows = input_shape[2]
			num_cols = input_shape[3]
		elif self.dim_ordering == 'channels_last':
			num_rows = input_shape[1]
			num_cols = input_shape[2]

		row_length = [K.cast(num_rows, dtype='float32') / i for i in self.pool_list]
		col_length = [K.cast(num_cols, dtype='float32') / i for i in self.pool_list]

		outputs = []

		if self.dim_ordering == 'channels_first':
			for pool_num, num_pool_regions in enumerate(self.pool_list):
				for jy in range(num_pool_regions):
					for ix in range(num_pool_regions):
						x1 = ix * col_length[pool_num]
						x2 = ix * col_length[pool_num] + col_length[pool_num]
						y1 = jy * row_length[pool_num]
						y2 = jy * row_length[pool_num] + row_length[pool_num]

						x1 = K.cast(K.round(x1), 'int32')
						x2 = K.cast(K.round(x2), 'int32')
						y1 = K.cast(K.round(y1), 'int32')
						y2 = K.cast(K.round(y2), 'int32')
						new_shape = [input_shape[0], input_shape[1],
									 y2 - y1, x2 - x1]
						x_crop = x[:, :, y1:y2, x1:x2]
						xm = K.reshape(x_crop, new_shape)
						pooled_val = K.max(xm, axis=(2, 3))
						outputs.append(pooled_val)

		elif self.dim_ordering == 'channels_last':
			for pool_num, num_pool_regions in enumerate(self.pool_list):
				for jy in range(num_pool_regions):
					for ix in range(num_pool_regions):
						x1 = ix * col_length[pool_num]
						x2 = ix * col_length[pool_num] + col_length[pool_num]
						y1 = jy * row_length[pool_num]
						y2 = jy * row_length[pool_num] + row_length[pool_num]

						x1 = K.cast(K.round(x1), 'int32')
						x2 = K.cast(K.round(x2), 'int32')
						y1 = K.cast(K.round(y1), 'int32')
						y2 = K.cast(K.round(y2), 'int32')

						new_shape = [input_shape[0], y2 - y1,
									 x2 - x1, input_shape[3]]

						x_crop = x[:, y1:y2, x1:x2, :]
						xm = K.reshape(x_crop, new_shape)
						pooled_val = K.max(xm, axis=(1, 2))
						outputs.append(pooled_val)

		if self.dim_ordering == 'channels_first':
			outputs = K.concatenate(outputs)
		elif self.dim_ordering == 'channels_last':
			# outputs = K.concatenate(outputs, axis = 1)
			outputs = K.concatenate(outputs)
			# outputs = K.reshape(outputs, (len(self.pool_list), self.num_outputs_per_channel, input_shape[0], input_shape[1]))
			# outputs = K.permute_dimensions(outputs, (3, 1, 0, 2))
			outputs = K.reshape(outputs, (input_shape[0], self.num_outputs_per_channel * self.nb_channels))

		return outputs
	
index_to_angle = tf.constant([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0], dtype=tf.float32)

def preprocess_image(image, label):
        # image = tf.image.crop_to_bounding_box(image, 0, 25, 224, 270)
        image = tf.image.resize(image, [224, 224])          # Resize the cropped image to 160x160
        # Convert the image to grayscale,  Replicate the grayscale image across three channels
        # image = tf.image.rgb_to_grayscale(image)
        # image = tf.image.grayscale_to_rgb(image)
        # Convert the index to the corresponding angle
        label_angle = tf.gather(index_to_angle, label)
        label_norm = (label_angle - 50) / 80
        return tf.cast(image, tf.float64), label_norm


# Model definition with preprocessing included
def build_model(num_classes):

    # with tf.device('/cpu:0'):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.1, seed=123),
                                             tf.keras.layers.RandomContrast(0.1, seed=123),
                                            # tf.keras.layers.RandomFlip('horizontal', seed=123),
                                            # tf.keras.layers.CenterCrop(192,192),
                                            # tf.keras.layers.RandomZoom(0.1, fill_mode='nearest', seed=123),
                                            tf.keras.layers.GaussianNoise(0.1)
                                            # tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_hsv(x)),
                                            # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),

    ])

    base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    inputs = Input(shape=(224,224,3))
    aug_inputs = data_augmentation(inputs)
    x = preprocess_input(aug_inputs)  # Preprocessing for Resnet
    x = base_model(x, training=False)
    # x = GlobalAveragePooling2D()(x)
    x = SpatialPyramidPooling(pool_list=[1, 2, 4])(x)
    # x = GlobalMaxPooling2D()(x)
    # x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

# Fine-tuning function
def fine_tune_model(model):
    # Unfreeze all layers in base model
    print("Finetuning ...")

    model.layers[4].trainable = True
    for layer in model.layers[4].layers[:-10]:
        layer.trainable = False
    for layer in model.layers[4].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00005)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()

finetuning = False

if __name__ == "__main__":
    directory = 'angle_class_data'
    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='int',
                color_mode='rgb',
                batch_size=128,
                image_size = (240,320),
                shuffle=True,
                seed=123,
                validation_split=0.3,
                subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='int',
                color_mode='rgb',
                batch_size=32,
                image_size = (240,320),
                shuffle=True,
                seed=123,
                validation_split=0.3,
                subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    ts = datetime.now().strftime('%Y%m%d')
    checkpoint_path = f"models/angle/resnet_best_{ts}"

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
        model = build_model(17)
        model.summary()

        history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[model_checkpoint_callback], verbose=1)

    # Load the best model for fine-tuning
    model = tf.keras.models.load_model(checkpoint_path)

    # Fine-tuning
    fine_tune_model(model)
    # Fine-tuning training with a smaller learning rate
    fine_tune_epochs = 10

    history_finetune = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=fine_tune_epochs,
                            callbacks=[model_checkpoint_callback],
                            verbose=1)

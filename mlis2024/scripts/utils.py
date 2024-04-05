import os
import cv2
import pandas as pd
import shutil
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer, MaxPooling2D
import tensorflow.keras.backend as K

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def save_image(image, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):  # Check if the file has already been copied/augmented
        cv2.imwrite(target_path, image)

def count_images_in_directories(directory):
    counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len(os.listdir(class_path))
    return counts

def identify_classes_for_augmentation(counts):
    average_count = np.mean(list(counts.values()))
    return [class_name for class_name, count in counts.items() if count < average_count]

def horizontal_flip(image_path, label):
    image = cv2.imread(image_path)
    augmented = False  # Flag to track if the image has been augmented
    new_label = label
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    augmented = True
    # Adjust the label for the augmented image
    new_label = 180 - label
    return image, augmented, new_label

def tf_adjust_brightness(image):
    return tf.image.random_brightness(image, max_delta=0.2)

def tf_adjust_contrast(image):
    return tf.image.random_contrast(image, lower=0.5, upper=1.5)

def save_original_images(angle_class_dir, speed_class_dir, image_dir, data):
    # Clear directories first
    clear_directory(angle_class_dir)
    clear_directory(speed_class_dir)

        # Create directories and save original images
    for _, row in data.iterrows():
        source_path = os.path.join(image_dir, f"{int(row['image_id'])}.png")
        for class_dir, class_label in [(angle_class_dir, "angle"), (speed_class_dir, "speed")]:
            target_dir = os.path.join(class_dir, str(row[class_label]))
            os.makedirs(target_dir, exist_ok=True)
            save_image(cv2.imread(source_path), os.path.join(target_dir, f"{int(row['image_id'])}.png"))

    print("Saved original files")

def augment_images_for_class(row, image_dir, class_dir, class_label):
    source_path = os.path.join(image_dir, f"{int(row['image_id'])}.png")
    original_target_path = os.path.join(class_dir, str(row[class_label]), f"{int(row['image_id'])}.png")
    flipped_image, _, new_label = horizontal_flip(source_path, row[class_label])

    if class_label == "angle":
        flipped_target_path = os.path.join(class_dir, str(new_label), f"{int(row['image_id'])}_flipped.png")
    else:
        flipped_target_path = os.path.join(class_dir, str(row[class_label]), f"{int(row['image_id'])}_flipped.png")
    save_image(flipped_image, flipped_target_path)

    # # Apply brightness and contrast adjustments based on augmentation_intensity
    # for i in range(int((augmentation_intensity-1)/2)):
    #     if class_label == 'angle':
    #         if random.random() < 0.1:
    #             image_path = flipped_target_path
    #         else:
    #             image_path = original_target_path
    #     else:
    #         if random.random() < 0.5:
    #             image_path = flipped_target_path
    #         else:
    #             image_path = original_target_path

    #     image = tf.io.read_file(image_path)
    #     image = tf.image.decode_png(image, channels=3)
    #     image = tf.cast(image, tf.float32) / 255.0  # Normalize image

    #     bright_image = tf_adjust_brightness(image)
    #     bright_image_path = image_path.replace('.png', f'_bright_{i}.png')
    #     tf.keras.preprocessing.image.save_img(bright_image_path, bright_image.numpy())

    #     contrast_image = tf_adjust_contrast(image)
    #     contrast_image_path = image_path.replace('.png', f'_contrast_{i}.png')
    #     tf.keras.preprocessing.image.save_img(contrast_image_path, contrast_image.numpy())

def restructuring_data(load_original, is_augment):
    data = pd.read_csv('../../training_norm.csv')
    data['angle'] = data['angle'] * 80 + 50
    data.loc[data['speed'] > 1, 'speed'] = 0
    
    image_dir = '../../training_data/training_data/'
    angle_class_dir = '../../data/angle_class_data'
    speed_class_dir = '../../data/speed_class_data'

    if load_original:
        save_original_images(angle_class_dir, speed_class_dir, image_dir, data)

    # Analyze class distribution for angle and speed
    angle_counts = count_images_in_directories(angle_class_dir)
    speed_counts = count_images_in_directories(speed_class_dir)
    max_count_angle = max(angle_counts.values())
    max_count_speed = max(speed_counts.values())
    print(angle_counts, speed_counts)

    angles_for_upscaling = identify_classes_for_augmentation(angle_counts)
    speed_for_upscaling = identify_classes_for_augmentation(speed_counts)
    print(angles_for_upscaling, speed_for_upscaling)

    if is_augment:
    # Apply augmentations based on class imbalance
        for _, row in data.iterrows():
            # angle_aug_intensity = max(1, int(max_count_angle / angle_counts.get(str(row['angle']), max_count_angle)))
            # speed_aug_intensity = max(1, int(max_count_speed / speed_counts.get(str(row['speed']), max_count_speed)))
            if str(180-row['angle']) in angles_for_upscaling:
                # Augment images for angle class
                if random.random() < 0.5:
                    augment_images_for_class(row, image_dir, angle_class_dir, "angle")

            # if str(row['speed']) in speed_for_upscaling:
            #     # Augment images for speed class
            #     augment_images_for_class(row, image_dir, speed_class_dir, "speed")


    print("Data restructuring complete.")

class GaussianBlurLayer(Layer):
    def __init__(self, kernel_size=(5, 5), sigma=0, **kwargs):
        super(GaussianBlurLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def call(self, inputs):
        # Wrap the cv2 GaussianBlur operation in tf.py_function to ensure compatibility with TensorFlow tensors
        def blur_function(image):
            if not isinstance(image, np.ndarray):
                print("Image is not a numpy array. Attempting conversion.")
            # image is a numpy array here
            blurred_image = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
            return blurred_image

        # Apply the blur function on each input
        blurred_inputs = tf.map_fn(lambda img: tf.numpy_function(blur_function, [img], tf.float32), inputs, fn_output_signature=tf.float32)
        blurred_inputs.set_shape(inputs.shape)  # Ensure output shape is set correctly
        return blurred_inputs

    def get_config(self):
        config = super(GaussianBlurLayer, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'sigma': self.sigma,
        })
        return config

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

if __name__ == "__main__":
    load_original = True
    is_augment = False
    restructuring_data(load_original, is_augment)

# def get_merged_df(data_dir, norm_csv_path):
#     # Read the normalized CSV data
#     df = pd.read_csv(norm_csv_path)

#     # Initialize lists to store the data
#     image_id = []
#     image_path = []
#     image_array = []
#     file_size = []

#     # List files in the specified directory
#     file_list = os.listdir(data_dir)

#     # Process each file in the directory
#     for filename in file_list:
#         # Read the image
#         im = cv2.imread(os.path.join(data_dir, filename))

#         # Append data to the lists
#         image_id.append(int(filename.split('.')[0]))
#         image_array.append(im)
#         image_path.append(os.path.join(data_dir, filename))
#         file_size.append(os.path.getsize(os.path.join(data_dir, filename)))

#     # Create a DataFrame from the collected data
#     data = {
#         'image_id': image_id,
#         'image': image_array,
#         'image_path': image_path,
#         'file_size': file_size
#     }
#     df_image = pd.DataFrame(data)

#     # Merge the DataFrame with the CSV data
#     merged_df = pd.merge(df, df_image, how='left', on='image_id')

#     # Clean the merged DataFrame
#     cleaned_df = merged_df[merged_df['speed'] <= 1]

#     # Return the cleaned and merged DataFrame
#     return cleaned_df







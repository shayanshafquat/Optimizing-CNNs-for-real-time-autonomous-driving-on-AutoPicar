import os
import cv2
import pandas as pd
import shutil
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer, MaxPooling2D
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight

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
    # clear_directory(angle_class_dir)
    # clear_directory(speed_class_dir)

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

def get_class_weights_for_angle_model(directory_path):
    directory = directory_path
    # Ensure we list only directories
    class_names = ['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0']
    # class_names = [dir_name for dir_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir_name))]
    class_counts = {}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        # Count only files, ignore subdirectories
        class_counts[class_name] = len([item for item in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, item))])

    # Calculating class weights
    classes = list(class_counts.keys())
    # Convert class names to indices (necessary if using `compute_class_weight`)
    class_indices = {class_name: index for index, class_name in enumerate(classes)}
    y = [class_indices[class_name] for class_name, count in class_counts.items() for _ in range(count)]

    # Use scikit-learn's compute_class_weight to calculate class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(weights))

    # Normalize the class weights
    total = sum(class_weights.values())
    normalized_class_weights = {k: v / total for k, v in class_weights.items()}

    print(class_counts)
    print("Normalized class weights:", normalized_class_weights)
    return normalized_class_weights

def get_class_weights_for_speed_model(directory_path):
    directory = directory_path
    # Ensure we list only directories
    class_names = ['0.0', '1.0']
    # class_names = [dir_name for dir_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir_name))]
    class_counts = {}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        # Count only files, ignore subdirectories
        class_counts[class_name] = len([item for item in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, item))])

    # Calculating class weights
    classes = list(class_counts.keys())
    # Convert class names to indices (necessary if using `compute_class_weight`)
    class_indices = {class_name: index for index, class_name in enumerate(classes)}
    y = [class_indices[class_name] for class_name, count in class_counts.items() for _ in range(count)]

    # Use scikit-learn's compute_class_weight to calculate class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(weights))
    # Normalize the class weights
    total = sum(class_weights.values())
    normalized_class_weights = {k: v / total for k, v in class_weights.items()}

    print(class_counts)
    print("Normalized class weights:", normalized_class_weights)
    return normalized_class_weights

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
    angle_class_dir = '../../data/angle_class_data_new2'
    speed_class_dir = '../../data/speed_class_data_new'

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



class RandomGaussianBlur(Layer):
    def __init__(self, kernel_size, factor, **kwargs):
        super(RandomGaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.factor = factor  # Assuming factor is either a float or a tuple of floats.

    def get_random_transformation(self, img_shape):
        # Calculate random factor, ensuring numerical stability
        if isinstance(self.factor, tuple):
            factor = tf.random.uniform((), self.factor[0], self.factor[1])
        else:
            factor = self.factor

        # Prevent division by zero or near-zero factors which might lead to numerical issues
        factor = tf.maximum(factor, K.epsilon())

        # Generate the Gaussian kernels
        kernel_v = self.get_kernel(factor, self.kernel_size)
        kernel_h = tf.transpose(kernel_v)

        kernel_v = tf.reshape(kernel_v, [self.kernel_size, 1, 1, 1])
        kernel_h = tf.reshape(kernel_h, [1, self.kernel_size, 1, 1])
        return kernel_v, kernel_h

    def call(self, inputs):
        blur_v, blur_h = self.get_random_transformation(tf.shape(inputs)[-3:])
        num_channels = tf.shape(inputs)[-1]
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])

        # Apply the blurring kernels
        blurred = tf.nn.depthwise_conv2d(inputs, blur_h, strides=[1, 1, 1, 1], padding="SAME")
        blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME")
        return blurred

    @staticmethod
    def get_kernel(factor, filter_size):
        """Generates a Gaussian kernel for use in RandomGaussianBlur."""
        range = tf.range(-filter_size // 2 + 1, filter_size // 2 + 1, dtype=tf.float32)
        gaussian_kernel = tf.exp(-0.5 * (range / factor) ** 2)
        gaussian_kernel = gaussian_kernel / tf.reduce_sum(gaussian_kernel)
        return gaussian_kernel

    def get_config(self):
        config = super(RandomGaussianBlur, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'factor': self.factor
        })
        return config

def categorical_focal_crossentropy(alpha, gamma=2.0):
    """
    Focal loss function for multi-class classification.
    Args:
        alpha (array-like): Array of shape (num_classes,) with class weights.
        gamma (float): Focusing parameter for modulating factor.
    Returns:
        loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

if __name__ == "__main__":
    load_original = False
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







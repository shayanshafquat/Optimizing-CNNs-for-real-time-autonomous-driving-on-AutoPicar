import tensorflow as tf
import os
import numpy as np
import cv2
from datetime import datetime
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import SpatialPyramidPooling, GaussianBlurLayer, get_class_weights_for_angle_model
from vit_keras import vit

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

# def preprocess_image(image, label):
#         # category_names = ['100.0', '105.0', '110.0', '115.0', '120.0', '125.0', '130.0', '50.0', '55.0', '60.0', '65.0', '70.0', '75.0', '80.0', '85.0', '90.0', '95.0']
#         # image = tf.image.crop_to_bounding_box(image, 0, 25, 224, 270)
#         image = tf.image.resize(image, [160, 160])          # Resize the cropped image to 160x160
#         # Convert the image to grayscale,  Replicate the grayscale image across three channels
#         # image = tf.image.rgb_to_yuv(image)
#         # image = tf.image.rgb_to_grayscale(image)
#         # image = tf.image.grayscale_to_rgb(image)

#         # image = tf.image.random_flip_left_right(image, seed=123)
#         return tf.cast(image, tf.float32), label

def preprocess_image(image, label):
    angles = tf.constant([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0], dtype=tf.float32)
    mirrored_angles = tf.constant([80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0, 85.0], dtype=tf.float32)
    image = tf.image.resize(image, [192, 192]) 
    image = tf.image.random_flip_left_right(image, seed=123)

    def adjust_label(label):
        # Assuming label is a batch of one-hot encoded labels with shape [batch_size, num_classes]
        # Find the indices of the max value in each row (highest probability class)
        label_indices = tf.argmax(label, axis=-1)
        
        # For each label index, find the corresponding angle and then find the mirrored angle's index
        angles_batch = tf.gather(angles, label_indices)
        
        # This operation needs to be adapted to handle a batch of angles
        # Use tf.map_fn to apply the finding operation across the batch
        def find_mirrored_index(angle):
            # Find the index of this angle in the mirrored_angles list
            mirrored_index = tf.where(tf.equal(mirrored_angles, angle))
            return tf.reshape(mirrored_index, [-1])[0]

        mirrored_indices = tf.map_fn(find_mirrored_index, angles_batch, dtype=tf.int64)
        
        # Create new one-hot encoded labels based on the mirrored indices
        new_labels = tf.one_hot(mirrored_indices, depth=tf.size(angles), dtype=label.dtype)
        
        return new_labels

    flipped_image = tf.image.flip_left_right(image)
    is_flipped = tf.reduce_any(flipped_image != image)
    
    # Use tf.cond to conditionally adjust the label, ensuring the output matches the input type
    label = tf.cond(is_flipped, lambda: adjust_label(label), lambda: tf.identity(label))
    
    return tf.cast(image, tf.float32), label


# Model definition with preprocessing included
def build_model(num_classes, model_name):

    # with tf.device('/cpu:0'):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.2, seed=123),
                                             tf.keras.layers.RandomContrast(0.2, seed=123),
                                            # tf.keras.layers.RandomFlip('horizontal', seed=123),
                                            # tf.keras.layers.CenterCrop(160,160),
                                            # tf.keras.layers.RandomZoom(0.1, fill_mode='nearest', seed=123),
                                            tf.keras.layers.GaussianNoise(0.1),
                                            # tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_hsv(x)),
                                            # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),

    ])

    if model_name == 'resnetv2':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = resnetv2_preprocess
    elif model_name == 'resnet':
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = resnet_preprocess
    elif model_name == 'vgg':
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = vgg16_preprocess

    # base_model = vit.vit_b16(
    #     image_size = 192,
    #     pretrained = True,
    #     include_top = False,
    #     pretrained_top = False,
    #     classes = num_classes)

    base_model.trainable = False  # Freeze base model

    inputs = Input(shape=(192,192,3))
    x = data_augmentation(inputs)
    x = GaussianBlurLayer(kernel_size=(5, 5), sigma=0, input_shape=(192, 192, 3))(x)
    x = preprocess_input(x) # Preprocessing for Resnet

    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    # x = SpatialPyramidPooling(pool_list=[1, 2, 4])(x) 
    # x = GlobalMaxPooling2D()(x)
    # x = Flatten()(x)
    # x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=model_name)

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
    return model

# Fine-tuning function
def fine_tune_model(model, checkpoint_path):
    print("Finetuning ...")

    # Unfreeze all layers in base model
    model.layers[5].trainable = True
    for layer in model.layers[5].layers[:-30]:
        layer.trainable = False
    for layer in model.layers[5].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
    model.summary()


if __name__ == "__main__":
    directory = '../../data/angle_class_data'
    finetuning = False
    
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

    model_name = 'resnetv2'
    model = build_model(17, model_name)
    model.summary()

    checkpoint_path = f"../../saved_models/angle/{model.name}_{ts}"
    model_checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch')

    if not finetuning:
        print("Base Learning")
        history = model.fit(train_ds,
                epochs=15,
                validation_data=val_ds,
                callbacks=[model_checkpoint_callback],
                verbose=1, class_weight=class_weights)

    # Load the best model for fine-tuning
    model = tf.keras.models.load_model(checkpoint_path)

    # Fine-tuning
    fine_tune_model(model, checkpoint_path)
    # Fine-tuning training with a smaller learning rate
    fine_tune_epochs = 8

    history_finetune = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=fine_tune_epochs,
                            callbacks=[model_checkpoint_callback],
                            verbose=1,
                            class_weight=class_weights)

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





    # def preprocess_image(image, label):
#     # Convert the TensorFlow tensor to a NumPy array
#     image_np = image.numpy()
    
#     # Resize the cropped image to 224x224
#     image_np = cv2.resize(image_np, (224, 224))
    
#     # Convert the image to grayscale
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
#     # Apply Gaussian Blur
#     image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
    
#     # Replicate the grayscale image across three channels to feed into ResNet
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
#     # Convert the NumPy array back to TensorFlow tensor
#     image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    
#     return image_tf, label






# angles = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
# mirrored_angles = [80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0, 85.0]

# angle_to_index = {angle: i for i, angle in enumerate(angles)} # Create a mapping from angle to its index
# angle_to_mirrored_index = {angle: angle_to_index[mirrored] for angle, mirrored in zip(angles, mirrored_angles)} # Create a mapping from angle to its mirrored angle index

# # TensorFlow constant for angles, to be used in TF operations
# angles_tensor = tf.constant(angles, dtype=tf.float32)
# # TensorFlow constant for mirrored indices
# mirrored_indices = [angle_to_index[angle] for angle in mirrored_angles]
# mirrored_indices_tensor = tf.constant(mirrored_indices, dtype=tf.int64)

# def preprocess_image(image, label):
#     image = tf.image.resize(image, [224, 224])  # Resize to 224x224
#     image = tf.image.rgb_to_yuv(image)
#     image = tf.cast(image, tf.float32)

    # # Randomly apply horizontal flip
    # flip = tf.random.uniform([]) < 0.5
    # image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)

    # def adjust_label(label):
    #     # Convert one-hot encoded label to index
    #     label_idx = tf.argmax(label, axis=-1)
        
    #     # Map index to its mirrored index
    #     def mirror_index(idx):
    #         # Use the TensorFlow gather operation to fetch the mirrored index
    #         return tf.gather(mirrored_indices_tensor, idx)
        
    #     # Adjust index for flip and convert back to one-hot encoding
    #     adjusted_idx = tf.cond(flip, lambda: mirror_index(label_idx), lambda: label_idx)
    #     return tf.one_hot(adjusted_idx, depth=len(angles))

    # label = adjust_label(label)
    # return image, label

# def calculate_class_weights(dataset):
#     # Initialize counts for each label to 0
#     label_counts = {}
#     num_labels = next(iter(dataset))[1].shape[1]  # Get the number of labels from the shape of the labels in the first batch

#     for i in range(num_labels):
#         label_counts[i] = 0

#     # Iterate over each batch in the dataset
#     for images, labels in dataset:
#         # Sum up the labels for the current batch. This works because labels are one-hot encoded.
#         labels_sum = tf.reduce_sum(labels, axis=0).numpy()
        
#         # Update counts in the dictionary
#         for i in range(num_labels):
#             label_counts[i] += labels_sum[i]

#     # Calculate the total count of images
#     total_count = sum(label_counts.values())

#     # Number of classes
#     num_classes = len(label_counts)

#     # Create a dictionary for class weights with indices as keys
#     class_weights = {}
#     for index, count in label_counts.items():
#         # Calculate the class weight
#         weight = total_count / (num_classes * count)
#         # Assign the weight using the index as key
#         class_weights[index] = weight

#     return class_weights
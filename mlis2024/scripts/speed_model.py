import tensorflow as tf
import os
from datetime import datetime
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input 
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import SpatialPyramidPooling, GaussianBlurLayer, RandomGaussianBlur
# from tensorflow.keras.utils import get_custom_objects
from utils import get_class_weights_for_speed_model
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

input_shape = (160,160,3)

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
# np.random.seed(123)

def preprocess_image(image, label):
        # image = tf.image.crop_to_bounding_box(image, 0, 25, 224, 270)
        image = tf.image.resize(image, [160, 160])          # Resize the cropped image to 160x160
        return tf.cast(image, tf.float32), label

# Model definition with preprocessing included
def build_model(num_classes, model_name, learning_rate):


    model = Sequential(name = model_name)
    # Data augmentation layers
    model.add(Input(shape=input_shape))  # Define input shape
    model.add(tf.keras.layers.RandomBrightness(0.15, seed=123))
    model.add(tf.keras.layers.RandomContrast(0.15, seed=123))
    # model.add(tf.keras.layers.GaussianNoise(10))
    # model.add(tf.keras.layers.CenterCrop(192,192))
    # model.add(RandomGaussianBlur(5,0.2))

    # # with tf.device('/cpu:0'):
    # data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.2, seed=123),
    #                                          tf.keras.layers.RandomContrast(0.2, seed=123),
    #                                         tf.keras.layers.RandomFlip('horizontal', seed=123),
    #                                         tf.keras.layers.CenterCrop(160,160),
    #                                         # tf.keras.layers.GaussianNoise(0.2),
    #                                         # tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=123),
    #                                         # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),

    # ])

    # Select the base model based on the provided model_name
    if model_name == 'resnetv2':
        base_model = tf.keras.applications.ResNet50(input_shape=(192, 192, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif model_name == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'mobilenetv3':
        base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    elif model_name == 'mobilenetv3L':
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    elif model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'vgg':
        base_model = tf.keras.applications.VGG16(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Add the base model
    base_model.trainable = False  # Freeze base model
    model.add(tf.keras.layers.Lambda(preprocess_input))  # Preprocessing using Lambda layer
    model.add(base_model)

    # Additional layer configurations - Uncomment as needed
    model.add(GlobalAveragePooling2D())  # Global average pooling layer
    # model.add(SpatialPyramidPooling(pool_list=[1, 2, 4]))
    # model.add(GlobalMaxPooling2D())
    # model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # Output layer

    model.add(Dense(num_classes, activation='sigmoid'))
    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='binary_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])
    return model, len(base_model.layers)

# Fine-tuning function
def fine_tune_model(model, checkpoint_path, learning_rate, layers_to_tune):
    # Unfreeze all layers in base model
    print("Finetuning ...")
    # learning_rate /= 10 
    print(f"Learning Rate:{learning_rate}")
    model.layers[3].trainable = True
    for layer in model.layers[3].layers[:-1*layers_to_tune]:
        layer.trainable = False
    if layers_to_tune < 50:
        for layer in model.layers[3].layers[:]:  # check for this
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='binary_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])
    model.summary()
    return learning_rate

def plot_combined_loss(all_histories, title, filename):
    plt.figure(figsize=(10, 6))
    epochs = sum(len(hist.history['loss']) for hist in all_histories)
    combined_epochs = list(range(1, epochs + 1))

    combined_loss = []
    combined_val_loss = []
    for hist in all_histories:
        combined_loss.extend(hist.history['loss'])
        combined_val_loss.extend(hist.history['val_loss'])

    plt.plot(combined_epochs, combined_loss, label='Training Loss')
    plt.plot(combined_epochs, combined_val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    
    directory = '../../data/speed_class_data'
    finetuning = False
    learning_rate = 0.001
    all_histories = []

    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='binary',
                class_names=['0.0','1.0'], 
                color_mode='rgb',
                batch_size=128,
                # image_size = (224, 224),
                shuffle=True,
                seed=123,
                validation_split=0.2,
                subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='binary',
                class_names=['0.0','1.0'], 
                color_mode='rgb',
                batch_size=128,
                # image_size = (224,224),
                shuffle=True,
                seed=123,
                validation_split=0.2,
                subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    ts = datetime.now().strftime('%Y%m%d')

    class_weights = get_class_weights_for_speed_model(directory)
    print("Class weights for binary classification:", class_weights)

    model_name = 'resnet'
    model, base_layers = build_model(1, model_name, learning_rate)
    # get_custom_objects().update({'RandomGaussianBlur': RandomGaussianBlur})
    model.summary()

    checkpoint_path = f"../../saved_models/speed/{model.name}_{ts}_{input_shape[0]}"

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
                             epochs=20,
                               validation_data=val_ds,
                                 callbacks=[model_checkpoint_callback],
                                   verbose=1,
                                   class_weight=class_weights)
        
        all_histories.append(history)

    layers_to_tune = [15, 30, 60]
    lr_tune = [0.00005, 0.00001, 0.000005]
    for i in range(len(layers_to_tune)):
        # Load the best model for fine-tuning
        model = tf.keras.models.load_model(checkpoint_path)

        # Fine-tuning
        learning_rate = fine_tune_model(model, checkpoint_path, lr_tune[i], layers_to_tune[i])
        # Fine-tuning training with a smaller learning rate
        fine_tune_epochs = 15
        total_epochs = fine_tune_epochs  # Initial epochs + fine-tune epochs

        history_finetune = model.fit(
                                train_ds,
                                validation_data=val_ds,
                                epochs=total_epochs,
                                callbacks=[model_checkpoint_callback],
                                verbose=1, class_weight=class_weights)
        
        all_histories.append(history_finetune)
        # if i == 0:
        #     plot_combined_loss(history, history_finetune, 'Combined Initial and Fine Tuning Loss (Speed)', '../plots/combined_training_fine_tuning_loss_speed.png')
        plot_combined_loss(all_histories, 'Combined Training and Validation Loss', '../plots/multi_phase_loss_speed.png')
        
    # # Evaluate the fine-tuned model
    # evaluate_model(model, valid_generator)
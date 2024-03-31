import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

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

class ImageClassificationModel:
    def __init__(self, num_classes, input_shape=(160, 160, 3), initial_epochs=10, fine_tune_epochs=5, dataset_path='path/to/your/dataset'):
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.dataset_path = dataset_path
        self.model = None
        self.history = []

        # Prepare datasets
        self.train_ds, self.val_ds = self.prepare_datasets()

    def prepare_datasets(self):
        """
        Prepares the training and validation datasets using the specified path.
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=64,
            image_size=(192, 256),
            shuffle=True,
            seed=123,
            validation_split=0.3,
            subset="training")

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=32,
            image_size=(192, 256),
            shuffle=True,
            seed=123,
            validation_split=0.3,
            subset="validation")

        return train_ds.map(self.preprocess_image).cache().prefetch(tf.data.AUTOTUNE), val_ds.map(self.preprocess_image).cache().prefetch(tf.data.AUTOTUNE)


    def preprocess_image(self, image, label):
        # Crop from the upper half and resize
        # Original size: 192x256
        # Crop to 160x256 from the upper half, meaning y=0, x=0, height=160, width=256
        image = tf.image.crop_to_bounding_box(image, 0, 0, 160, 256)
        # Resize the cropped image to 160x160
        image = tf.image.resize(image, [160, 160])
        return tf.cast(image, tf.float32), label

    def build_model(self):
        # with tf.device('/cpu:0'):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomBrightness(0.2, seed=123),
            tf.keras.layers.RandomContrast(0.2, seed=123),
            tf.keras.layers.RandomFlip('horizontal', seed=123),
            tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=123),
            # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),
            # tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_hsv(x)),
            ])

        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                     include_top=False,
                                                     weights='imagenet')
        base_model.trainable = False  # Freeze base model

        
        inputs = Input(shape=self.input_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.35)(x)
        # x = Dense(512, activation='relu')(x)
        # x = Dropout(0.35)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)

        # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_focal_crossentropy',
                           metrics=['accuracy',
                                     F1Score(),
                                     tf.metrics.MeanSquaredError(),
                                     tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])

    # def fine_tune_model(self):
    #     self.model.layers[4].trainable = True
    #     for layer in self.model.layers[4].layers[:-15]:
    #         layer.trainable = False

    #     optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.000001)
    #     self.model.compile(optimizer=optimizer,
    #                        loss='categorical_crossentropy',
    #                        metrics=['accuracy', F1Score(num_classes=self.num_classes)])

    #     self.model.summary()

    def fine_tune_model(self, unfreeze_layers, base_learning_rate=0.001):
        base_model = self.model.layers[4]
        base_model.trainable = True

        # Freeze all layers before the `unfreeze_layers` last layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Adjust the learning rate for fine-tuning
        learning_rate = base_learning_rate / 10
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='categorical_focal_crossentropy',
                           metrics=['accuracy',
                                     F1Score(),
                                     tf.metrics.MeanSquaredError(),
                                     tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
        self.model.summary()
        print(f"Model compiled with last {unfreeze_layers} layers unfrozen and learning rate adjusted to {learning_rate}.")

    def configure_callbacks(self):
        ts = datetime.now().strftime('%Y%m%d')
        checkpoint_path = f"models/angle/resnet_best_{ts}.h5"
        model_checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor='val_mean_squared_error',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            save_freq='epoch')
        return [model_checkpoint_callback]
    
    def train(self):
        # Initial training
        model_callbacks = self.configure_callbacks()
        print("Starting initial training...")
        self.history.append(self.model.fit(self.train_ds, epochs=self.initial_epochs, validation_data=self.val_ds, callbacks=model_callbacks, verbose=1))
        # Fine-tuning with progressively more layers unfrozen
        base_learning_rate = 0.001
        for layers_to_unfreeze in [5, 10, 20]:
            print(f"Fine-tuning with last {layers_to_unfreeze} layers unfrozen.")
            self.fine_tune_model(layers_to_unfreeze, base_learning_rate)
            fine_tune_history = self.model.fit(self.train_ds, epochs=self.fine_tune_epochs, validation_data=self.val_ds, callbacks=model_callbacks, verbose=1)
            self.history.append(fine_tune_history)
            base_learning_rate /= 10             # Reduce the base learning rate for the next fine-tuning step

    def plot_training_summary(self, save_path='training_results.png'):

        """
        Plots and saves the training and validation accuracy, loss, and MSE.
        """
        plt.figure(figsize=(18, 6))

        # Plot Accuracy
        plt.subplot(1, 3, 1)
        for i, history in enumerate(self.history):
            plt.plot(history.history['accuracy'], label=f'Train Acc - Step {i}')
            plt.plot(history.history['val_accuracy'], label=f'Val Acc - Step {i}')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 3, 2)
        for i, history in enumerate(self.history):
            plt.plot(history.history['loss'], label=f'Train Loss - Step {i}')
            plt.plot(history.history['val_loss'], label=f'Val Loss - Step {i}')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot MSE
        plt.subplot(1, 3, 3)
        for i, history in enumerate(self.history):
            plt.plot(history.history['mean_squared_error'], label=f'Train MSE - Step {i}')
            plt.plot(history.history['val_mean_squared_error'], label=f'Val MSE - Step {i}')
        plt.title('Model MSE')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Results saved to {save_path}")

    def plot_confusion_matrix(self):
        # Assuming self.model is already trained and val_ds is a batched dataset
        predictions = []
        true_labels = []
        for images, labels in self.val_ds.unbatch().batch(1):
            true_labels.append(np.argmax(labels.numpy(), axis=1)[0])
            preds = self.model.predict(images)
            predictions.append(np.argmax(preds, axis=1)[0])

        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes), rotation=45)
        plt.yticks(tick_marks, range(self.num_classes))
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    num_classes = 17  # Example: adjust according to your dataset
    input_shape = (160, 160, 3)  # Adjust if needed
    initial_epochs = 5
    fine_tune_epochs = 5
    dataset_path = 'angle_class_data'  # Update with your dataset path

    model = ImageClassificationModel(num_classes, input_shape, initial_epochs, fine_tune_epochs, dataset_path)
    model.build_model()
    model.train()
    model.plot_training_summary()
    model.plot_confusion_matrix()
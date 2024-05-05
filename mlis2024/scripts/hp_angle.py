import tensorflow as tf
import os
import numpy as np
import random
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from utils import get_class_weights_for_angle_model
import matplotlib.pyplot as plt


tf.random.set_seed(123)

def preprocess_image(image, label):
    # Resize the image to 100x100
    image = tf.image.resize(image, [128, 128])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    image = preprocess_input(image)
    return tf.cast(image, tf.float32), label

def build_model(hp):
    model = Sequential(name='mobilenetv2')
    input_shape = (128, 128, 3)
    model.add(Input(shape=input_shape))
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    model.add(base_model)

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))  #HP
    model.add(Dense(hp.Int('dense_units', min_value=256, max_value=1024, step=256), activation='relu'))   #HP
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_final', 0.2, 0.5, step=0.1)))  #HP
    model.add(Dense(17, activation='softmax'))

    learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, sampling='log')  # HP
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_focal_crossentropy', metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

    return model

def fine_tune_model(model, hp, base_layers):
    layers_to_tune = hp.Int('layers_to_tune', min_value=10, max_value=round(base_layers/3, -1), step=15)  #HP
    for layer in model.layers[0].layers[-layers_to_tune:]:
        layer.trainable = True
    learning_rate = hp.Float('fine_tune_learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')  #HP
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_focal_crossentropy', metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
    return model

def save_summary(tuner, phase):
    results = f"Summary for {phase}:\nBest Hyperparameters: {tuner.get_best_hyperparameters()[0].values}\n"
    results += "\nAll trials:\n"
    for trial in tuner.oracle.get_best_trials(num_trials=4):
        results += f"Trial {trial.trial_id}: Score={trial.score}, Hyperparameters={trial.hyperparameters.values}\n"
    with open(f'../hp/hyperparameter_summary_{phase}.txt', 'w') as f:
        f.write(results)

def plot_combined_loss(history_initial, history_fine, title='Combined Training and Validation Loss', filename='combined_loss_plot.png'):
    plt.figure(figsize=(10, 6))
    # Combine the history data
    epochs_initial = len(history_initial.history['loss'])
    epochs_fine = len(history_fine.history['loss'])
    epochs_total = epochs_initial + epochs_fine

    # Create combined lists for loss and validation loss
    combined_loss = history_initial.history['loss'] + history_fine.history['loss']
    combined_val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
    combined_epochs = list(range(1, epochs_total + 1))

    # Plotting
    plt.plot(combined_epochs, combined_loss, label='Training Loss')
    plt.plot(combined_epochs, combined_val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)  # Save the plot to a file
    plt.close()

def run_hyperparameter_tuning(train_ds, val_ds):
    tuner_initial = RandomSearch(build_model, 
                                 objective='val_loss', 
                                 max_trials=10, 
                                 executions_per_trial=1, 
                                 directory='tuner_results', 
                                 project_name='InitialTuning')
    tuner_initial.search(train_ds, epochs=20, validation_data=val_ds)

    tuner_initial.results_summary(num_trials=10)
    save_summary(tuner_initial, "InitialTraining")

    best_hp_initial = tuner_initial.get_best_hyperparameters()[0]
    model = build_model(best_hp_initial)
    history_initial = model.fit(train_ds, epochs=30, validation_data=val_ds)

    base_layers = len(model.layers[0].layers)

    tuner_fine = RandomSearch(lambda hp: fine_tune_model(model, hp, base_layers), 
                              objective='val_loss', 
                              max_trials=10, 
                              executions_per_trial=1, 
                              directory='tuner_results', 
                              project_name='FineTuning')
    tuner_fine.search(train_ds, epochs=15, validation_data=val_ds)

    tuner_fine.results_summary(num_trials=10)
    save_summary(tuner_fine, "FineTuning")

    best_hp_fine = tuner_fine.get_best_hyperparameters()[0]
    model = fine_tune_model(model, best_hp_fine, base_layers)
    history_fine = model.fit(train_ds, epochs=20, validation_data=val_ds)
    # Plot the combined loss after both initial training and fine-tuning
    plot_combined_loss(history_initial, history_fine, 'Combined Initial and Fine Tuning Loss', '../plots/combined_training_fine_tuning_loss.png')



if __name__ == "__main__":
    directory = '../../data/angle_class_data'
    train_ds = tf.keras.utils.image_dataset_from_directory(directory,
                                                            labels='inferred',
                                                            label_mode='categorical', 
                                                            class_names=['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'], 
                                                            color_mode='rgb', 
                                                            batch_size=128, 
                                                            shuffle=True, 
                                                            seed=123, 
                                                            validation_split=0.15, 
                                                            subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(directory, 
                                                         labels='inferred', 
                                                         label_mode='categorical', 
                                                         class_names=['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'], 
                                                         color_mode='rgb', 
                                                         batch_size=64, 
                                                         shuffle=True, 
                                                         seed=123, 
                                                         validation_split=0.15, 
                                                         subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)
    run_hyperparameter_tuning(train_ds, val_ds)
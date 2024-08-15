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
from tensorflow.keras.optimizers import Adam, RMSprop
from keras_tuner import RandomSearch
from utils import get_class_weights_for_angle_model, SpatialPyramidPooling
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K



tf.random.set_seed(123)

# @tf.keras.utils.register_keras_serializable()
# def categorical_focal_crossentropy(alpha, gamma=2.0):
#     """
#     Focal loss function for multi-class classification.
#     Args:
#         alpha (array-like): Array of shape (num_classes,) with class weights.
#         gamma (float): Focusing parameter for modulating factor.
#     Returns:
#         loss function
#     """
#     def focal_loss_fixed(y_true, y_pred):
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         cross_entropy = -y_true * K.log(y_pred)
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
#         return K.sum(loss, axis=1)
#     return focal_loss_fixed

# def get_class_weights(class_names):
#     class_counts = {}
#     for class_name in class_names:
#         class_dir = os.path.join(directory, class_name)
#         class_counts[class_name] = len([item for item in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, item))])
#     labels = list(class_counts.keys())
#     weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=[label for label, count in class_counts.items() for _ in range(count)])
#     normalized_weights = weights / np.sum(weights)
#     return {label: weight for label, weight in zip(labels, normalized_weights)}

def preprocess_image(image, label):
    # Resize the image to 100x100
    image = tf.image.resize(image, [128, 128])
    # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # image = preprocess_input(image)
    return tf.cast(image, tf.float32), label

def build_model(hp):
    class_names = ['65.0', '50.0', '75.0', '115.0', '130.0', '85.0', '105.0', '120.0', '95.0', '80.0', '110.0', '125.0', '90.0', '100.0', '60.0', '70.0', '55.0']
    # class_weights = get_class_weights(class_names)
    # alpha_tensor = tf.constant([class_weights[name] for name in class_names], dtype=tf.float32)

    model = Sequential(name='mobilenetv2')
    input_shape = (128, 128, 3)

    model.add(Input(shape=input_shape))
    model.add(tf.keras.layers.RandomBrightness(hp.Float('brightness', min_value=0.1, max_value=0.3, step=0.1, default=0.2), seed=123))
    model.add(tf.keras.layers.RandomContrast(hp.Float('contrast', min_value=0.1, max_value=0.3, step=0.1, default=0.2), seed=123))

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    model.add(tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input))
    model.add(base_model)

    # Choose pooling type based on hyperparameter
    pooling_type = hp.Choice('pooling_type', ['avg', 'max', 'spp'])
    if pooling_type == 'avg':
        model.add(GlobalAveragePooling2D())
    elif pooling_type == 'max':
        model.add(GlobalMaxPooling2D())
    elif pooling_type == 'spp':
        model.add(SpatialPyramidPooling(pool_list=[1, 2, 4]))  # Ensure this layer is correctly implemented or available

    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units', min_value=256, max_value=1024, step=256), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_final', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(17, activation='softmax'))
    
    # Choose optimizer based on hyperparameter
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, sampling='log')
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                loss='categorical_focal_crossentropy', 
                metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])

    return model

def fine_tune_model(model, hp, base_layers, best_optimizer):
    layers_to_tune = hp.Int('layers_to_tune', min_value=10, max_value=round(base_layers/3, -1), step=15)  #HP
    for layer in model.layers[3].layers[:-layers_to_tune]:
        layer.trainable = True
    for layer in model.layers[3].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    learning_rate = hp.Float('fine_tune_learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')  #HP
    if best_optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif best_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                loss='categorical_focal_crossentropy', 
                metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])
    model.summary()
    return model

def save_summary(tuner, phase):
    results = f"Summary for {phase}:\nBest Hyperparameters: {tuner.get_best_hyperparameters()[0].values}\n"
    results += "\nAll trials:\n"
    for trial in tuner.oracle.get_best_trials(num_trials=10):
        results += f"Trial {trial.trial_id}: Score={trial.score}, Hyperparameters={trial.hyperparameters.values}\n"
    with open(f'../hp/hyperparameter_summary_{phase}_v2.txt', 'w') as f:
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
    class_weights = get_class_weights_for_angle_model(directory)
    print("Class weights for multi classification:", class_weights)
    class_names = ['65.0', '50.0', '75.0', '115.0', '130.0', '85.0', '105.0', '120.0', '95.0', '80.0', '110.0', '125.0', '90.0', '100.0', '60.0', '70.0', '55.0']
    # class_weights_1 = get_class_weights(class_names)
    # alpha_tensor = tf.constant([class_weights_1[name] for name in class_names], dtype=tf.float32)

    # early_stopping = EarlyStopping(
    # monitor='val_loss',
    # patience=10,  # Adjust this based on how long you are willing to wait for an improvement
    # restore_best_weights=True,
    # verbose=1
    # )

    # tuner_initial = RandomSearch(build_model, 
    #                              objective='val_loss', 
    #                              max_trials=36, 
    #                              executions_per_trial=1, 
    #                              directory='tuner_results_angle_v2', 
    #                              project_name='InitialTuning')
    # tuner_initial.search(train_ds, epochs=25, validation_data=val_ds, class_weight=class_weights, callbacks=[early_stopping])

    # tuner_initial.results_summary(num_trials=36)
    # save_summary(tuner_initial, "InitialTraining")

    # best_hp_initial = tuner_initial.get_best_hyperparameters()[0]
    # best_optimizer = best_hp_initial.get('optimizer')
    best_optimizer = 'rmsprop'
    # # best_gamma = best_hp_initial.get('gamma')

    # model_after_initial = build_model(best_hp_initial)
    # early_stopping = EarlyStopping(
    # monitor='val_loss',
    # patience=10,  # Adjust this based on how long you are willing to wait for an improvement
    # restore_best_weights=True,
    # verbose=1
    # )
    # history_initial = model_after_initial.fit(train_ds, epochs=25, validation_data=val_ds, class_weight=class_weights, callbacks=[early_stopping])
    # model_after_initial.save('best_angle_model_initial', save_format='tf')  # Save the best model directory in .pb format

    model_to_fine_tune = tf.keras.models.load_model('best_angle_model_initial')

    base_layers = len(model_to_fine_tune.layers[3].layers)

    tuner_fine = RandomSearch(lambda hp: fine_tune_model(model_to_fine_tune, hp, base_layers, best_optimizer), 
                              objective='val_loss', 
                              max_trials=12, 
                              executions_per_trial=1, 
                              directory='tuner_results_angle_v2', 
                              project_name='FineTuning')
    early_stopping_fine = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    tuner_fine.search(train_ds, epochs=25, validation_data=val_ds, class_weight=class_weights, callbacks=[early_stopping_fine])

    tuner_fine.results_summary(num_trials=12)
    save_summary(tuner_fine, "FineTuning")

    best_hp_fine = tuner_fine.get_best_hyperparameters()[0]
    final_model = fine_tune_model(model_to_fine_tune, best_hp_fine, base_layers, best_optimizer)
    early_stopping_fine = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    history_fine = final_model.fit(train_ds, epochs=25, validation_data=val_ds, class_weight=class_weights, callbacks=[early_stopping_fine])
    # Plot the combined loss after both initial training and fine-tuning
    plot_combined_loss(history_initial, history_fine, 'Combined Initial and Fine Tuning Loss (Angle)', '../plots/combined_training_fine_tuning_loss_angle_v2.png')

directory = '../../data/angle_class_data'

if __name__ == "__main__":
    train_ds = tf.keras.utils.image_dataset_from_directory(directory,
                                                            labels='inferred',
                                                            label_mode='categorical', 
                                                            class_names=['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'], 
                                                            color_mode='rgb', 
                                                            batch_size=64, 
                                                            shuffle=True, 
                                                            seed=123, 
                                                            validation_split=0.2, 
                                                            subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(directory, 
                                                         labels='inferred', 
                                                         label_mode='categorical', 
                                                         class_names=['65.0','50.0','75.0','115.0','130.0','85.0','105.0','120.0','95.0','80.0','110.0','125.0','90.0','100.0','60.0','70.0','55.0'], 
                                                         color_mode='rgb', 
                                                         batch_size=64, 
                                                         shuffle=True, 
                                                         seed=123, 
                                                         validation_split=0.2, 
                                                         subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)
    run_hyperparameter_tuning(train_ds, val_ds)
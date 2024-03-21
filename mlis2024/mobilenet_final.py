import tensorflow as tf
import os
from datetime import datetime
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

tf.random.set_seed(123)
# np.random.seed(123)

# Model definition with preprocessing included
def build_model(num_classes):

    # with tf.device('/cpu:0'):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.1, seed=123),
                                            tf.keras.layers.RandomFlip('horizontal', seed=123),
                                            tf.keras.layers.CenterCrop(224,224),
                                            tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=123),
                                            # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),

    ])

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model

    inputs = Input(shape=(240,240,3))
    aug_inputs = data_augmentation(inputs)
    x = preprocess_input(aug_inputs)  # Preprocessing for MobileNetV2
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)


    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', Precision(), Recall(), tf.metrics.MeanSquaredError()])
    return model


directory = 'angle_class_data'
train_ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=64,
            image_size = (240,240),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset="training")

val_ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=64,
            image_size = (240,240),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset="validation")

model = build_model(17)
model.summary()

ts = datetime.now().strftime('%Y%m%d')
checkpoint_path = f"models/angle/mobilenet_v1_{ts}"

model_checkpoint_callback = ModelCheckpoint(
checkpoint_path,
monitor='val_loss',     # Monitor validation loss
verbose=1,              # Log a message each time the callback saves the model
save_best_only=True,    # Only save the model if 'val_loss' has improved
save_weights_only=False, # Only save the weights of the model
mode='min',             # 'min' means the monitored quantity should decrease
save_freq='epoch')       # Check every epoch

history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[model_checkpoint_callback], verbose=1)
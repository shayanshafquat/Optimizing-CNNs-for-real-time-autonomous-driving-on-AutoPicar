# from tensorflow import keras
# import tensorflow as tf
# import numpy as np
# # import imutils
# import cv2
# import os
# from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
# from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


import numpy as np
import tensorflow as tf
import os

class Model:

    saved_speed_model = 'speed_model/mobilenetv2_20240503'
    saved_angle_model = 'angle_model/mobilenetv2_20240502'
    def __init__(self):
        self.speed_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))
        self.angle_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [192, 192])
        im = tf.expand_dims(im, axis=0)
        return im
    
    
    def predict(self, image):
        angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
        image = self.preprocess(image)
        
        pred_speed = self.speed_model.predict(image)[0]
        pred_speed = np.round(pred_speed[0]).astype(int)
        speed = pred_speed.astype(int)*35
        pred_angle = self.angle_model.predict(image)[0]
        angle = angles[np.argmax(pred_angle)]
        print('angle:', angle,'speed:', speed)
        
        return angle, speed
    

    # def predict_angle(image_path, model):
    #     angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
    #     image = preprocess(image_path)
    #     angle = model.predict(image)[0]
    #     pred_angle = angles[np.argmax(angle)]
    #     return (pred_angle - 50 )/80

    # def predict_speed(image, model):
    #     # image = preprocess_speed_image(image_path)
    #     speed = model.predict(image)[0]
    #     # print(speed)
    #     # Assuming binary classification for speed, adjust as necessary
    #     speed_pred = np.round(speed[0]).astype(int)
    #     return speed_pred

# class Model:

#     speed = 'models/speed/lane_navigation_check.h5'
#     angle = 'models/angle/angle_navigation.h5'

#     def __init__(self):
#         self.speed_model = keras.models.load_model(self.speed)
#         # self.angle_model = keras.models.load_model(self.angle_model)
#         # self.model.summary()
#             # Create the model
#         self.angle_model = self.nvidia_model()
    
#         # Load the weights
#         self.angle_model.load_weights(self.angle)

#     def preprocess(self, image):
#         # height, _, _ = image.shape
#         # image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
#         image = cv2.GaussianBlur(image, (3,3), 0)
#         image = cv2.resize(image, (200,200)) # input image size (200,66) Nvidia model
#         image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
#         return image


#     def nvidia_model(self):
#         model = Sequential(name='Nvidia_Model')
#         model.add(tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False, weights="imagenet", input_shape=(200, 200, 3)))
#         model.add(tf.keras.layers.GlobalAveragePooling2D())
#         model.add(Dense(1, activation = 'sigmoid'))
#         model.layers[0].trainable = False

#         model.compile(optimizer=tf.optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics='accuracy')

#         return model

#     def predict(self, image):
#         image = self.preprocess(image)
#         speed = self.speed_model.predict(np.array(image))[0]
#         angle = self.angle_model.predict(np.array(image))[0]
#         # angle, speed = self.model.predict(np.array([image]))[0]
#         # Training data was normalised so convert back to car units
#         angle = 80 * np.clip(angle, 0, 1) + 50
#         speed = 35 * np.clip(speed, 0, 1)
#         return angle, speed
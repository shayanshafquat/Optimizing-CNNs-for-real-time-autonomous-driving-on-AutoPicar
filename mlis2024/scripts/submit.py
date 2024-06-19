from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

speed_resize_shape = [128,128]
angle_resize_shape = [128,128]

model_name = 'mobilenetv2'

def preprocess_image(image):
    # Resize the image to 160x160
    image = tf.image.resize(image, [160, 160])

    # # Apply random brightness and contrast adjustments
    # image = tf.image.random_brightness(image, max_delta=0.2, seed=123)
    # image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=123)

    # # Preprocessing input based on the model
    # if model_name == 'mobilenetv2':
    #     preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # else:
    #     preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    # # Apply the specific preprocessing
    # image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return tf.cast(image, tf.float32)

# def preprocess_speed_image(image_path):
#         # img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(192,256))
#         image = tf.keras.utils.load_img(image_path, target_size=(224,224))
#         # img_array = tf.keras.preprocessing.image.img_to_array(img)
#         # img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#         # image = tf.image.crop_to_bounding_box(image, 0, 25, 224, 270)
#         image = tf.image.resize(image, [160, 160])          # Resize the cropped image to 160x160
#         image = tf.cast(image, tf.float32)
#         # im = tf.image.convert_image_dtype(img, tf.float32)
#         image = tf.expand_dims(image, axis=0)
#         return image

# def preprocess_angle(image_path):
#     image = tf.keras.utils.load_img(image_path, target_size=(224,224))
#     im = tf.cast(image, tf.float32)
#     im = tf.image.resize(im, [160, 160])
#     im = tf.expand_dims(im, axis=0)
#     return im

def predict_speed(image, model):
    image = preprocess_image(image)
    speed = model.predict(image)[0]
    # print(speed)
    # Assuming binary classification for speed, adjust as necessary
    speed_pred = np.round(speed[0]).astype(int)
    return speed_pred

def predict_angle(image, model):
    angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
    image = preprocess_image(image)
    angle = model.predict(image)[0]
    pred_angle = angles[np.argmax(angle)]
    return (pred_angle - 50 )/80

# def predict_angle_reg(image_path, model):
#     # angles = [65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0]
#     # angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
#     standard_angles = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130])
#     normalized_angles = (standard_angles - 50) / 80
#     image = preprocess(image_path)

#     angle = model.predict(image)[0]
#     print(model.predict(image))
#     idx = (np.abs(normalized_angles - angle)).argmin()
#     pred_angle = normalized_angles[idx]
#     # pred_angle = angles[np.argmax(angle)]
#     # print(pred_angle)
#     return pred_angle

timestamp = datetime.now().strftime('%Y%m%d')

angle_path = '../../saved_models/angle/resnetv2'
speed_path = '../../saved_models/speed/vgg'


angle_model = load_model(angle_path)
speed_model = load_model(speed_path)
print(angle_model.summary())
print(speed_model.summary())

# TO CREATE PREDICTIONS OF THE TEST DATA
test_dir = os.path.abspath('../../test_data/test_data')
file_list = os.listdir(test_dir)
predictions = []

for file_name in sorted(file_list, key=lambda x: int(x.split('.')[0])):
    img_path = os.path.join(test_dir, file_name)
    image_id = int(file_name.split('.')[0])
    
    image = tf.keras.utils.load_img(img_path)

    angle_pred = predict_angle(image, angle_model)
    speed_pred = predict_speed(image, speed_model)
    
    print(f"Image ID: {image_id}, Angle: {angle_pred}, Speed: {speed_pred}")
    predictions.append([image_id, angle_pred.numpy(), speed_pred])
    # print(f"Image ID: {image_id}, Speed: {speed_pred}")
    # predictions.append([image_id, speed_pred])

df_pred = pd.DataFrame(predictions, columns=['image_id', 'angle', 'speed'])
df_pred.to_csv(f'submission_mobilenetv2_{timestamp}_new.csv', index=False)




# def preprocess_angle_image(image_path):
#         # image = tf.image.crop_to_bounding_box(image, 0, 25, 224, 270)
#         img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(192,256))
#         # im = tf.cast(img, tf.float32)
#         # img_array = tf.keras.preprocessing.image.img_to_array(img)
#         # img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#         im = tf.image.resize(img, [192, 192])          # Resize the cropped image to 160x160
#         im = tf.image.convert_image_dtype(im, tf.float32)
#         im = tf.expand_dims(im, axis=0)
#         return im



# def preprocess_angle(image_path):
#     image = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(192,256), keep_aspect_ratio=True)
#     image_tensor = tf.keras.preprocessing.image.img_to_array(image)
#     im = tf.image.resize(image_tensor, [192, 192])
#     # im = tf.image.grayscale_to_rgb(im)
#     im = tf.cast(im, tf.float32)
#     im = tf.expand_dims(im, axis=0)
#     return im

# def preprocess_speed(image_path):
#     image = tf.keras.utils.load_img(image_path, target_size=(192,256), keep_aspect_ratio=True)
#     image_tensor = tf.keras.preprocessing.image.img_to_array(image)
#     im = tf.image.resize(image_tensor, [192, 192])
#     im = tf.cast(im, tf.float32)
#     im = tf.expand_dims(im, axis=0)
#     return im

# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(192, 192))
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return img_array_expanded_dims
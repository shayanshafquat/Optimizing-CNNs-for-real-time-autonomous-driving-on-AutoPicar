import tensorflow as tf
import numpy as np
import os
import pandas as pd
from datetime import datetime

tflite_angle_path = '../../tflite_saved_models/angle/mobilenetv3_20240504_160.tflite'
tflite_speed_path = '../../tflite_saved_models/speed/mobilenetv3_20240504_160.tflite'

angle_interpreter = tf.lite.Interpreter(tflite_angle_path)
speed_interpreter = tf.lite.Interpreter(tflite_speed_path)
angle_interpreter.allocate_tensors()
speed_interpreter.allocate_tensors()

# Get input and output tensors.
angle_input_details = angle_interpreter.get_input_details()
angle_output_details = angle_interpreter.get_output_details()
speed_input_details = speed_interpreter.get_input_details()
speed_output_details = speed_interpreter.get_output_details()

def preprocess(image):
    im = tf.image.convert_image_dtype(image, tf.float32)
    im = tf.image.resize(im, [160, 160])
    im = tf.expand_dims(im, axis=0) #add batch dimension
    return im

def predict(image):
    # angles = np.arange(17)*5+50
    angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
    image = preprocess(image)

    # self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)
    angle_interpreter.set_tensor(angle_input_details[0]['index'], image)
    speed_interpreter.set_tensor(speed_input_details[0]['index'], image)

    # self.speed_interpreter.invoke()
    angle_interpreter.invoke()
    speed_interpreter.invoke()
    
    pred_angle = angle_interpreter.get_tensor(angle_output_details[0]['index'])[0]
    angle = angles[np.argmax(pred_angle)]
    
    pred_speed = speed_interpreter.get_tensor(speed_output_details[0]['index'])[0]
    speed = np.around(pred_speed[0]).astype(int)
    print(pred_speed)
    return (angle - 50 )/80, speed

test_dir = os.path.abspath('../../test_data/test_data')
file_list = os.listdir(test_dir)
predictions = []
timestamp = datetime.now().strftime('%Y%m%d')

for file_name in sorted(file_list, key=lambda x: int(x.split('.')[0])):
    img_path = os.path.join(test_dir, file_name)
    image_id = int(file_name.split('.')[0])
    
    image = tf.keras.utils.load_img(img_path, target_size=(224,224))
    angle_pred, speed_pred = predict(image)
    
    print(f"Image ID: {image_id}, Angle: {angle_pred}, Speed: {speed_pred}")
    predictions.append([image_id, angle_pred.numpy(), speed_pred])


df_pred = pd.DataFrame(predictions, columns=['image_id', 'angle', 'speed'])
df_pred.to_csv(f'submission_tf_lite_v3{timestamp}.csv', index=False)
# import tensorflow as tf

# angle_path = '../../saved_models/angle/mobilenetv2_20240504_160'
# # converter = tf.lite.TFLiteConverter.from_keras_model(angle_model)
# # Assuming angle_path points to the saved directory of your trained model
# converter = tf.lite.TFLiteConverter.from_saved_model(angle_path)

# # To ensure that all ops are compatible with TensorFlow Lite
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# # Convert the model
# try:
#     tflite_model = converter.convert()
# except Exception as e:
#     print(e)

import tensorflow as tf

angle_path = '../../saved_models/angle/mobilenetv3_20240504_160'
speed_path = '../../saved_models/speed/mobilenetv3_20240504_160'
# converter = tf.lite.TFLiteConverter.from_keras_model(angle_model)
# Assuming angle_path points to the saved directory of your trained model
angle_converter = tf.lite.TFLiteConverter.from_saved_model(angle_path)
speed_converter = tf.lite.TFLiteConverter.from_saved_model(speed_path)
# To ensure that all ops are compatible with TensorFlow Lite
angle_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
speed_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Convert the model
try:
    tflite_angle_model = angle_converter.convert()
except Exception as e:
    print(e)

# Convert the model
try:
    tflite_speed_model = speed_converter.convert()
except Exception as e:
    print(e)

tflite_angle_path = '../../tflite_saved_models/angle/mobilenetv3_20240504_160.tflite'
tflite_speed_path = '../../tflite_saved_models/speed/mobilenetv3_20240504_160.tflite'

# Save the quantized TFLite model
with open(tflite_angle_path, 'wb') as f:
    f.write(tflite_angle_model)

# Save the quantized TFLite model
with open(tflite_speed_path, 'wb') as f:
    f.write(tflite_speed_model)
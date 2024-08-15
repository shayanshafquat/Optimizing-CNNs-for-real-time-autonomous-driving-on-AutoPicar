import numpy as np
import tensorflow as tf
import os

class Model:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #enable gpu
    speed_model = 'speed_model/mobilenetv2_20240504_160.tflite'
    angle_model = 'angle_model/mobilenetv2_20240504_160.tflite'

    def __init__(self):
        
        # try:
        #     delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1') #'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
        #     print('Using TPU')
            
        #     self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                                                      self.speed_model),experimental_delegates=[delegate])
        #     self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                                                      self.angle_model), experimental_delegates=[delegate])
          
        # except ValueError:
        print('Fallback to CPU')

        self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                            self.speed_model))
        self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                            self.angle_model))
        

        self.speed_interpreter.allocate_tensors()
        self.angle_interpreter.allocate_tensors()
        self.speed_input_details = self.speed_interpreter.get_input_details()
        self.speed_output_details = self.speed_interpreter.get_output_details()
        self.angle_input_details = self.angle_interpreter.get_input_details()
        self.angle_output_details = self.angle_interpreter.get_output_details()
        # self.floating_model = self.speed_input_details[0]['dtype'] == np.float32         # check the type of the input tensor

    # def preprocess(self, image):
    #     im = tf.image.convert_image_dtype(image, tf.float32)
    #     im = tf.image.resize(im, [100, 100])
    #     im = tf.expand_dims(im, axis=0) #add batch dimension
    #     return im
    
    def preprocess(self, image):
        # im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(image, [160, 160])
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        im = preprocess_input(im)
        im = tf.expand_dims(im, axis=0) #add batch dimension
        return im

    # def predict(self, image):
    #     angles = np.arange(17)*5+50
    #     image = self.preprocess(image)

    #     self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)
    #     self.angle_interpreter.set_tensor(self.angle_input_details[0]['index'], image)

    #     self.speed_interpreter.invoke()
    #     self.angle_interpreter.invoke()

    #     pred_speed = self.speed_interpreter.get_tensor(self.speed_output_details[0]['index'])[0]
    #     speed = np.around(pred_speed[0]).astype(int)*35

    #     pred_angle = self.angle_interpreter.get_tensor(self.angle_output_details[0]['index'])[0]
    #     angle = angles[np.argmax(pred_angle)]
        
    #     return angle, speed
    
    def predict(self, image):
        # angles = np.arange(17)*5+50
        angles = tf.constant([65.0, 50.0, 75.0, 115.0, 130.0, 85.0, 105.0, 120.0, 95.0, 80.0, 110.0, 125.0, 90.0, 100.0, 60.0, 70.0, 55.0], dtype=tf.float32)
        image = self.preprocess(image)
        # im = tf.cast(image, tf.float32)
        # self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)
        self.angle_interpreter.set_tensor(self.angle_input_details[0]['index'], image)
        self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)

        # self.speed_interpreter.invoke()
        self.angle_interpreter.invoke()
        self.speed_interpreter.invoke()
        
        pred_angle = self.angle_interpreter.get_tensor(self.angle_output_details[0]['index'])[0]
        angle = angles[np.argmax(pred_angle)]
        
        pred_speed = self.speed_interpreter.get_tensor(self.speed_output_details[0]['index'])[0]
        speed = np.around(pred_speed[0]).astype(int)*35

        return angle, speed

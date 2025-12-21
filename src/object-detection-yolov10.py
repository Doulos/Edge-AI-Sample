import sys
import time
import cv2
import paho.mqtt.client as mqtt
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# Broker_address for prototyping MQTT communications
broker_address = "test.mosquitto.org"
# In case "test.mosquitto.org" is not working, use the broker from HiveMQ mentioned below
#broker_address = "broker.hivemq.com"

#Create new MQTT client instance
client = mqtt.Client()

#Connect to broker. Broker can be located on edge or cloud
client.connect(broker_address, 1883, 60)


# Function to load labels from a file
def load_labels(file_path):
    with open(file_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Load labels from the provided text file
label2string = load_labels('src/coco_labels.txt')

def detect_from_image():
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    
    success, img_org = cap.read()
    #     if not success:
    #        sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

    # prepare input image
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    input_tensor = np.asarray(img, dtype=np.float32)

    # Normalize pixel values to be between 0 and 1.0
    input_tensor /= 255.0

    # Add batch dimension (1, 640, 640, 3) as expected by the TFLite model
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Overview of Object Detection - YOLOv10 model: https://docs.ultralytics.com/models/yolov10/
   
    interpreter = Interpreter(
        model_path="src/yolov10n_float16.tflite")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # execute model graph using LiteRT
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    first_detection = output_data[0,0]
    
    # get output tensor details for first detection [0]
    x, y, w, h, confidence, class_id = first_detection

    output_detection = str(f' "Label:", {class_id}, {label2string[class_id]}, ",Score:", {confidence} ", Image coordinates:", {x,y,w,h}' )

    return output_detection


def publish_inference():
    detection = detect_from_image()
    print("Payload being sent:", detection)
    client.publish("object/type/location", payload=detection)
    print ('Inference published to MQTT topic')

if __name__ == '__main__':
    client.loop_start()
    publish_inference()
    time.sleep(5)
    publish_inference()
    client.disconnect()
    client.loop_stop()
    print("MQTT client disconnected and loop stopped.")
    sys.exit(0) # Exit the Python program    


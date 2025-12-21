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
client.connect(broker_address)

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
    start = time.time()
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1],
                      img.shape[2])  # (1, 300, 300, 3)
    img = img.astype(np.uint8)

    # Overview of Object Detection: https://www.tensorflow.org/lite/examples/object_detection/overview
    # Load pretrained model:https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/default/1

    interpreter = Interpreter(
        model_path="src/ssd_mobilenet_v1.tflite")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # execute model graph using LiteRT
    interpreter.invoke()


    # get output tensor details
    boxes = interpreter.get_tensor(output_details[0]['index'])
    boxes_shape = output_details[0]['shape_signature']
    labels = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])
    labels_list = labels.tolist()
    boxes = np.array(boxes, dtype=np.float32)
    
    output_detection = str(f' "Label:", {label2string[labels[0][0]]}, ",Score:", {scores[0][0]}, ",Bounding box coodinates:" , {boxes[0][0]}')

    return output_detection


def publish_inference():
    bounding_box = detect_from_image()
    print("Payload being sent:", bounding_box)
    client.publish("object/type/location", payload=bounding_box)
    print ('Inference published to MQTT topic')

if __name__ == '__main__':
    client.loop_start()
    publish_inference()
    time.sleep(5)
    # Keep the client connected and processing messages
    client.loop_forever()


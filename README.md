Create a Docker image on Arduino UNO Q that runs an object detection model using LiteRT.
The hardware setup consists of a Webcam connected to the Arduino UNO Q board using a USB Hub.
Fundamental ideas on AI model output tensor, LiteRT runtime, MQTT and Docker are discussed in 
Doulos course on [Edge AI for Embedded Developers](https://www.doulos.com/training/ai-and-deep-learning/deep-learning/essential-edge-ai-for-embedded-developers/)

Step 1:
Clone the repository on Arduino UNO Q

```bash
$git clone https://github.com/Rahuldee18/UNO-Q-Docker-Edge-AI
Cloning into 'UNO-Q-Docker-Edge-AI'...
```
Cloning will create a new folder called UNO-Q-Docker-Edge-AI

Now, copy the contents of this downloaded folder to a folder called camera-project

```bash
$arduino@Rahul-Q:~/UNO-Q-Docker-Edge-AI$ cp -r /home/arduino/UNO-Q-Docker-Edge-AI/ /home/arduino/camera-project
```


Step 2:
Create Docker image using build. Docker is preinstalled on the Arduino UNO Q. 
- The image is built locally on Arduino UNO Q and takes about 2 minutes.
- Change to the camera-project directory and issue the build command.

```bash
$sudo docker build -t camera-project 
```

Step 3:
Check if the image is successfully created

```bash
$docker images

REPOSITORY                                    TAG       IMAGE ID       CREATED        SIZE
camera-project                                latest    1e6593dff2ed   2 hours ago    724MB
```

Step 4:
Run the Docker container created in step 3 in the background.  Camera is connected on the /dev/video0

```bash
$docker run --privileged -v /dev/video0:/dev/video0 camera-project &
```

Step 5:
Observe the output from the application. 
- The output is a MQTT payload consisting of object label, confidence score and bounding box coordinates.
- Sample output with camera pointing at a person. 

```bash

[3] 5227
arduino@Rahul-Q:~/camera-project$ INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Payload being sent:  "Label:", person, ",Score:", 0.6328125, ",Bounding box coodinates:" , [0.04125446 0.15246686 0.9867869  0.8984213 ]
Inference published to MQTT topic
```

Step 6:

- Change the MQTT topic name in the application code (object-detection.py) and subscribe to the topic on another computer to view the output from the object detection model.
- Also, try updating the code to give continuous inference.  Right now, it is setup to provide only one inference at start. 

Step 7: Experiment with different LiteRT object detection models such as the [EfficientDet](https://www.kaggle.com/models/tensorflow/efficientdet/tfLite).  The output tensor of EfficientDet is setup the same way as the object detection model used in thie example. 

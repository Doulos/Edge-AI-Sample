Create a Docker image on Raspberry Pi that runs an object detection model using LiteRT.
The hardware setup consists of a Webcam connected to the Raspberry Pi board using a USB Hub.
Fundamental ideas on AI model output tensor, LiteRT runtime, MQTT and Docker are discussed in 
Doulos course on [Edge AI for Embedded Developers](https://www.doulos.com/training/ai-and-deep-learning/deep-learning/essential-edge-ai-for-embedded-developers/)

**Step 1:**
Clone the repository on Arduino UNO Q

```bash
$git clone https://github.com/Rahuldee18//RPi-YOLOv10

```
Cloning will create a new folder called RPi-YOLOv10.

Change directory to this newly created folder. 

**Step 2:**

Install Docker on Raspberry Pi using this command

$ curl -sSL https://get.docker.com | sh

Add your user to the `docker` group

$ sudo usermod -aG docker $USER

**Step 3:**

Create Docker image using build.

- The image is built locally on Raspberry Pi -3B and takes about 3-4 minutes.
- Change to the camera-project directory and issue the build command.

```bash
$sudo docker build -t yolov10-mqtt .
```

**Step 4:**

 Check if the image is successfully created

```bash
$docker images

REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
yolov10-mqtt     latest    c73cb468959d   7 minutes ago    405MB

```

**Step 5:**
Run the Docker container created in step 3 in the background.  Camera is connected on the /dev/video0

```bash
$sudo docker run --privileged -v /dev/video0:/dev/video0 yolov10-mqtt &
```

**Step 6:**
Observe the output from the application. 

- The output is a MQTT payload consisting of object label, confidence score and bounding box coordinates.
- Sample outputs with camera pointing at a person. 

```bash
[3] 37984
pi@raspberrypi:~/RPi-YOLOv10$ INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Payload being sent:  "Label:", 0.0, person, ",Score:", 0.8530769348144531 ", Image coordinates:", (np.float32(0.19415493), np.float32(0.6156122), np.float32(0.62505543), np.float32(0.9993543))
Inference published to MQTT topic
Payload being sent:  "Label:", 0.0, person, ",Score:", 0.8997121453285217 ", Image coordinates:", (np.float32(0.20572251), np.float32(0.47419235), np.float32(0.63347673), np.float32(0.99792933))
Inference published to MQTT topic
MQTT client disconnected and loop stopped.


```

**Step 7:**

- Change the MQTT topic name in the application code (object-detection-yolov10.py) and subscribe to the topic on another computer to view the output from the object detection model.
- Also, try updating the code to give continuous inference.  Right now, it is setup to provide only one inference at start. 


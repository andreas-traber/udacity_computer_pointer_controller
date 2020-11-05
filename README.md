# Computer Pointer Controller

This project was was done for the course Intel® Edge AI for IoT Developers Nanodegree on Udacity. The goal was do make a prototype for controlling mouse movement with eye movement from a video stream.

## Project Set Up and Installation
For this project [Openvino](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) 2020.4 with following models was used:

* [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

After installing Openvino(refer to the manual in the link) you can download the models using following commands:

```bash
{PATH_TO_OPENVINO}/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o models
{PATH_TO_OPENVINO}/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o models
{PATH_TO_OPENVINO}/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o models
{PATH_TO_OPENVINO}/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o models
```

(Optional) You can use a Python Virtual Environment to install the requirements
```bash
pip3 install virtualenv
source venv/bin/activate 
pip3 install -r requirements.txt
```

## Demo
Source the Openvino Environment
```bash
source {PATH_TO_OPENVINO}/scripts/setupvars/setupvars.sh
```

(Optional)  Source the python virtual environment
```bash
source venv/bin/activate
```

Run the program
```bash
python3 src/__main__.py
```

## Documentation
You can see all the arguments with --help
```
python3 src/__main__.py --help
usage: __main__.py [-h] [--model_face_detection MODEL_FACE_DETECTION] [--model_head_pose MODEL_HEAD_POSE] [--model_gaze_estimation MODEL_GAZE_ESTIMATION] [--model_landmarks MODEL_LANDMARKS] [--input INPUT] [--extensions EXTENSIONS]
                   [--device DEVICE] [--mouse_speed MOUSE_SPEED] [--mouse_precision MOUSE_PRECISION] [--log_level LOG_LEVEL] [--save_error_frame] [--video_out] [--show_image_steps] [--moving_mouse]

optional arguments:
  -h, --help            show this help message and exit
  --model_face_detection MODEL_FACE_DETECTION, -m MODEL_FACE_DETECTION
                        Path to an xml file with a trained face detection model.
  --model_head_pose MODEL_HEAD_POSE, -mhp MODEL_HEAD_POSE
                        Path to an xml file with a trained model.
  --model_gaze_estimation MODEL_GAZE_ESTIMATION, -mge MODEL_GAZE_ESTIMATION
                        Path to an xml file with a trained Gaze Estimation model.
  --model_landmarks MODEL_LANDMARKS, -ml MODEL_LANDMARKS
                        Path to an xml file with a trained Landmark model.
  --input INPUT, -i INPUT
                        Path to image or video file, 'CAM' for webcam-input
  --extensions EXTENSIONS, -e EXTENSIONS
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with thekernels impl.
  --device DEVICE, -d DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)
  --mouse_speed MOUSE_SPEED, -ms MOUSE_SPEED
                        Specify how fast the mouse moves
  --mouse_precision MOUSE_PRECISION, -mp MOUSE_PRECISION
                        Specify how precise the mouse moves
  --log_level LOG_LEVEL, -l LOG_LEVEL
                        Set the log Level
  --save_error_frame, -sef
                        Save a frame, that can not be processed
  --video_out, -vo      show the video(-stream) with cv2.show()
  --show_image_steps, -s
                        Show an image of every step with cv2.show()
  --moving_mouse, -mm   Turn the mouse movement on
```

The directory structure is as following

``` 
.
├── bin                                            - contains video or picture files
├── LICENSE                                        - LICENSE file
├── models                                         - contains OpenVino Models
├── README.md                                      - README file
├── requirements.txt                               - required python libarys 
├── src                                            - contains source code
│   ├── face_detection.py                          - implementation of face detection
│   ├── facial_landmarks_detection.py              - implementation of facial landmark detection
│   ├── gaze_estimation.py                         - implementation of gaze estimation
│   ├── head_pose_estimation.py                    - implementation of head pose estimation
│   ├── __init__.py                                - INIT file
│   ├── __main__.py                                - main application
│   ├── model.py                                   - base clase for models
│   └── mouse_controller.py                        - implementation of mouse control
└── venv                                           - contains the virtual environment

```

## Benchmarks
### AMD FX(tm)-8350 Eight-Core Processor
This benchmark was made on a Desktop PC with an AMD FX(tm)-8350 Eight-Core Processor.
```
python3 src/__main__.py --stats -i bin/demo.mp4
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|420.8|83.6|79.3|90.8
Head Pose|132.7|3.8|3.2|7.7
Landmarks|129.3|1.2|0.9|5.9
Gaze Estimation|160.6|5.4|4.0|14.1

```
python3 src/__main__.py --stats -i bin/demo.mp4 --model_landmarks models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml --model_gaze_estimation models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml --model_head_pose models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml --model_face_detection models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|681.1|83.2|79.1|97.5
Head Pose|263.6|3.8|3.2|6.2
Landmarks|197.1|1.2|0.9|1.6
Gaze Estimation|269.0|5.3|4.0|10.3

```
python3 src/__main__.py --stats -i bin/demo.mp4  --model_landmarks models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml --model_gaze_estimation models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml --model_head_pose models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml --model_face_detection models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|1552.2|39.2|35.7|53.2
Head Pose|354.6|2.7|2.3|8.6
Landmarks|181.1|0.9|0.7|1.6
Gaze Estimation|419.8|2.6|2.1|4.3

### Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz
This benchmark was made on a Desktop PC with  Intel(R) Core(TM) i5-7500 CPU processor.
```
python3 src/__main__.py --stats -i bin/demo.mp4
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|260.6|18.7|14.6|69.0
Head Pose|85.2|1.4|1.1|6.3
Landmarks|74.1|0.5|0.3|2.1
Gaze Estimation|91.1|1.6|1.2|6.4

```
python3 src/__main__.py --stats -i bin/demo.mp4 --model_landmarks models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml --model_gaze_estimation models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml --model_head_pose models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml --model_face_detection models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|277.5|18.9|14.6|59.3
Head Pose|131.7|1.4|1.1|5.7
Landmarks|78.5|0.5|0.3|3.9
Gaze Estimation|135.5|1.6|1.2|5.7

```
python3 src/__main__.py --stats -i bin/demo.mp4  --model_landmarks models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml --model_gaze_estimation models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml --model_head_pose models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml --model_face_detection models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml
```
Model Name|Load Time in ms|Avg. Inference Time in ms|Min. Inference Time in ms|Max. Inference Time in ms
---|---|---|---|---
Face Detection|585.4|13.6|10.7|38.4
Head Pose|237.3|1.0|0.8|7.2
Landmarks|122.2|0.5|0.3|1.6
Gaze Estimation|257.3|1.0|0.7|6.6




## Results
The benchmark clearly shows, that the Intel-Processor is much better suited for this application.
Face Detection is the slowest of the models, so if we wanted to optimize a model, this would be our choice.
Using different precisions doesn't make much difference between FP32 and FP16. FP16 even seems worse, since the load time is higher.
There is a performance boost, when using FP16-INT8 on the interference time, but worse performance for the load time. For this application load time doesn't matter that much, since it should be loaded once and the inference running on a lot of frames.
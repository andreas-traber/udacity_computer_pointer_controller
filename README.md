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

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

import sys

from face_detection import ModelFaceDetection
from head_pose_estimation import ModelHeadPoseEstimation
from facial_landmarks_detection import ModelFacialLandmarksDetection
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController

import logging as log
import cv2
import time

from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model_face_detection", "-m", required=False, type=str,
                        help="Path to an xml file with a trained face detection model.",
                        default='models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001'
                                '.xml')
    parser.add_argument("--model_head_pose", "-mhp", required=False, type=str,
                        help="Path to an xml file with a trained model.",
                        default='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml')
    parser.add_argument("--model_gaze_estimation", "-mge", required=False, type=str,
                        help="Path to an xml file with a trained Gaze Estimation model.",
                        default='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml')
    parser.add_argument("--model_landmarks", "-ml", required=False, type=str,
                        help="Path to an xml file with a trained Landmark model.",
                        default='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
                                '.xml')
    parser.add_argument("--input", "-i", required=False, type=str,
                        help="Path to image or video file, 'CAM' for webcam-input",
                        default='bin/single_frame.png')
    parser.add_argument("--extensions", "-e", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("--device", "-d", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("--mouse_speed", "-ms", type=str, default="fast",
                        help="Specify how fast the mouse moves")
    parser.add_argument("--mouse_precision", "-mp", type=str, default="medium",
                        help="Specify how precise the mouse moves")
    parser.add_argument("--log_level", "-l", type=str, default="INFO",
                        help="Set the log Level")
    parser.add_argument("--save_error_frame", "-sef", default=False, action='store_true',
                        help="Save a frame, that can not be processed")
    parser.add_argument("--video_out", "-vo", default=False, action='store_true',
                        help="show the video(-stream) with cv2.show()")
    parser.add_argument("--show_image_steps", "-s", default=False, action='store_true',
                        help="Show an image of every step with cv2.show()")
    parser.add_argument("--moving_mouse", "-mm", default=False, action='store_true',
                        help="Turn the mouse movement on")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.input == 'CAM':
        args.input=0
    log.basicConfig(format='%(levelname)s: %(message)s', level=args.log_level)
    mouse = MouseController(args.mouse_precision, args.mouse_speed)
    face_det = ModelFaceDetection(model_name=args.model_face_detection, device=args.device, extensions=args.extensions)
    head_pose = ModelHeadPoseEstimation(args.model_head_pose, args.device, args.extensions)
    gaze_est = ModelGazeEstimation(args.model_gaze_estimation, args.device, args.extensions)
    landmarks = ModelFacialLandmarksDetection(args.model_landmarks, args.device, args.extensions)
    cap = cv2.VideoCapture()
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    while cap.isOpened():
        try:
            flag, frame = cap.read()
            if not flag:
                break
            if args.video_out:
                cv2.imshow('video', frame)
            face_pred = face_det.predict(frame)
            if not face_pred:
                log.warning('Face not Found')
                continue
            log.debug('Face Prediction at %s' % face_pred)
            if args.show_image_steps:
                out_frame = face_det.draw_prediction(frame, face_pred, width, height)
                cv2.imshow('Head Boundaries', out_frame)
                _ = cv2.waitKey()
            img_head = face_det.preprocess_output(frame, face_pred, width, height)
            if args.show_image_steps:
                cv2.imshow('Head', img_head)
                _ = cv2.waitKey()
            landmark_pred = landmarks.predict(img_head)
            log.debug('Landmark Prediction at %s' % landmark_pred)
            left_eye_image, right_eye_image = landmarks.preprocess_output(img_head, landmark_pred,
                                                                          img_head.shape[1], img_head.shape[0])
            if not left_eye_image.any():
                log.warning('Eye not Found')
                continue

            if args.show_image_steps:
                out_frame = landmarks.draw_prediction(img_head, landmark_pred, img_head.shape[1], img_head.shape[0])
                cv2.imshow('Landmarks Boundaries', out_frame)
                _ = cv2.waitKey()
                cv2.imshow('left eye', left_eye_image)
                _ = cv2.waitKey()
                cv2.imshow('right eye', right_eye_image)
                _ = cv2.waitKey()
            head_pose_angles = head_pose.predict(img_head)
            log.debug('Head Pose Prediction at %s' % head_pose_angles)
            gaze = gaze_est.predict(left_eye_image, right_eye_image, head_pose_angles)
            log.debug('Gaze Prediction at %s' % gaze)
            if args.show_image_steps:
                out_frame = gaze_est.draw_prediction(img_head, gaze, img_head.shape[1], img_head.shape[0])
                cv2.imshow('Gaze', out_frame)
                _ = cv2.waitKey()
            mouse_movement = gaze_est.preprocess_output(gaze)

            key = cv2.waitKey(1)

            if key in {ord("q"), ord("Q"), 27}:  # ESC key
                log.info('%s pressed, stopping program' % key)
                break
            if args.moving_mouse:
                mouse.move(mouse_movement[0], mouse_movement[1])
        except:
            e = sys.exc_info()
            log.error(e)
            if args.save_error_frame:
                out_frame = 'q%s.png' % time.strftime("%Y%m%d%H%M%S")
                log.info('write file: %s' % out_frame)
                cv2.imwrite(out_frame, frame)
            raise

if __name__ == '__main__':
    main()

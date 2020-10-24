from src.face_detection import ModelFaceDetection
from src.head_pose_estimation import ModelHeadPoseEstimation
from src.facial_landmarks_detection import ModelFacialLandmarksDetection
from src.gaze_estimation import ModelGazeEstimation
from src.mouse_controller import MouseController


import cv2

from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model_face_detection", "-m", required=False, type=str,
                        help="Path to an xml file with a trained face detection model.",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller'
                                '/models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001'
                                '.xml')
    parser.add_argument("--model_head_pose", "-mhp", required=False, type=str,
                        help="Path to an xml file with a trained model.",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller/models/intel'
                                '/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml')
    parser.add_argument("--model_gaze_estimation", "-mge", required=False, type=str,
                        help="Path to an xml file with a trained Gaze Estimation model.",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller/models/intel'
                                '/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml')
    parser.add_argument("--model_landmarks", "-ml", required=False, type=str,
                        help="Path to an xml file with a trained Landmark model.",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller/models/intel'
                                '/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml')
    parser.add_argument("--input", "-i", required=False, type=str,
                        help="Path to image or video file",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller/bin/vlcsnap'
                                '-2020-10-20-12h42m10s763.png')
    parser.add_argument("--extensions", "-l", required=False, type=str,
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
    parser.add_argument("--video_out", "-vo", dest='video_out', default=False, action='store_true',
                        help="Show an image of every step with cv2.show()")
    parser.add_argument("--show_image_steps", "-s", dest='show_image_steps', default=False, action='store_true',
                        help="Show an image of every step with cv2.show()")
    return parser


def main():
    args = build_argparser().parse_args()
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
        flag, frame = cap.read()
        if not flag:
            break
        if args.video_out:
            cv2.imshow('video', frame)
        face_pred = face_det.predict(frame)
        if args.show_image_steps:
            out_frame = face_det.draw_prediction(frame, face_pred, width, height)
            cv2.imshow('Head Boundaries', out_frame)
            _ = cv2.waitKey()
        img_head = face_det.preprocess_output(frame, face_pred, width, height)
        if args.show_image_steps:
            cv2.imshow('Head', img_head)
            _ = cv2.waitKey()
        landmark_pred = landmarks.predict(img_head)
        left_eye_image, right_eye_image = landmarks.preprocess_output(img_head, landmark_pred,
                                                                  img_head.shape[1], img_head.shape[0])
        if args.show_image_steps:
            out_frame = landmarks.draw_prediction(img_head, landmark_pred, img_head.shape[1], img_head.shape[0])
            cv2.imshow('Landmarks Boundaries', out_frame)
            _ = cv2.waitKey()
            cv2.imshow('left eye', left_eye_image)
            _ = cv2.waitKey()
            cv2.imshow('right eye', right_eye_image)
            _ = cv2.waitKey()
        head_pose_angles = head_pose.predict(img_head)
        gaze = gaze_est.predict(left_eye_image, right_eye_image, head_pose_angles)
        if args.show_image_steps:
            out_frame = gaze_est.draw_prediction(img_head, gaze, img_head.shape[1], img_head.shape[0])
            cv2.imshow('Gaze', out_frame)
            _ = cv2.waitKey()

        key = cv2.waitKey(1)

        if key in {ord("q"), ord("Q"), 27}: # ESC key
            break
        gaze = gaze_est.preprocess_output(gaze)
        print(gaze)
        mouse.move(gaze[0], gaze[1])


if __name__ == '__main__':
    main()

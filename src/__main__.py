from src.face_detection import ModelFaceDetection
from src.head_pose_estimation import ModelHeadPoseEstimation
from src.facial_landmarks_detection import ModelFacialLandmarksDetection
from src.gaze_estimation import ModelGazeEstimation


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
    return parser


def main():
    args = build_argparser().parse_args()
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
        face_pred = face_det.predict(frame)
        #out_frame = face_det.draw_bbox(frame, face_pred, width, height)
        #cv2.imshow('test', out_frame)
        #key = cv2.waitKey()
        img_head = face_det.preprocess_output(frame, face_pred, width, height)
        #cv2.imshow('Head', img_head)
        #key = cv2.waitKey()
        landmark_pred = landmarks.predict(img_head)
        #out_frame = landmarks.draw_bbox(img_head, landmark_pred, img_head.shape[1], img_head.shape[0])
        #cv2.imshow('landmarks', out_frame)
        #key = cv2.waitKey()


if __name__ == '__main__':
    main()

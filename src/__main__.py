from src.face_detection import ModelFaceDetection

import cv2

from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=False, type=str,
                        help="Path to an xml file with a trained model.",
                        default='/home/andi/python_projects/udacity/udacity_computer_pointer_controller'
                                '/models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001'
                                '.xml')
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
    face_det = ModelFaceDetection(model_name=args.model, device=args.device, extensions=args.extensions)
    cap = cv2.VideoCapture()
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        face_pred = face_det.predict(frame)
        out_frame = face_det.draw_bbox(frame, face_pred, width, height)
        cv2.imshow('test', out_frame)
        key = cv2.waitKey()
        out_frame = face_det.preprocess_output(frame, face_pred, width, height)
        cv2.imshow('test', out_frame)
        key = cv2.waitKey()

if __name__ == '__main__':
    main()

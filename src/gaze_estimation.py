import copy

from model import Model
import cv2

class ModelGazeEstimation(Model):
    """
    Class for the Gaze Estimation Model.
    """
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        super().__init__(model_name, device, extensions, threshold)
        self.net_input_shape = self.net.input_info['left_eye_image'].input_data.shape

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        """
        This method is meant for running predictions on the input image.
        """
        preprocessed_left_eye = self.preprocess_input(left_eye_image)
        preprocessed_right_eye = self.preprocess_input(right_eye_image)
        result = self.exec_network.infer(inputs={'left_eye_image': preprocessed_left_eye,
                                                 'right_eye_image': preprocessed_right_eye,
                                                 'head_pose_angles': head_pose_angles})
        return result

    def get_coordinates(self, outputs, width, height):
        out_flat = outputs['gaze_vector'].flatten()
        pt1 = (int(width/2), int(height/2))
        pt2 = (int((0.5+out_flat[0]) * width),
               int((0.5-out_flat[1]) * height))
        # pt2_alt = (int(width/2+out_flat[0] * 240),
        #       int(height/2-out_flat[1] * 240))
        return pt1, pt2

    def draw_prediction(self, image, outputs, width, height):
        """
        Draws the result of the prediction on the original image
        """
        ret_image = copy.copy(image)
        pt1, pt2 = self.get_coordinates(outputs, width, height)
        cv2.arrowedLine( ret_image, pt1, pt2, [0, 0, 255], 6)
        return ret_image

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        return outputs['gaze_vector'].flatten()

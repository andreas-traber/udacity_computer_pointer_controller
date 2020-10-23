from src.model import Model

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

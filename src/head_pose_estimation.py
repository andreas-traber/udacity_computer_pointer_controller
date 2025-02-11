import time

from model import Model


class ModelHeadPoseEstimation(Model):
    """
    Class for the Face Detection Model.
    """

    def predict(self, image):
        """
        This method is meant for running predictions on the input image.
        """
        preprocessed_image = self.preprocess_input(image)
        start = time.time()
        result = self.exec_network.infer(inputs={self.input_name: preprocessed_image})
        self.infer_times.append(time.time() - start)
        return [[result['angle_p_fc'][0][0],result['angle_r_fc'][0][0],result['angle_y_fc'][0][0]]]

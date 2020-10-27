import time

from model import Model


class ModelFaceDetection(Model):
    """
    Class for the Face Detection Model.
    """

    def predict(self, image):
        start = time.time()
        ret = super().predict(image)
        self.infer_times[-1] = time.time() - start
        for x in ret[0][0]:
            if x[2] > self.threshold:
                return [x]

    def get_coordinates(self, outputs, width, height):
        bbox = []
        for res in outputs:
            bbox.append([int(res[3] * width), int(res[4] * height), int(res[5] * width), int(res[6] * height)])
        return bbox

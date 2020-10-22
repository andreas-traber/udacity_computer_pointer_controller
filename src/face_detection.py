from src.model import Model


class ModelFaceDetection(Model):
    """
    Class for the Face Detection Model.
    """

    def predict(self, image):
        ret = super().predict(image)
        return [x for x in ret[0][0] if x[2] > self.threshold]

    def get_coordinates(self, outputs, width, height):
        bbox = []
        for res in outputs:
            bbox.append([int(res[3] * width), int(res[4] * height), int(res[5] * width), int(res[6] * height)])
        return bbox

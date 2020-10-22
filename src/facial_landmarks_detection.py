from src.model import Model

class ModelFacialLandmarksDetection(Model):
    """
    Class for the Facial Landmarks Detection Model.
    """

    def get_coordinates(self, outputs, width, height):
        bbox = []
        out_flat = outputs.flatten()
        for i in [0,2]:
            bbox.append([int(out_flat[i] * width), int(out_flat[i+1] * height),
                         int(out_flat[i] * width), int(out_flat[i+1] * height)])
        return bbox


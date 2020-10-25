from model import Model

class ModelFacialLandmarksDetection(Model):
    """
    Class for the Facial Landmarks Detection Model.
    """


    def get_coordinates(self, outputs, width, height):
        bbox = []
        out_flat = outputs.flatten()
        # get only eyes
        for i in [0, 2]:
            bbox.append([int((out_flat[i]-0.12) * width), int((out_flat[i+1]-0.06) * height),
                         int((out_flat[i]+0.12) * width), int((out_flat[i+1]+0.06) * height)])
        return bbox

    def preprocess_output(self, image, outputs, width, height):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        ret = []
        bbox = self.get_coordinates(outputs, width, height)
        for rect in bbox:
            ret.append(image[rect[1]:rect[3], rect[0]:rect[2]])
        return ret


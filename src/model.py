"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import os
from abc import abstractmethod

from openvino.inference_engine import IECore
import cv2
import copy


class Model:
    """
    Class for Model.
    """

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        """
        TODO: Use this to set your instance variables.
        """
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        # Load the model
        model_bin = os.path.splitext(self.model_name)[0] + ".bin"

        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model_name, weights=model_bin)
        self.exec_network = self.ie.load_network(network=self.net, device_name=self.device)
        self.output_blobs = next(iter(self.net.outputs))
        self.input_name = next(iter(self.net.input_info.keys()))

        # Check for supported layers
        supported_layers = self.ie.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")

        # Add any necessary extensions
        if self.extensions:
            self.ie.add_extension(self.extensions, "CPU")

        self.net_input_shape = self.net.input_info[self.input_name].input_data.shape

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        image_resized = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]))
        image_resized = image_resized.transpose((2, 0, 1))
        image_resized = image_resized.reshape(1, *image_resized.shape)
        return image_resized

    def predict(self, image):
        """
        This method is meant for running predictions on the input image.
        """
        preprocessed_image = self.preprocess_input(image)
        return self.exec_network.infer(inputs={self.input_name: preprocessed_image})[self.output_blobs]

    def preprocess_output(self, image, outputs, width, height):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        ret_image = copy.copy(image)
        bbox = self.get_coordinates(outputs, width, height)
        for rect in bbox:
            return ret_image[rect[1]:rect[3], rect[0]:rect[2]]

    @abstractmethod
    def get_coordinates(self, outputs, width, height):
        None

    def draw_bbox(self, image, outputs, width, height):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        ret_image = copy.copy(image)
        bbox = self.get_coordinates(outputs, width, height)
        print(bbox)
        for rect in bbox:
            cv2.rectangle(ret_image, (rect[0], rect[1]), (rect[2], rect[3]), [0, 0, 255], 6)
        return ret_image

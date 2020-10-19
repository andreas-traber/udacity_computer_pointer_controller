"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import os
from openvino.inference_engine import IECore
import cv2


class Model:
    """
    Class for Model.
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
        TODO: Use this to set your instance variables.
        """
        self.model_name = model_name
        self.device = device
        self.extensions = extensions

    def load_model(self):
        """
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """

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
        if self.cpu_extension:
            self.ie.add_extension(self.cpu_extension, "CPU")

        self.net_input_shape = self.net.input_info[self.input_name].input_data.shape

    def predict(self, image):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        return self.exec_network.infer(request_id=0, inputs={self.input_name: image})

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        image_resized = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]))
        image_resized = image_resized.transpose((2, 0, 1))
        image_resized = image_resized.reshape(1, * image_resized.shape)
        return image_resized

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        raise NotImplementedError

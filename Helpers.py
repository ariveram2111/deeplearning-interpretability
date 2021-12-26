import numpy as np
import tensorflow as tf
import cv2

from matplotlib import pyplot as plt
from copy import copy

class ImageVisualizer:

    @staticmethod
    def visualize_image(image, cmap='gray', title='', show=True, save=False, path=None):
        # Assert that the image is a numpy array
        assert type(image) == np.ndarray

        # Assert that if save is True then path is a string
        assert (not save) or (save and type(path) == str)

        # Disable interactive mode
        plt.ioff()

        # Create the image
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')

        # Save the image
        if save:
            plt.savefig(path)

        # Show the image
        if show:
            plt.show()

        # Close all opened figures/images
        plt.close('all')

    @staticmethod
    def visualize_multiple_images(images, grid_rows, grid_cols, classes=None, title='', figsize=(20,20), show=True, save=False, path=None):
        # Assert that the image is a numpy array
        assert type(images) == np.ndarray
        assert classes is None or (type(classes) == np.ndarray and classes.shape[0] == images.shape[0])

        # Assert that if save is True then path is a string
        assert (not save) or (save and type(path) == str)

        # Disable interactive mode
        plt.ioff()

        # Define the figure size
        fig = plt.figure(figsize=figsize)
        count = 0

        # Each row represents a filter. Each column has each channel of the filter
        for i in range(0, grid_rows):
            for j in range(0, grid_cols):
                fig.add_subplot(grid_rows, grid_cols, count+1)
                plt.imshow(images[count])
                plt.axis('off')

                if classes is not None:
                    plt.title('Class {}'.format(classes[count]))

                count += 1

        # Add a suptitle to the figure
        fig.suptitle(title)
        
        # Save the image
        if save:
            plt.savefig(path)

        # Show the image
        if show:
            plt.show()

        # Close all opened figures/images
        plt.close('all')

    @staticmethod
    def visualize_grad_CAM(map, image=None, alpha=0.4, title='', show=True, save=False, path=None):
        # Assert that the map is a numpy array
        assert type(map) == np.ndarray

        # Check that the image is None or a numpy array and shapes
        assert image is None or (type(image) == np.ndarray and image.shape[0:2] == map.shape[0:2])

        # Assert that if save is True then path is a string
        assert (not save) or (save and type(path) == str)

        # Disable interactive mode
        plt.ioff()

        # Apply color map
        jet_heatmap = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose the heatmap on original image
        if image is not None:
            heatmap = cv2.addWeighted(image, 1-alpha, jet_heatmap, alpha, 0)
        else:
            heatmap = jet_heatmap

        # Rescale heatmap to a range 0-255
        heatmap = np.minimum(heatmap, 255.0).astype(np.uint8)

        # Visualize the heatmap
        plt.imshow(heatmap)
        plt.title(title)
        
        # Save the image
        if save:
            plt.savefig(path)

        # Show the image
        if show:
            plt.show()

        # Close all opened figures/images
        plt.close('all')

        return heatmap

    @staticmethod
    def visualize_important_segments_heatmap(heatmap, image, threshold_r=180, title='', show=True, save=False, path=None):
        # Assert that the heatmap is a numpy array
        assert type(heatmap) == np.ndarray

        # Check that the image is None or a numpy array and shapes
        assert image is None or (type(image) == np.ndarray and image.shape[0:2] == heatmap.shape[0:2])

        # Assert that if save is True then path is a string
        assert (not save) or (save and type(path) == str)

        # Disable interactive mode
        plt.ioff()

        # Compute the mask
        mask = np.where((heatmap[:, :, 0] > threshold_r), 1.0, 0.0)
        mask = mask.astype(bool)

        # Do a copy of the image
        masked_img = image.copy()

        # All values that does not match the mask, put them 0 value
        masked_img[~mask] = 0

        # Visualize the masked_img
        plt.imshow(masked_img)
        plt.title(title)

        # Save the image
        if save:
            plt.savefig(path)

        # Show the image
        if show:
            plt.show()

        # Close all opened figures/images
        plt.close('all')

class ImageProcessor:

    @staticmethod
    def min_max_normalization(input_images, min=None, max=None):
        # Check that images are a numpy array and has at least 3 dimensions
        assert type(input_images) == np.ndarray and input_images.ndim >= 3

        # If min or/and max is specified, then use them
        if min is not None:
            n_min = min
        else:
            n_min = np.min(input_images)

        if max is not None:
            n_max = max
        else:
            n_max = np.max(input_images)

        # Apply the normalization
        norm_images = (input_images - n_min) / (n_max - n_min)

        return norm_images

    @staticmethod
    def standardization(input_images, mean=None, stddev=None):
        # Check that images are a numpy array and has at least 3 dimensions
        assert type(input_images) == np.ndarray and input_images.ndim >= 3

        # If mean or/and stddev is specified, then use them
        if mean is not None:
            n_mean = mean
        else:
            n_mean = np.mean(input_images)

        if stddev is not None:
            n_stddev = stddev
        else:
            n_stddev = np.std(input_images)

        # Apply standardization
        standardized_images = (input_images - n_mean) / n_stddev

        return standardized_images

    @staticmethod
    def clipping(input_images, min=0, max=1):
        # Check that images are a numpy array and has at least 3 dimensions
        assert type(input_images) == np.ndarray and input_images.ndim >= 3

        return np.clip(input_images, min, max)

    @staticmethod
    def adjust_hsv(image, saturation_exp = 2.0, value_exp = 0.5):
        # Do less emphasis on lower saturation
        # Note: the image must be normalized and must have 3 channels
        assert type(image) == np.ndarray and image.ndim == 3 

        # From RGB to HSV
        hsv = tf.image.rgb_to_hsv(image)
        hue, saturation, value = tf.split(hsv, 3, axis=2)

        # Change saturation and value
        saturation = tf.math.pow(saturation, saturation_exp)
        value = tf.math.pow(value, value_exp)

        # HSV to RGB
        hsv_new = tf.squeeze(tf.stack([hue, saturation, value], axis=2), axis = 3)
        rgb = tf.image.hsv_to_rgb(hsv_new)
        
        return rgb.numpy()

    @staticmethod
    def upscaling(image, upscaling_factor=1.1):
        # Check that the image has a valid type
        assert type(image) == np.ndarray and image.ndim >= 2

        if upscaling_factor == 1.0:
            # If upscaling factor is 1, let the image as it is
            upscaled_img = image
        
        else:
            # Upscale image
            sz = np.array(np.shape(image))[0]
            sz_up = (upscaling_factor * sz).astype("int")
            lower = int(np.floor((sz_up - sz) / 2))
            upper = int(np.ceil((sz_up - sz) / 2))

            upscaled_img = cv2.resize(image.astype("float"), dsize=(sz_up, sz_up))
            upscaled_img = upscaled_img[lower:-lower, lower:-lower, :]

        return upscaled_img

    @staticmethod
    def to_integer_conversion(input_images, min=0, max=255):
        # Check that images are a numpy array and has at least 3 dimensions
        assert type(input_images) == np.ndarray and input_images.ndim >= 3

        # Multiply all values by max
        input_images *= max

        # Clip values that are lower and higher than 0 and 255
        image = np.clip(input_images, min, max).astype("uint8")

        return image

class ObjectiveFunction:

    @staticmethod
    def compute_channel_function(input_image, model, filter_index, tf_regularizer_function=None, tf_regularizer_params=None, regularization_factor=0.1):
        # Check that the model is a Keras model
        assert isinstance(model, tf.keras.Model)

        # Check the regularizer function
        assert (tf_regularizer_function is None) or (tf_regularizer_function is not None and isinstance(tf_regularizer_params, dict))

        # Create a batch with 1 image
        image = tf.expand_dims(input_image, axis=0)

        # Compute the activations
        activations = model(image)

        # Get filter activations. Avoid border artifacts by only involving non-border activations
        filter_activations = activations[:, 1:-1, 1:-1, filter_index]

        # Compute the mean
        result = tf.reduce_mean(filter_activations)

        # Add regularization
        if tf_regularizer_function is not None:
            norm = tf_regularizer_function(input_image, **tf_regularizer_params)
            result = (1-regularization_factor) * result + regularization_factor * norm

        return result

    @staticmethod
    def compute_class_score_function(input_image, model, class_index, tf_regularizer_function=None, tf_regularizer_params=None, regularization_factor=0.1):
        # Check that the model is a Keras model
        assert isinstance(model, tf.keras.Model)

        # Check the regularizer function
        assert (tf_regularizer_function is None) or (tf_regularizer_function is not None and isinstance(tf_regularizer_params, dict))

        # Create a batch with 1 image
        image = tf.expand_dims(input_image, axis=0)

        # Delete the activation of the last layer
        model.layers[-1].activation = None

        # Compute the output
        output_logits = model(image)

        # Get the logits of the class
        result = output_logits[0,class_index]

        # Add regularization
        if tf_regularizer_function is not None:
            norm = tf_regularizer_function(input_image, **tf_regularizer_params)
            result = (1-regularization_factor) * result + regularization_factor * norm

        return result

    @staticmethod
    def compute_grad_CAM_conv_function(input_image, model, pred_index):
        # Check that the model is a Keras model
        assert isinstance(model, tf.keras.Model)

        # Create a batch with 1 image
        image = tf.expand_dims(input_image, axis=0)

        # Get outputs of the model
        output_conv_layer, class_probabilities = model(image)

        # Take as class the class with the higher probability or the specified one
        if pred_index is None:
            pred_index = tf.argmax(class_probabilities[0])

        class_channel = class_probabilities[:, pred_index]

        return class_channel, output_conv_layer

class GradientAscent:

    def __init__(self, model):
        self.model = copy(model)

    @staticmethod
    def get_random_initialized_image(image_shape, min=0, max=1, type=float):
        # Depending the type, use an int type or a float type
        if type == int:
            type_n = tf.int64
        elif type == float:
            type_n = tf.float32
        else:
            raise ValueError('Specified type is not valid: it must be int or float')

        # Start from a gray image with some random noise
        img = tf.random.uniform(image_shape, minval=min, maxval=max, dtype=type_n)

        return img.numpy()

    @staticmethod
    def deprocess_image(img, adjust_hsv=True, upscaling_factor=1.1, std=0.15):
        # Check that the image has a valid type
        assert type(img) == np.ndarray and img.ndim >= 2

        image = np.copy(img)

        # Standardize the image
        ImageProcessor.standardization(image)

        # Ensure std is what is specified
        image *= std

        # Clipping values between 0 and 1
        ImageProcessor.clipping(np.array([image]), 0, 1)

        # Adjust HSV, for more info look at the function
        if adjust_hsv:
            image = ImageProcessor.adjust_hsv(image)

        # Upscale the image to ignore borders
        image = ImageProcessor.upscaling(image, upscaling_factor=upscaling_factor)

        # Conversion to int between 0 and 255
        image = ImageProcessor.to_integer_conversion(np.array([image])) [0]

        return image

    @tf.function
    def compute_gradient_input_space(self, img, objective_function, objective_function_params, learning_rate, normalize_grad=True, ascent=True):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = objective_function(img, **objective_function_params)

        # Compute the gradient
        grad = tape.gradient(loss, img)

        # Normalize the gradient (L2)
        if normalize_grad:
            grad = tf.math.l2_normalize(grad)

        # Update the image in the direction of the gradient (or in the opposite direction of the gradient)
        if ascent:
            img += learning_rate * grad
        else:
            img -= learning_rate * grad

        return loss, img

    def get_filters_layer(self, initial_image, layer, filter_indexes, iterations=30, learning_rate=10, image_norm_function=None, image_norm_params=None,
        tf_regularizer_function=None, tf_regularizer_params=None, regularization_factor=0.1):
        # Check that it's a numpy image
        assert isinstance(initial_image, np.ndarray) and initial_image.ndim >= 2
        assert (image_norm_function is None and image_norm_params is None) or (image_norm_function is not None and isinstance(image_norm_params, dict))

        # Check the regularizer function
        assert (tf_regularizer_function is None) or (tf_regularizer_function is not None and isinstance(tf_regularizer_params, dict))

        # Get the layer
        if type(layer) == int:
            model_layer = self.model.layers[layer]
        else:
            model_layer = self.model.get_layer(layer)

        # Redefine the model to set as output layer the given layer
        n_model = tf.keras.Model(inputs=self.model.inputs, outputs=model_layer.output)

        # Compute gradient ascent
        filters_losses = []
        filters_images = []

        for filter_index in filter_indexes:
            losses = []
            images = []
            img = np.copy(initial_image)
            for i in range(iterations):
                loss, img = self.compute_gradient_input_space(img, ObjectiveFunction.compute_channel_function, {'model':n_model, 
                    'filter_index':filter_index, 'tf_regularizer_function':tf_regularizer_function, 
                    'tf_regularizer_params':tf_regularizer_params, 'regularization_factor':regularization_factor}, learning_rate)

                # Normalize output image
                if image_norm_function is not None and image_norm_params is not None:
                    img = image_norm_function(img.numpy(), **image_norm_params)

                losses.append(loss.numpy())
                images.append(img.numpy())

            filters_losses.append(losses)
            filters_images.append(images)

        return np.array(filters_losses), np.array(filters_images)

    def get_class(self, initial_image, class_indexes, iterations=30, learning_rate=10, image_norm_function=None, image_norm_params=None, logits=True,
        tf_regularizer_function=None, tf_regularizer_params=None, regularization_factor=0.1):
        # Check that it's a numpy image
        assert isinstance(initial_image, np.ndarray) and initial_image.ndim >= 2
        assert (image_norm_function is None and image_norm_params is None) or (image_norm_function is not None and isinstance(image_norm_params, dict))

        # Check the regularizer function
        assert (tf_regularizer_function is None) or (tf_regularizer_function is not None and isinstance(tf_regularizer_params, dict))

        # Compute gradient ascent
        classes_losses = []
        classes_images = []

        for class_index in class_indexes:
            losses = []
            images = []
            img = np.copy(initial_image)
            for i in range(iterations):
                # Use logits or probabilities
                if logits:
                    objective_function = ObjectiveFunction.compute_class_score_function
                else:
                    objective_function = ObjectiveFunction.compute_class_probability_function

                loss, img = self.compute_gradient_input_space(img, objective_function, 
                    {'model':self.model, 'class_index':class_index, 'tf_regularizer_function':tf_regularizer_function, 
                    'tf_regularizer_params':tf_regularizer_params, 'regularization_factor':regularization_factor}, learning_rate)

                # Normalize output image
                if image_norm_function is not None and image_norm_params is not None:
                    img = image_norm_function(img.numpy(), **image_norm_params)

                losses.append(loss.numpy())
                images.append(img.numpy())

            classes_losses.append(losses)
            classes_images.append(images)

        return np.array(classes_losses), np.array(classes_images)


class GradCAM:

    def __init__(self, model):
        self.model = copy(model)

    @tf.function
    def __compute_gradient(self, img, objective_function, objective_function_params):
        # Compute the gradient of the top predicted class respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            class_channel, output_last_conv_layer = objective_function(img, **objective_function_params)

        # We are using as objective function the class channel and computing the gradient with respect to the output of the last conv layer
        grads = tape.gradient(class_channel, output_last_conv_layer)

        return grads, output_last_conv_layer

    def compute_grad_CAM(self, initial_image, conv_layer, pred_index=None):
        # Check that it's a numpy image
        assert isinstance(initial_image, np.ndarray) and initial_image.ndim >= 2

        # Assert that the layer is Conv2D
        layer_index_bool = type(conv_layer) == int
        layer_string_bool = type(conv_layer) == str
        assert (layer_index_bool and isinstance(self.model.layers[conv_layer], tf.keras.layers.Conv2D)) or (layer_string_bool and isinstance(self.model.get_layer(conv_layer), tf.keras.layers.Conv2D))

        # Remove last layer's softmax
        self.model.layers[-1].activation = None

        # Get the layer
        if layer_index_bool:
            model_layer = self.model.layers[conv_layer]
        else:
            model_layer = self.model.get_layer(conv_layer)

        # Take as output last conv layer activations and probabilities prediction
        n_model = tf.keras.Model(inputs=self.model.inputs, outputs=[model_layer.output, self.model.output])

        grads, output_conv_layer = self.__compute_gradient(initial_image, ObjectiveFunction.compute_grad_CAM_conv_function, 
            {'model':n_model, 'pred_index':pred_index})

        # Compute the guided gradients
        cast_conv_outputs = tf.cast(output_conv_layer > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads

        # The convolution and guided gradients have a batch dimension (which we don't need) so let's grab the volume itself and discard the batch
        output_conv_layer = output_conv_layer[0]
        guided_grads = guided_grads[0]

        # compute the average of the gradient values, and using them as weights, compute the ponderation of the filters with respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output_conv_layer), axis=-1)

        # Normalize the heatmap between 0 and 1
        heatmap = ImageProcessor.min_max_normalization(np.array([cam.numpy()])) [0]

        return heatmap

    def compute_grad_CAM_multiple_layers(self, initial_image, conv_layers, pred_index=None):
        # Check that it's a numpy image
        assert isinstance(initial_image, np.ndarray) and initial_image.ndim >= 2

        # Assert that the layer is Conv2D
        for conv_layer in conv_layers:
            layer_index_bool = type(conv_layer) == int
            layer_string_bool = type(conv_layer) == str
            assert (layer_index_bool and isinstance(self.model.layers[conv_layer], tf.keras.layers.Conv2D)) or (layer_string_bool and isinstance(self.model.get_layer(conv_layer), tf.keras.layers.Conv2D))

        # List to store heatmaps
        cam_list = []

        # Compute CAM of each layer
        for layer in conv_layers:
            cam = self.compute_grad_CAM(initial_image, layer, pred_index=pred_index)
            # Rseize the CAM
            cam = cv2.resize(cam, dsize=(initial_image.shape[0], initial_image.shape[1]))
            cam_list.append(cam)

        # Get fused CAM doing a mean
        fused_cam = np.mean(cam_list, axis=0)

        return fused_cam

    def compute_grad_CAM_all_conv_layers(self, initial_image, pred_index=None):
        # Check that it's a numpy image
        assert isinstance(initial_image, np.ndarray) and initial_image.ndim >= 2
        
        # Get all CONV2D layers
        conv2D_layer_indexes = [i for i in range(len(self.model.layers)) if len(self.model.layers[i].output_shape) == 4 and isinstance(self.model.layers[i], tf.keras.layers.Conv2D)]

        # Compute fused CAM
        fused_cam = self.compute_grad_CAM_multiple_layers(initial_image, conv2D_layer_indexes, pred_index=pred_index)

        return fused_cam

    def compute_heatmap_postprocessing(self, heatmap, resize_shape):
        # Check that it's a numpy array
        assert isinstance(heatmap, np.ndarray)

        # Rescale heatmap to a range 0-255
        heatmap = ImageProcessor.to_integer_conversion(np.array([heatmap])) [0]

        # Resize the heatmap
        heatmap = cv2.resize(heatmap, dsize=resize_shape)

        return heatmap
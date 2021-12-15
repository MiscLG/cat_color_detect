# import training as train  # import 3.'s training.py
"""
    These file contains the functions used to detect cat colors
"""
import sys
import os
from PIL import Image
import numpy as np
import keras.models

loc = os.path.dirname(os.path.realpath(__file__))
image_size = 160
detector_path = os.path.join(loc, 'color_detector')
weights_path = os.path.join(loc, 'snapshot/cat-bestmodel.hdf5')

source_dir = os.path.join(loc, "catimage")
categories = [name for name in os.listdir(
    source_dir) if name != ".DS_Store"]


def load_model():
    """loads the keras model from the saved files"""
    model = keras.models.load_model(detector_path)
    model.load_weights(weights_path)
    return model


def predict_color(image_path, model, verbose=False):
    """Applies the predictor to a given image"""
    image = get_image(image_path)
    predict = model.predict(image)
    color = None
    for i, pre in enumerate(predict):
        y = pre.argmax()
        color = categories[y]
        if verbose:
            print(sys.argv[i+1], categories[y])
    return color


def get_image(file_name):
    """retrieves an image from a file and returns it as an np array of pixels"""
    image_array = []
    file_name = os.path.abspath(file_name)
    img = Image.open(file_name)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    image_array.append(in_data)
    return np.array(image_array)


def main():
    if len(sys.argv) <= 1:
        quit()
    model = load_model()
    for file_name in sys.argv[1:]:
        print(detect_color(file_name, model))


if __name__ == "__main__":
    main()

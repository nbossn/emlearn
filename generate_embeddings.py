#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from absl import app, flags
from collections import deque
from load import load
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

FLAGS = flags.FLAGS
flags.DEFINE_string('embeddings_images_path', None, 'path to folder of images')
flags.DEFINE_string("embeddings_export_path",
                    "raw_embeddings",
                    "path to export embeddings")


class GenerateEmbeddings:

    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path
        self.base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=self.base_model.input,
                           outputs=self.base_model.get_layer('fc2').output)
        self.session = tf.Session()
        self.files, self.features = self.create_embeddings(image_folder_path)
        np.savez(FLAGS.embeddings_export_path,
                 files=self.files,
                 features=self.features)

    def preprocess_image(self, image):
        return preprocess_input(np.asarray(image))

    def get_features(self, images):
        extracted_features = deque()
        extracted_features.extend(self.model.predict(
            self.preprocess_image(images[0:])))
        return np.asarray(extracted_features, np.float32)

    # @ray.remote(num_return_vals=2)
    def create_embeddings(self, image_folder):
        features_deque = deque()
        names_deque = deque()
        for raw_images, file_paths, total in load(image_folder):
            features_deque.extend(self.get_features(raw_images))
            names_deque.extend(file_paths)
        self.session.close()
        return np.array(names_deque, np.unicode_), \
            np.array(features_deque, np.float32)


def main(argv):
    del argv  # unused
    GenerateEmbeddings(FLAGS.embeddings_images_path)


if __name__ == "__main__":
    app.run(main)

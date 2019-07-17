#!/usr/bin/env python
# -*- coding: utf-8 -*-
from absl import app, flags
from umap import GenerateUMAP
from generate_embeddings import GenerateEmbeddings
from pca import GeneratePCAEmbeddings

# TODO:
# implement ray
# load model from a tensorflow file as absl arg
# resize images if necessary

FLAGS = flags.FLAGS
flags.DEFINE_string('input_images', None, 'path to folder of images')
flags.mark_flag_as_required('input_images')

# ray.init(redis_address="localhost:6379")


def main(argv):
    del argv  # unused
    embeddings = GenerateEmbeddings(FLAGS.input_images)
    pca_embeddings = GeneratePCAEmbeddings(embeddings.files,
                                           embeddings.features)
    umap = GenerateUMAP(pca_embeddings.files,
                        pca_embeddings.PCA_embeddings)


if __name__ == "__main__":
    app.run(main)

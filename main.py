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
# provide pipeline to allow execution at multiple stages

FLAGS = flags.FLAGS
flags.DEFINE_string('images_path', None, 'path to folder of images')
flags.mark_flag_as_required('images_path')

# ray.init(redis_address="localhost:6379")

def main(argv):
    del argv  # unused
    embeddings = GenerateEmbeddings(FLAGS.images_path)
    pca_embeddings = GeneratePCAEmbeddings(embeddings.files, embeddings.features)
    umap_gen = GenerateUMAP(pca_embeddings.files, pca_embeddings.PCA_embeddings)


if __name__ == "__main__":
    app.run(main)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from absl import app, flags
from cuml.decomposition import PCA

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_embeddings_path', None, 'path to raw embeddings')
flags.DEFINE_string("pca_embeddings_path", "pca_embeddings",
                    "filename to save embeddings as")
flags.DEFINE_integer("pca_n_components", 512, "length of PCA embeddings")
flags.DEFINE_string("pca_svd_solver", "full", "PCA svd solver")
flags.DEFINE_integer("pca_iterated_power", 15, "PCA iterated power int")
flags.DEFINE_float("pca_tol", 1e-7, "PCA tolfloat")
flags.DEFINE_integer("pca_random_state", None, "PCA random state int")
flags.DEFINE_bool("pca_copy", True, "copies data then removes the mean")
flags.DEFINE_bool("pca_whiten", False, "de-correlate components")
flags.DEFINE_bool('pca_verbose', True, 'verbose for PCA')


class GeneratePCAEmbeddings:
    def __init__(self, files, features):
        self.files = files
        self.features = features
        self.PCA_embeddings = self.generate_PCA(self.features)
        np.savez(FLAGS.pca_embeddings_path,
                 files=self.files,
                 pca_embeddings=self.PCA_embeddings)

    def generate_PCA(self, features):
        pca = PCA(n_components=FLAGS.pca_n_components,
                  svd_solver=FLAGS.pca_svd_solver,
                  iterated_power=FLAGS.pca_iterated_power,
                  tol=FLAGS.pca_tol,
                  random_state=FLAGS.pca_random_state,
                  copy=FLAGS.pca_copy,
                  whiten=FLAGS.pca_whiten,
                  ).fit_transform(features)
        return pca.as_matrix()


def main(argv):
    del argv  # unused
    embeddings = np.load(FLAGS.raw_embeddings_path)
    GeneratePCAEmbeddings(embeddings['files'], embeddings['features'])


if __name__ == "__main__":
    app.run(main)

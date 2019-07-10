import cudf
import numpy as np
from absl import flags
from cuml.decomposition import PCA

FLAGS = flags.FLAGS
flags.DEFINE_string("pca_embeddings_path", "pca_embeddings", "path to export PCA embedddings")
flags.DEFINE_int("pca_n_components", 512, "length of PCA embeddings")
flags.DEFINE_string("pca_svd_solver", "full", "PCA svd solver")
flags.DEFINE_int("pca_iterated_powerint", 15, "PCA iterated power int")
flags.DEFINE_float("pca_tolfloat", 1e-7, "PCA tolfloat")
flags.DEFINE_int("pca_random_stateint", None, "PCA random state int")
flags.DEFINE_bool("pca_copyboolean", True, "copies data then removes the mean")
flags.DEFINE_bool("pca_whitenboolean", False, "de-correlate components")


class GeneratePCAEmbeddings:
    def __init__(self, files, features):
        self.files = files
        self.features = features
        self.PCA_embeddings = self.generate_PCA(self.features)
        np.savez(FLAGS.pca_embeddings_path, files=self.files, pca_embeddings=self.PCA_embeddings)

    def generate_PCA(self, features):
        pca = PCA(n_components=FLAGS.pca_n_components,
                  svd_solver=FLAGS.pca_svd_solver,
                  iterated_powerint=FLAGS.pca_iterated_powerint,
                  tolfloat=FLAGS.pca_tolfloat,
                  random_stateint=FLAGS.pca_random_stateint,
                  copyboolean=FLAGS.pca_copyboolean,
                  whitenboolean=FLAGS.pca_whitenboolean
                  ).fit_transform(features)
        return pca

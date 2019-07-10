import numpy as np
from absl import flags
from cuml.manifold.umap import UMAP

FLAGS = flags.FLAGS

flags.DEFINE_string("umap_path", "umap", "path to export UMAP")
flags.DEFINE_float("umap_n_neighbors", 15.0, "nearest neighbors")
flags.DEFINE_integer("umap_n_components", 3, "number of components")  # default is 2
flags.DEFINE_integer("umap_n_epochs", 200, "number of epochs")  # default is None
flags.DEFINE_float("umap_learning_rate", 1.0, "learning rate")
flags.DEFINE_string("umap_init", 'spectral', "dimension initialization")
flags.DEFINE_float("umap_min_dist", 0.1, "minimum distance")
flags.DEFINE_float("umap_spread", 1.0, "spread")
flags.DEFINE_float("umap_set_op_mix", 1.0, "set op mix ratio")
flags.DEFINE_integer("umap_local_connectivity", 1, "local connectivity")
flags.DEFINE_float("umap_repulsion_strength", 1.0, "repulsion strength")
flags.DEFINE_integer("umap_negative_sample_rate", 5, "negative sample rate")
flags.DEFINE_float("umap_transform_queue_size", 4.0, "transform queue size")
flags.DEFINE_bool("umap_verbose", False, "verbose")


class GenerateUMAP:
    def __init__(self, files, features):
        self.files = files
        self.features = features
        self.umap = self.create_umap(self.features)
        np.savez(FLAGS.umap_path, files=files, umap=self.umap)

    # @ray.remote
    def create_umap(self, features):
        # move flags to constructor?
        map = UMAP(n_neighbors=FLAGS.umap_n_neighbors,
                   n_components=FLAGS.umap_n_components,
                   n_epochs=FLAGS.umap_n_epochs,
                   learning_rate=FLAGS.umap_learning_rate,
                   init=FLAGS.umap_init,
                   min_dist=FLAGS.umap_min_dist,
                   spread=FLAGS.umap_spread,
                   set_op_mix_ratio=FLAGS.umap_set_op_mix,
                   local_connectivity=FLAGS.umap_local_connectivity,
                   repulsion_strength=FLAGS.umap_repulsion_strength,
                   negative_sample_rate=FLAGS.umap_negative_sample_rate,
                   transform_queue_size=FLAGS.umap_transform_queue_size,
                   a=None,
                   b=None,
                   verbose=FLAGS.umap_verbose
                   ).fit_transform(self.features)
        return map

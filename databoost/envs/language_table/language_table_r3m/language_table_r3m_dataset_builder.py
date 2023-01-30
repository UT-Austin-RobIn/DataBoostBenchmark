"""language_table_r3m dataset."""

import apache_beam as beam
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from r3m import load_r3m
from language_table.environments import language_table

SUBSEQ_LEN = 1
R3M_FEATURE_SIZE = 2048
ACTION_DIM = 2

ORIG_DATASET_PATH = "gs://gresearch/robotics/language_table_sim"

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for language_table_r3m dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      self.r3m = load_r3m("resnet50")  # resnet18, resnet34
      self.r3m.eval()
      self.r3m.to(self.device)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'observations': tfds.features.Tensor(shape=(SUBSEQ_LEN, R3M_FEATURE_SIZE), dtype=tf.float32),
            'actions': tfds.features.Tensor(shape=(SUBSEQ_LEN, ACTION_DIM), dtype=tf.float32),
        }),
        supervised_keys=None,
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # builder = tfds.builder_from_directory(os.path.join(ORIG_DATASET_PATH, '0.0.1'))

    # TODO(language_table_r3m): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(None, 'train'), #builder.as_dataset(split='train[:10]'), 'train'),
        # 'val': self._generate_examples(builder.as_dataset(split='val'), 'val'),
        # 'test': self._generate_examples(builder.as_dataset(split='test'), 'test'),
    }

  def _generate_examples(self, ds, split):
    """Yields examples."""
    def _generate_example(episode):
        # extract raw observation-action pair
        print("ALLO")
        obs, act = [], []
        instruct = None
        for step in episode['steps']:
            obs.append(step['observation']['rgb'])
            act.append(step['action'])
            if instruct is None:
                instruct = step['observation']['instruction'].numpy()

        # encode all observations with R3M
        imgs = torch.from_numpy(tf.stack(obs).numpy().transpose(0, 3, 1, 2)).to(self.device)
        encs = self.r3m(imgs).data.cpu().numpy()     # [seq_len, 2048]

        # write windows to output
        acts = tf.stack(act).numpy()                # [seq_len, 2]
        for k in range(len(act) - SUBSEQ_LEN):
            yield split + '_' + str(i) + '_' + str(k), {
                'observations': encs[i : i+SUBSEQ_LEN],
                'actions': acts[i : i+SUBSEQ_LEN],
            }

    builder = tfds.builder_from_directory("/data/karl/data/table_sim/language_table_sim/0.0.1")
    return (tfds.beam.ReadFromTFDS(builder, split='train[:2000]')
            | beam.FlatMap(_generate_example))

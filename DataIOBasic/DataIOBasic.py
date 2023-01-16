import os

import torch

import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss

os.chdir("/Users/user/Documents/Projects/SpeechBrain/src/SpeechBrain/DataIOBasic")


class SimpleBrain(speechbrain.Brain):
    def compute_forward(self, batch, stage):
        example_batch = batch
        x = self.modules.features(batch.signal.data)
        x = self.modules.encoder(x)
        x = self.modules.pooling(x, batch.signal.lengths)
        x = self.modules.to_output(x)
        return self.modules.softmax(x)

    def compute_objectives(self, predictions, batch, stage):
        return nll_loss(predictions, batch.spk_encoded.data)


def prepare_dataset():
    dataset = DynamicItemDataset.from_json("data.json")
    spk_id_encoder = CategoricalEncoder()
    spk_id_encoder.update_from_didataset(dataset, "spkID")
    dataset.add_dynamic_item(
        spk_id_encoder.encode_label_torch, takes="spkID", provides="spk_encoded"
    )
    dataset.add_dynamic_item(
        speechbrain.dataio.dataio.read_audio, takes="file_path", provides="signal"
    )
    dataset.set_output_keys(["id", "signal", "spk_encoded"])
    sorted_data = dataset.filtered_sorted(sort_key="length")
    return sorted_data, spk_id_encoder


def run():
    dataset, spk_id_encoder = prepare_dataset()

    modules = {
        "features": Fbank(left_frames=1, right_frames=1),
        "encoder": torch.nn.Sequential(torch.nn.Linear(40, 256), torch.nn.ReLU()),
        "pooling": speechbrain.nnet.pooling.StatisticsPooling(),
        "to_output": torch.nn.Linear(512, len(spk_id_encoder)),
        "softmax": speechbrain.nnet.activations.Softmax(apply_log=True),
    }

    brain = SimpleBrain(modules=modules, opt_class=lambda x: torch.optim.SGD(x, 1))

    brain.fit(
        range(3),
        train_set=dataset,
        train_loader_kwargs={"batch_size": 8, "drop_last": True},
    )


if "__main__" == "__main__":
    run()

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Training the model

Basic run (on CPU for 50 epochs):
    python examples/asr/speech_to_text.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=0 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.args.betas=[0.8,0.5] \
    model.optim.args.weight_decay=0.0001

Override optimizer entirely
    python speech_to_text.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""
import os
import jiwer
import gspread
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def eval(doc1, doc2):
  transform = jiwer.Compose([
      jiwer.RemovePunctuation(),
      jiwer.RemoveMultipleSpaces(),
      jiwer.ToLowerCase()
  ])

  hypothesis = transform(doc1)
  ground_truth = transform(doc2)

  measures = jiwer.compute_measures(ground_truth, hypothesis)

#   print('Evaluation:')
  scores = {
      'Substitutions': measures['substitutions'],
      'Deletions': measures['deletions'],
      'Insertions': measures['insertions'],
      'Total Error Rate': measures['wer'],
  }
#   print(data)
#   print('----------------------------------------')

  return scores['Total Error Rate']

def test(checkpoint_path):
    scores = []
    asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint_path)
    test_files = [file.split('.')[0].split('/')[-1] for file in os.listdir('/home/visha/NeMo_test/examples/asr/test_files') if file.split('.')[-1] == 'wav']
    for file in test_files:
        asr_out = asr_model.transcribe([f"/home/visha/NeMo_test/examples/asr/test_files/{file}.wav"])[0]
        truth = open(f"/home/visha/NeMo_test/examples/asr/test_files/{file}.txt").read()

        scores.append(eval(asr_out, truth))

    return scores

def update_sheet(scores, checkpoint_path, name):
    gc = gspread.service_account(filename='/home/visha/NeMo_test/examples/asr/desicrew-v1-088082cf46f3.json')
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1-OUI0RkDLE0OfOVS0a1PqBTNfGP0fjTMHkJWMUnoj9U/edit#gid=0")
    sheet = sh.worksheet('Sheet2')

    data = sheet.get_all_values()[1:]
    index = len(data)


    sheet.update_cell(index+2, 1, str(name))
    sheet.update_cell(index+2, 2, str(checkpoint_path))
    sheet.update_cell(index+2, 3, scores[0])
    sheet.update_cell(index+2, 3, scores[1])
    sheet.update_cell(index+2, 3, scores[2])
    sheet.update_cell(index+2, 3, scores[3])
    print('Sheet updated!')


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    log_directory = exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)
    
    checkpoints = [checkpoint for checkpoint in os.listdir(str(log_directory) + '/checkpoints') if checkpoint.split('.')[-1] == "ckpt"]

    for checkpoint_path in checkpoints:
        update_sheet(test(str(log_directory) + '/checkpoints/' + checkpoint_path), checkpoint_path, log_directory)
        


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

# Pytorch Image Classifier

## Task

- Build an image classifier for the 4 tissue types
- 100 images per class; equally distributed -> acc values have more meaning
- `.tif` file format with a resolution of `(2048x1536)`; 18MB per image
- Framework: PyTorch

## Preprocessing

- Initially create dataset with reduced resolution to develop pipeline and
faster feedback loop
- Image handling via Pillow
- Data split `0.8/0.1/0.1` for `test/eval/test`; images are shuffled
- Split can be disabled for crossvalidation
- Resolution set to `256x256`; default filter `BICUBIC`; file format `*.png`
- A job list is created and processed via `multiprocessing`

## Usage

- Use the `job_*.sh` scripts to commence data preparation, training,
  evaluation or crossvalidation
- Tasks are managed via `task.py`
- Assumed folder structure for datasets:

```bash
data
└── challenge_256
    ├── test
    |   ├── class1
    |   ├── class2
    |   ├── class3
    |   └── class4
    ├── training
    |   └── ...
    └── validation
        └── ...
```

- Subfolders `training`, `validation` and `test` are not applicable in case of
  crossvalidation

## Local results

- SimpleCNN: accuracy of ~0.48 on test set (`training_20201221201500`)
- CV: 0.50+=0.07 (10 fold)
- Restnet18: accuracy of ~0.80 on test set (`training_20201221204908`)
- CV: 0.78+=0.07 (10 fold)

## Cloud

- Test with AutoML
- `.tif` not supported currently
- Trying `.png`
- Current upload limit size at 30MB per file
- Rescaling to `256x256` yields 0.77, 0.74, 0.73 (acc, prec, rec)
- Original size yields 0.88, 0.84, 0.78 (acc, prec, rec)

## Next steps

- Checkpoints (pick specific epoch)
- Early stopping
- Both can be helpful to avoid overfitting
- Expand methods of augmentation:
  - Affine transformations
  - Cropping
  - Other noise forms
  - The paper also suggests occlusions
  - Maybe other tools for augmentation, e.g. Albumentations, Imgaug
  - More image inspection
- Artifact management, e.g. MLflow, Kubeflow, Allegro Trains(?)
- Better fitting metrics, e.g. Recall or F1
- Systematic hyperparameter tuning, e.g. Optuna, Hyperopt
- Other pretrained architectures (transfer learning)
- Potentially try ensemble methods or Neural Architecture Search
  - [Freiburg Group](https://www.automl.org/)
  - [Survey2020](https://arxiv.org/pdf/2006.02903.pdf)

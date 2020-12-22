# Pytorch Image Classifier

## Task

- Build an image classifier for the 4 tissue types
- 100 images per class; equally distributed -> acc is ok
- `.tif` file format with a resolution of `(2048x1536)`
- Based on PyTorch

## Preprocessing

- Initially create dataset with reduced resolution to develop pipeline
- Resolution set to `256x256`

## Usage

- Use the `job_*.sh` scripts to commence training or evaluation
- Tasks are managed via `task.py`
- Assumed folder structure:

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

## Local results

- SimpleCNN: accuracy of ~0.50 on test set (`training_20201221201500`)
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

- Checkpoints
- Early stopping
- Augmentation: affine transformations
- Better fitting metrics; probably recall
- Systematic hyperparameter tuning
- Crossvalidation for model training
- Other pretrained architectures
- Eventually NAS

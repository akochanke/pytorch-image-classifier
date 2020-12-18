# Pytorch Image Classifier

## Task

- Build an image classifier for the 4 tissue types
- 100 images per class; equally distributed -> acc is ok
- `.tif` file format with a resolution of `(2048x1536)`
- Based on PyTorch

## Preprocessing

- Initially create dataset with reduced resolution to develop pipeline

## Cloud

- `.tif` not supported currently
- Trying `.png`
- Current upload limit size at 30MB per file
- Rescaling to `256x256` yields 0.77, 0.74, 0.73 (acc, prec, rec)
- Original size yields 0.88, 0.84, 0.78 (acc, prec, rec)

## TODOs

- Load model and predict
- Opt: checkpoints

## Next steps

- Better fitting metrics; probably recall
- Systematic hyperparameter tuning
- Other pretrained architectures
- Eventually NAS

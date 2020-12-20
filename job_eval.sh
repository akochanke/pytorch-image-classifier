# evaluation job with previously trained model
python -m trainer.task \
    --job evaluation \
    --model artifacts/training_20201220192649 \
    --input data/challenge_256

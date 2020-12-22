# evaluation job with previously trained model

INPUT=data/challenge_256
MODEL_FOLDER=artifacts/training_20201221204908

python -m trainer.task \
    --job evaluation \
    --input ${INPUT} \
    --model ${MODEL_FOLDER}

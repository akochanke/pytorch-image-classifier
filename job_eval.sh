# evaluation job with previously trained model

INPUT=data/challenge_256
MODEL_FOLDER=artifacts/training_20201221204908 # resnet18
#MODEL_FOLDER=artifacts/training_20201221201500 # simplecnn
#MODEL_FOLDER=artifacts/training_20210104202754 # simplecnn step_size=5 0.53
#MODEL_FOLDER=artifacts/training_20210104205154 # resnet18 step_size=5
#MODEL_FOLDER=artifacts/training_20210104211235 # resnet18 setp_size=9 0.85
#MODEL_FOLDER=artifacts/training_20210105120117 # simplecnn step_size=7 0.58

python -m trainer.task \
    --job evaluation \
    --input ${INPUT} \
    --model ${MODEL_FOLDER}

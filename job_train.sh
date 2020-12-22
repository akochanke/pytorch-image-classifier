# script to start a training job

INPUT=data/challenge_256
MODEL_TYPE=resnet18
EXPORT=artifacts

python -m trainer.task \
    --job training \
    --input ${INPUT} \
    --model_type ${MODEL_TYPE} \
    --export ${EXPORT} \

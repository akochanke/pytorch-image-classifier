# script to start a crossvalidation job

INPUT=data/challenge_256_nosplit
MODEL_TYPE=resnet18
EXPORT=""

python -m trainer.task \
    --job crossvalidation \
    --input ${INPUT} \
    --model_type ${MODEL_TYPE}

# script to start a crossvalidation job

INPUT=data/challenge_256_nosplit # without train/eval/test split
#MODEL_TYPE=resnet18
MODEL_TYPE=simplecnn
#EXPORT=""

python -m trainer.task \
    --job crossvalidation \
    --input ${INPUT} \
    --model_type ${MODEL_TYPE}

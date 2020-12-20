# starting train job
python -m trainer.task \
    --job training \
    --input data/challenge_256 \
    --export artifacts \
    --model_type resnet18

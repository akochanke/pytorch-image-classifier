# script to invoke data_prep module

OUTPUT_FOLDER=data/challenge_256_nosplit

python -m data_prep.main \
    --input_folder data/challenge \
    --output_folder ${OUTPUT_FOLDER} \
    --resolution 256 \
#    --split ${SPLIT}

# script to invoke data_prep module

OUTPUT_FOLDER=data/try_256

python -m data_prep.main \
    --input_folder data/challenge \
    --output_folder ${OUTPUT_FOLDER} \
    --resolution 256 \
    --split

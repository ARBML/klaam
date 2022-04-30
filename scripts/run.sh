#!/bin/bash
python run_classifier.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --output_dir=/content/drive/MyDrive/Speech/dialects-classificationv3 \
    --cache_dir=/content/data_dir/ \
    --freeze_feature_extractor \
    --num_train_epochs="5" \
    --per_device_train_batch_size="16" \
    --preprocessing_num_workers="1" \
    --learning_rate="3e-5" \
    --warmup_steps="20" \
    --evaluation_strategy="steps"\
    --save_steps="1000" \
    --eval_steps="1000" \
    --save_total_limit="1" \
    --logging_steps="1000" \
    --do_eval \
    --do_train \

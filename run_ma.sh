#!/usr/bin/env bash
python run_recognition.py \
    --dataloader_num_workers="8" \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="ma_speech_corpus" \
    --train_split_name="train+dev" \
    --output_dir=./resources/output_models/ar/ma/wav2vec2-large-xlsr-arabic-ma \
    --cache_dir=./resources/data/ar \
    --freeze_feature_extractor \
    --num_train_epochs="50" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --gradient_accumulation_step="4"
    --preprocessing_num_workers="1" \
    --learning_rate="3e-5" \
    --warmup_steps="20" \
    --evaluation_strategy="steps"\
    --save_steps="100" \
    --eval_steps="100" \
    --save_total_limit="1" \
    --logging_steps="100" \
    --do_eval \
    --do_train \
    #--max_train_samples 100 \
    #--max_val_samples 100 \

#!/bin/bash

python ./run_generation.py \
--model_name_or_path /path/to/checkpoint \
--tokenizer_name t5-base \
--train_file /path/to/rct-train-de.csv \
--validation_file /path/to/rct-test-de.csv \
--output_dir /path/to/output_dir \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--text_column "inputs" \
--summary_column "targets" \
--num_train_epochs 1 \
--max_source_length 1800 \
--max_target_length 128 \
--min_length 65 \
--top_k 10 \
--temperature 0 \
--repetition_penalty 3 
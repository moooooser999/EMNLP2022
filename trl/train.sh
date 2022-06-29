#!/bin/bash

#for meteor_rl
python ./run_summarization_trl.py \
--model_name_or_path /path/to/xsum200k-pubmed-de \
--tokenizer_name t5-base \
--train_file /path/to/rct-train-de.csv \
--validation_file /path/to/rct-test-de.csv \
--output_dir /path/to/output_dir \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--text_column "inputs" \
--summary_column "targets" \
--num_train_epochs 5 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 4 \
--max_source_length 1800 \
--max_target_length 128 \
--min_length 65 \
--top_k 10 \
--temperature 0 \
--repetition_penalty 3 \
--reward_type "meteor" \
--reward_rate 0.5 \
--seed 0

#for bertscore_rl
python ./run_summarization_trl.py \
--model_name_or_path /path/to/xsum200k-pubmed-de \
--tokenizer_name t5-base \
--train_file /path/to/rct-train-de.csv \
--validation_file /path/to/rct-test-de.csv \
--output_dir /path/to/output_dir \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--text_column "inputs" \
--summary_column "targets" \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 4 \
--max_source_length 1024 \
--max_target_length 128 \
--min_length 65 \
--top_k 5 \
--temperature 0 \
--repetition_penalty 3 \
--reward_type "bertscore" \
--reward_rate 10 \
--seed 0

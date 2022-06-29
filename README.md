## Data
You can access rct200k data via: https://github.com/Franck-Dernoncourt/pubmed-rct (make sure to convert to csv file before preprocessing.)<br>
And rct summarization data is located at: https://github.com/bwallace/RCT-summarization-data , which is wallace mentioned below.
## Note
Since there are 2 different datasets, the ways we implement decoration are slightly differnt.<br>
Data from wallace need to be decorated before preprocessing.<br>
Data from RCT200K need to be preprocessed before decoration.<br>
And both of them need to be preprocessed before summerization.
## Set up environemnt for Scibert and Download Pretrained Model

You can pretrain the model from sketch via the instruction in scibert repo
```
git clone https://github.com/allenai/scibert.git
cd scibert
pip install -r requirements.txt
pip install overrides==3.1.0
```
## Create input for scibert
Preprocess the rct200k dataset before creating input for scibert
```
python3 preprocess/rct200k.py \
--dataset_dir path_to_rct200k_dataset \
--output_dir path_to_output
```
```
python3 ner/create_input.py \
--wallace_train_file_path  path_to_wallace_org_file (e.g. train-input.csv) \
--wallace_dev_file_path ... \
--wallace_test_file_path ... \
--rct200k_train_file_path path_to_preprocessed_rct200k_file \ 
--rct200k_dev_file_path ... \
--rct200k_test_file_path ... \

```
## Run NER
Run the following command with data created in `scibert_input`
```

python -m allennlp.run predict pretrained_model_path created_data_path --cuda-device 0 --silent --include-package scibert --use-dataset-reader --predictor sentence-tagger --output-file (output_path) -o {"dataset_reader":{"token_indexers":{"bert":{"truncate_long_sequences":false}}}}

```
## Decorate 
After NER prediction, run the following command
```
python3 ner/decorated.py \
--wallace_train_ner_output path_to_ner_output \
--wallace_dev_ner_output ... \
--wallace_test_ner_output ... \
--rct200k_train_ner_output ...\
--rct200k_dev_ner_output ...\
--rct200k_test_ner_output ...\
```

After the decoration, preprocess the wallace dataset
```
python3 preprocess/wallace.py \
--dataset_dir path_to_wallace_dataset\
--decorated_dir path_to_decorated_wallace \
--output_dir path_to_output
```

## Set up environemnt for training summarization model
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
cd examples/pytorch/summarization
pip install -r requirements.txt
```

## Train summarization model
We used t5-base finetuned with xsum dataset for 2 epochs as our pretrained model.<br>
(Note that T5-base is a quite large model, with 220M parameters, we pretrained it with NVIDIA Tesla P100and finetuned it with NVIDIA Tesla V100) <br>Runtime could differ depends on the computing devices(could be 10~20 hours).

```
python3 summarization/run_summarization.py \
--model_name_or_path path_to_pretrained_model \
--do_train \
--do_eval \
--train_file path_to_train_file \
--validation_file path_to_dev_file \
--output_dir path_to_output_dir \
--overwrite_output_dir \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--predict_with_generate True \
--max_source_length 2048 \
--max_target_length 128 \
--num_train_epochs 2 \
--text_column "input_decorated" \
--summary_column "targets"
```


## Generate output by summarization model
```
python3 summarization/run_generation.py \
--model_name_or_path path_to_pretrained_model \
--tokenizer_name t5-base \
--train_file path_to_train_file \
--validation_file path_to_test_file \
--output_dir path_to_output_dir \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--text_column "input_decorated" \
--summary_column "targets" \
--num_train_epochs 1 \
--max_source_length 2048 \
--max_target_length 128 \
--min_length 65 \
--top_k 10 \
--repetition_penalty 3
```

## Reinforcement Learning 

We trained RL with one NVIDIA RTX2080.

```
#please checkout setup.sh before start it.

bash ./trl/setup.sh;

```
```
# run trl/train.sh to start RL, note that there are two different scripts to train with different reward.
# remember to specify the path before running !!

bash train.sh

# to eval the result please refer to trl/eval.sh

bash eval.sh

```
## Evaluation Metrics
To evaluate the generated output we consider the following metrics:
- rouge
- bertscore
- meteor
- mnli
- [$\Delta$ EI][1]
- [FactCC][2]

[1]:https://github.com/jayded/evidence-inference
[2]:https://github.com/jayded/evidence-inference
Here is how we load the metrics
```
# Most of the metrics can be accessed via huggingface datasets library such as rouge, meteor, and bertscore.
load_metric('rouge')
load_metric('meteor')
load_metric('bertscore')

#For mnli we use the model from 'facebook/bart-large-mnli'.

logits = model(inputs,output_hidden_states=True)[0]
scores = torch.nn.Softmax(logtis)[0][2].item()

#For $\Delta$EI and FactCC, we do exactly the same as README.md in the original repo, please refer to those repo for more information.
```
## Human Evaluation

We ask Amazon Mechanical Turk to evaluate out generated summary in two different ways. 
Both of the tasks were done by workers with >95% Approval Rate.

- Wrong Sentences Detection
- Preference. 

For more information, please refer to "./crowd_worker_html"

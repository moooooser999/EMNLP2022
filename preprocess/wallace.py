import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Wallace Dataset for Summarization Task")
    parser.add_argument(
            '--dataset_dir',
            type = str,
            default = None,
            help='Path for wallace dataset directory' 
    )
    parser.add_argument(
            '--decorated_dir',
            type = str,
            default = None,
            help='Path for decorated wallace dataset directory' 
    )
    parser.add_argument(
            '--output_dir',
            type = str,
            default = None,
            help='Path for output directory' 
    )
    args = parser.parse_args()
    return args

# preprocessing wallace dataset
def preprocess(inputs_df, targets_df):
    reviewID, inputs_de, inputs, targets = [], [], [], []
    for i in range(len(inputs_df)):
        if inputs_df.iloc[i]['ReviewID'] not in reviewID:
            reviewID.append(inputs_df.iloc[i]['ReviewID'])
            if pd.isna(inputs_df.iloc[i]['Abstract_de']):
                input_de = '<T> ' + inputs_df.iloc[i]['Title']
            else:
                input_de = '<T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract_de']
            if pd.isna(inputs_df.iloc[i]['Abstract']):
                input = '<T> ' + inputs_df.iloc[i]['Title']
            else:
                input = '<T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract']
            inputs_de.append(input_de)
            inputs.append(input)
        else:
            idx = reviewID.index(inputs_df.iloc[i]['ReviewID'])
            if pd.isna(inputs_df.iloc[i]['Abstract_de']):
                inputs_de[idx] = inputs_de[idx] + ' <s> <T> ' + inputs_df.iloc[i]['Title']
            else:
                inputs_de[idx] = inputs_de[idx] + ' <s> <T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract_de']
            
            if pd.isna(inputs_df.iloc[i]['Abstract']):
                inputs[idx] = inputs[idx] + ' <s> <T> ' + inputs_df.iloc[i]['Title']
            else:
                inputs[idx] = inputs[idx] + ' <s> <T> ' + inputs_df.iloc[i]['Title'] + ' <ABS> ' + inputs_df.iloc[i]['Abstract']

    for i in range(len(targets_df)):
        targets.append(targets_df.iloc[i]['Target'])
    
    for i in range(len(inputs_de)):
        inputs_de[i] = ' '.join(inputs_de[i].replace('\n', ' ').split())

    for i in range(len(inputs)):
        inputs[i] = ' '.join(inputs[i].replace('\n', ' ').split())

    data = pd.DataFrame()
    data['ID'] = reviewID
    data['input_decorated'] = inputs_de
    data['inputs'] = inputs
    data['targets'] = targets
    return data

def main():
    args = parse_args()
    train_inputs_df = pd.read_csv(os.path.join(args.decorated_dir, "wallace_train_input_de.csv"))
    train_targets_df = pd.read_csv(os.path.join(args.dataset_dir, "train-targets.csv"))
    dev_inputs_df = pd.read_csv(os.path.join(args.decorated_dir, "wallace_dev_input_de.csv"))
    dev_targets_df = pd.read_csv(os.path.join(args.dataset_dir, "dev-targets.csv"))
    test_inputs_df = pd.read_csv(os.path.join(args.decorated_dir, "wallace_test_input_de.csv"))
    test_targets_df = pd.read_csv(os.path.join(args.dataset_dir, "test-targets.csv"))

    train_data = preprocess(train_inputs_df, train_targets_df)
    dev_data = preprocess(dev_inputs_df, dev_targets_df)
    test_data = preprocess(test_inputs_df, test_targets_df)

    train_data.to_csv(os.path.join(args.output_dir, 'wallace_train_de.csv'), index=False)
    dev_data.to_csv(os.path.join(args.output_dir, 'wallace_dev_de.csv'), index=False)
    test_data.to_csv(os.path.join(args.output_dir, 'wallace_test_de.csv'), index=False)

if __name__ == "__main__":
    main()
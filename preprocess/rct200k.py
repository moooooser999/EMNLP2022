import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RCT200K Dataset for Summarization Task")
    parser.add_argument(
            '--dataset_dir',
            type = str,
            default = None,
            help='Path for RCT200k dataset directory' 
    )
    parser.add_argument(
            '--output_dir',
            type = str,
            default = None,
            help='Path for output directory' 
    )
    args = parser.parse_args()
    return args

# preprocessing RCT200k dataset
def readfile(file_path):
    ID, inputs, targets = [], [], []
    flag = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                inputs.append(input.strip().replace('\t', ' '))
                targets.append(target.strip().replace('\t', ' '))
                flag = 0
                continue
            if flag:
                if line[:11] == 'CONCLUSIONS':
                    target += line[11:] + ' '
                else:
                    input += line + ' '
            else:
                if line[:3] == '###':
                    ID.append(line[3:])
                    input, target = '', ''
                    flag = 1
    
    df = pd.DataFrame()
    df['ID'] = ID
    df['inputs'] = inputs
    df['targets'] = targets
    df['targets'].replace('', np.nan, inplace=True)
    df.dropna(subset=['targets'], inplace=True)
    return df

def main():
    args = parse_args()
    train_df = readfile(os.path.join(args.dataset_dir, 'train.txt'))
    dev_df = readfile(os.path.join(args.dataset_dir, 'dev.txt'))
    test_df = readfile(os.path.join(args.dataset_dir, 'test.txt'))

    train_df.to_csv(os.path.join(args.output_dir, 'rct200k_train.csv'), index=False)
    dev_df.to_csv(os.path.join(args.output_dir, 'rct200k_dev.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'rct200k_test.csv'), index=False)

if __name__ == "__main__":
    main()
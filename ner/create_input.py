import pandas as pd
import argparse
import nltk
import os
from utils import create_input_wallace, create_input_rct200k



def parse_args():
    parser = argparse.ArgumentParser(description="Create Input for NER Task")
    parser.add_argument(
            '--wallace_train_file_path',
            type = str,
            default = None,
            help='Path for wallace original train-input.csv data' 
    )
    parser.add_argument(
            '--wallace_dev_file_path',
            type = str,
            default = None,
            help='Path for wallace original dev-input.csv data' 
    )
    parser.add_argument(
            '--wallace_test_file_path',
            type = str,
            default = None,
            help='Path for wallace original test-input.csv data' 
    )
    parser.add_argument(
            '--rct200k_train_file_path',
            type = str,
            default = None,
            help='Path for preprocessed rct200k train.csv data' 
    )
    parser.add_argument(
            '--rct200k_dev_file_path',
            type = str,
            default = None,
            help='Path for preprocessed rct200k dev.csv data' 
    )
    parser.add_argument(
            '--rct200k_test_file_path',
            type = str,
            default = None,
            help='Path for preprocessed rct200k test.csv data' 
    )
    args = parser.parse_args()
    return args


    
def main():
    args = parse_args()
    wallace_train_df = pd.read_csv(args.wallace_train_file_path)
    wallace_dev_df = pd.read_csv(args.wallace_dev_file_path)
    wallace_test_df = pd.read_csv(args.wallace_test_file_path)
    rct200k_train_df = pd.read_csv(args.rct200k_train_file_path)
    rct200k_dev_df = pd.read_csv(args.rct200k_dev_file_path)
    rct200k_test_df = pd.read_csv(args.rct200k_test_file_path)

    wallace_train_df = create_input_wallace('trian',wallace_train_df)
    wallace_dev_df = create_input_wallace('dev', wallace_dev_df)
    wallace_test_df = create_input_wallace('test', wallace_test_df)
    rct200k_train_df = create_input_rct200k('train', rct200k_train_df)
    rct200k_dev_df = create_input_rct200k('dev', rct200k_dev_df)
    rct200k_test_df = create_input_rct200k('test', rct200k_test_df)
    os.makedirs('./cache')
    wallace_train_df.to_csv('./cache/wallace_train_scibert.csv')
    wallace_dev_df.to_csv('./cache/wallace_dev_scibert.csv')
    wallace_test_df.to_csv('./cache/wallace_test_scibert.csv')
    rct200k_train_df.to_csv('./cache/rct200k_train_scibert.csv')
    rct200k_dev_df.to_csv('./cache/rct200k_dev_scibert.csv')
    rct200k_test_df.to_csv('./cache/rct200k_test_scibert.csv')
    

if __name__ == "__main__":
    main() 

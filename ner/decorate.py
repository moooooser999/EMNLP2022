import pandas as pd 
import argparse
import os
from utils import decorate_sent

def parse_args():
    parser = argparse.ArgumentParser(description="Create Input for NER Task")
    parser.add_argument(
            '--wallace_train_ner_output',
            type = str,
            default = None,
            help='Path for wallace train ner output' 
    )
    parser.add_argument(
            '--wallace_dev_ner_output',
            type = str,
            default = None,
            help='Path for wallace dev ner output' 
    )
    parser.add_argument(
            '--wallace_test_ner_output',
            type = str,
            default = None,
            help='Path for wallace test ner output' 
    )
    parser.add_argument(
            '--rct200k_train_ner_output',
            type = str,
            default = None,
            help='Path for rct200k ner output' 
    )
    parser.add_argument(
            '--rct200k_dev_ner_output',
            type = str,
            default = None,
            help='Path for rct200k dev ner output' 
    )
    parser.add_argument(
            '--rct200k_test_ner_output',
            type = str,
            default = None,
            help='Path for rct200k test ner output' 
    )
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    wallace_train_df = pd.read_csv('./cache/wallace_train_scibert.csv')
    wallace_dev_df = pd.read_csv('./cache/wallace_dev_scibert.csv')
    wallace_test_df = pd.read_csv('./cache/wallace_test_scibert.csv')
    rct200k_train_df = pd.read_csv('./cache/rct200k_train_scibert.csv')
    rct200k_dev_df = pd.read_csv('./cache/rct200k_dev_scibert.csv')
    rct200k_test_df = pd.read_csv('./cache/rct200k_test_scibert.csv')

    wallace_train_abstract = decorate_sent(wallace_train_df['sent_cnt'],args.wallace_train_ner_output)
    wallace_dev_abstract = decorate_sent(wallace_dev_df['sent_cnt'],args.wallace_dev_ner_output)
    wallace_test_abstract = decorate_sent(wallace_test_df['sent_cnt'],args.wallace_test_ner_output)

    rct200k_train_abstract = decorate_sent(rct200k_train_df['sent_cnt'],args.rct200k_train_ner_output)
    rct200k_dev_abstract = decorate_sent(rct200k_dev_df['sent_cnt'],args.rct200k_dev_ner_output)
    rct200k_test_abstract = decorate_sent(rct200k_test_df['sent_cnt'],args.rct200k_test_ner_output)

    os.makedirs('./decorated')
    os.makedirs('./decorated/wallace')
    os.makedirs('./decorated/rct200k')

    wallace_train_df['Abstract_de'] = wallace_train_abstract
    wallace_dev_df['Abstract_de'] = wallace_dev_abstract
    wallace_test_df['Abstract_de'] = wallace_test_abstract
    rct200k_train_df['input_decorated'] = rct200k_train_abstract
    rct200k_dev_df['input_decorated'] = rct200k_dev_abstract
    rct200k_test_df['input_decorated'] = rct200k_test_abstract

    wallace_train_df.to_csv('./decorated/wallace/wallace_train_input_de.csv')
    wallace_dev_df.to_csv('./decorated/wallace/wallace_dev_input_de.csv')
    wallace_test_df.to_csv('./decorated/wallace/wallace_test_input_de.csv')
    rct200k_train_df.to_csv('./decorated/rct200k/rct200k_train_de.csv')
    rct200k_dev_df.to_csv('./decorated/rct200k/rct200k_dev_de.csv')
    rct200k_test_df.to_csv('./decorated/rct200k/rct200k_test_de.csv')

if __name__ == "__main__":
    main()
    


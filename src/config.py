from pathlib import Path

RANDOM_STATE= 8

PARENT_DIR= Path.cwd()
OG_DATASET= PARENT_DIR / 'input/original_dataset.csv'
TRAIN_SET= PARENT_DIR / 'output/train.csv'
TEST_SET= PARENT_DIR / 'output/test.csv'

TRAIN_RESULT= PARENT_DIR / 'output/train_result.csv'
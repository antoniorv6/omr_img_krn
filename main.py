from CTCTraining import CTCTraining
from EncoderTraining import EncoderTrain
from utils.utils import DATA_TYPE
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train ISMIR 2020 from Image to **kern models')
    parser.add_argument('-data_path', dest='data_path', type=str, required=True, help='Path to data lists.')
    args = parser.parse_args()
    
    CTCTraining((DATA_TYPE.SKM).value, args.data_path)
    CTCTraining((DATA_TYPE.KERN).value, args.data_path)
    CTCTraining((DATA_TYPE.AGNOSTIC).value, args.data_path)
    EncoderTrain((DATA_TYPE.SKM).value, args.data_path)
    EncoderTrain((DATA_TYPE.KERN).value, args.data_path)

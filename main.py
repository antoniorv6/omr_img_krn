from CTCTraining import CTCTraining, RetrieveStats, CTCTest
import EncoderTraining
from utils.utils import DATA_TYPE
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train ISMIR 2020 from Image to **kern models')
    
    parser.add_argument('-data_path', dest='data_path', type=str, required=True, help='Path to data lists.')
    parser.add_argument('-output', dest='output', type=int, default=None, required=True, help='The output encoding you want to do -- 1: Agnostic, 2: **kern, 3: **skm')

    args = parser.parse_args()
    
    #CTCTraining(args.output, args.data_path)
    CTCTest(args.output, args.data_path)
    #RetrieveStats((DATA_TYPE.KERN).value, args.data_path)
    

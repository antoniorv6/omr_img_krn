import CTCTraining
import EncoderTraining
from utils.utils import DATA_TYPE
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

functionsDictionary = {
    'CTC' : CTCTraining.CTCTraining, 
    'Encoder' : EncoderTraining.EncoderTrain
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train ISMIR 2020 from Image to **kern models')
    
    parser.add_argument('-data_path', dest='data_path', type=str, required=True, help='Path to data lists.')
    parser.add_argument('-train_type', dest='train_type', type=str, required=True, help='Which training function to call.')
    parser.add_argument('-output', dest='output', type=int, default=None, required=True, help='The output encoding you want to do -- 1: Agnostic, 2: **kern, 3: **skm')

    args = parser.parse_args()
    
    functionsDictionary[args.train_type](args.output, args.data_path)
    
    #CTCTraining((DATA_TYPE.SKM).value, args.data_path)
    #CTCTraining((DATA_TYPE.KERN).value, args.data_path)
    #CTCTraining((DATA_TYPE.AGNOSTIC).value, args.data_path)
    #EncoderTrain((DATA_TYPE.SKM).value, args.data_path)
    #EncoderTrain((DATA_TYPE.KERN).value, args.data_path)

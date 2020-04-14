from CTCTraining import CTCTraining
from EncoderTraining import EncoderTrain
from utils.utils import DATA_TYPE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    #CTCTraining((DATA_TYPE.SKM).value)
    EncoderTrain((DATA_TYPE.SKM).value)

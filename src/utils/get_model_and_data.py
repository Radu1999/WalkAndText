from ..datasets.gait import Gait
from ..models.get_model import get_model_gait
import clip

def get_model_and_data_gait(parameters):
    train = Gait(datapath="caption_retrieval_dataset_train.pt")
    test = Gait(datapath="caption_retrieval_dataset_val.pt")
    model = get_model_gait(parameters)
    return model, {"train": train, "test": test}
        

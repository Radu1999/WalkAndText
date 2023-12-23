from ..datasets.get_dataset import get_datasets
from ..models.get_model import get_model as get_gen_model
import clip

def get_model_and_data(parameters, split="train", descriptions=None):
    datasets = get_datasets(parameters, split=split)
    model = get_gen_model(parameters, descriptions)
    return model, datasets

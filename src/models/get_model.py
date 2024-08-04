from src.models.architectures.transformer import Encoder_TRANSFORMER
from src.models.modeltype.walk_text import WalkText

def get_model_gait(parameters, **kwargs):
    encoder = Encoder_TRANSFORMER(**parameters)
    return WalkText(encoder, **parameters).to(parameters["device"])
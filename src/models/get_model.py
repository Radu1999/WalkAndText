from src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from src.models.modeltype.motionclip import CLIPose
from src.models.modeltype.transformer_classifier import PoseClassifier

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "rcxyz", "vel", "velxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

def get_model(parameters, **kwargs):
    encoder = Encoder_TRANSFORMER(**parameters)
    decoder = Decoder_TRANSFORMER(**parameters)
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    if parameters.get("model", "default") != "default":
        return PoseClassifier(encoder, **parameters).to(parameters["device"])
    return CLIPose(encoder, **parameters).to(parameters["device"])

from vit_v2 import Vit, VitHPs
from ml_model import exists as ml_model_exists
from ml_model import MLModel


def get_ml_model_constructor(model_name):
    # check if the model name is vit, not considering capitalization
    if model_name.lower() == "vit":
        return Vit
    else:
        raise Exception(f"Model {model_name} does not exist.")
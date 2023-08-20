from vit_v2 import Vit

def get_ml_model_constructor(model_name):
    # check if the model name is vit, not considering capitalization
    if model_name.lower() == "vit":
        return Vit
    else:
        raise Exception(f"Model {model_name} does not exist.")
# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment to disable GPU
from model import Model, DatasetName, load_model, remove_model
import numpy as np

__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../data_collectors/")
from covid19_genome import Covid19Genome

os.chdir(__ORIG_WD__)


# %%
try:
    model = load_model("covid19-reads-1024examples")
except Exception:
    covid19_genome = Covid19Genome()
    lineages = covid19_genome.getLocalLineages(1024)
    lineages.sort()
    dataset = []
    def get_dataset(lower, upper):
        for lineage in lineages[lower:upper]:
            dataset.append((lineage, covid19_genome.getLocalAccessionsPath(lineage)))
        return dataset

    portions = {
        DatasetName.trainset.name: 0.8,
        DatasetName.validset.name: 0.1,
        DatasetName.testset.name: 0.1
    }

    dataset = get_dataset(0, 200)
    model = Model("covid19-reads-1024examples")
    model.create_datasets(dataset, portions, minhash_dataset=True)

# %%
# model.remove_ml_model("vit.2.00001.adamw.coverage4")
try: 
    model.add_ml_model("vit", "vit.4.00001.adamw.coverage4", hps={
        "optimizer": {
            "name": "AdamW",
            "params": {
                "learning_rate": 0.001,
            },
        },
        "encoder_repeats": 4,
        "batch_size": 256,
        "regularizer": {
            "name": "l2",
            "params": {
                "l2": 0.0003
            }
        },
        "d_key": 128,
        "d_value": 128,
        "d_ff": 1024+256,
        "dropout": 0.2,
    })
except:
    print("Model already exists")

# %%
models = model.list_ml_models()
print(models)

# %%
# model.transfer("vit.2.00001.adamw.coverage4", "vit.3.00001.adamw.coverage4")

# %%
ml_model_name = 'vit.4.00001.adamw.coverage4'

# %%
model.change_ml_hps(ml_model_name, {
    "regularizer": {
        "name": "l2",
        "params": {
            "l2": 0.0004,
        },
    },
    "optimizer": {
        "name": "AdamW",
        "params": {
            "learning_rate": 0.00004,
        },
    },
})

# %%
model.set_coverage(4)
model.train(ml_model_name, epochs=1000)

# %%
model._load_ml_model(ml_model_name)
cnt = 0
for layer in model.ml_models[ml_model_name].net.layers:
    if (cnt > 2):
        layer.trainable = True
    else:
        layer.trainable = False
    cnt += 1
    print(layer.name, layer.trainable)

# %%




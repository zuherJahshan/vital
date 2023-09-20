import sys
sys.path.append(".")

import os
import argparse
import csv

parser = argparse.ArgumentParser(
    prog="ViTAL",
    description='''
    ViTAL, the lineage assignment algorithm proposed here, inputs
    a low-coverage genome, transforms it into embedded genome
    fragments which are then fed into a classification neural
    network, that outputs the most likely lineages the input genome
    might belong to
    ''',
)
parser.add_argument("-i", "--inputDir", type=str, help="The input directory containing the genomes (in fasta format) to be classified", required=False)
parser.add_argument("-c", "--coverage", type=int, help="The coverage of the genome to be classified", required=False)
parser.add_argument("-o", "--outputFile", type=str, help="The output file path where the classification results will be written to", required=False)
parser.add_argument("-m", "--ml-model-name", type=str, help="The model to be used for classification", required=False)
parser.add_argument("-k", "--top-k", type=int, help="The number of top lineages to be returned", required=False)
parser.add_argument(
    "--print-ml-models",
    action="store_true",
    help="""Prints the available ml_models,
    the naming of the models is as follows <ml structure>.<number of encoders>.<coverage>x.<sequencer_platform>""",
    required=False
)

args = parser.parse_args()

##################################
##################################
##### Activating the model #######
__ORIG_WD__ = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import glob


def load_model_aux():
    from model import Model, DatasetName, load_model, remove_model
    model_name = "covid19-1024examples"
    model = load_model(model_name)
    return model

if args.top_k:
    top_k = args.top_k
else:
    top_k = 1

if args.print_ml_models:
    model = load_model_aux()
    print("##############################")
    print("### Available models are: ####")
    print("##############################")
    for model_name in model.list_ml_models():
        print(model_name)
    print("##############################")
    exit(0)

elif args.inputDir and args.coverage:
    model = load_model_aux()
    ml_models = model.list_ml_models()
    if args.ml_model_name:
        ml_model_name = args.ml_model_name
    else:
        if len(ml_models) > 0:
            ml_model_name = ml_models[0]
        else:
            print("No models available, please train a model first, at the vital/models/covid_tran.ipynb")
            exit(0)
    if not ml_model_name in ml_models:
        print(f"Model {ml_model_name} not available, please choose one of the following:")
        for model_name in ml_models:
            print(model_name)
        exit(0)

    predictions = model.predict(
        ml_model_name=ml_model_name,
        dataset_type="MHGenomeDS",
        dataset_props={"coverage":32},
        examples=glob.glob(f"{args.inputDir}/*.fasta") + glob.glob(f"{args.inputDir}/*.fa") + glob.glob(f"{args.inputDir}/*.fna"),
        top_k=top_k,
    )
    if args.outputFile:
        output_file = args.outputFile
    else:
        output_file = "../predictions.csv"

    # write to the output file as a csv file
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        descriptor_raw = ["genome"]
        for i in range(top_k):
            descriptor_raw.append(f"predicted lineage {i+1}")
            descriptor_raw.append(f"predicted lineage {i+1} probability")
        writer.writerow(descriptor_raw)
        for genome, prediction in predictions.items():
            result_raw = [genome]
            for i in range(top_k):
                result_raw.append(prediction["labels"][i])
                result_raw.append(prediction["probs"][i])
            writer.writerow(result_raw)



    print(predictions)

else:
    print("Please read the help message by running: python vital.py -h")

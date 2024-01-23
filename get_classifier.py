import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import urllib
from tqdm import tqdm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
import matplotlib as mpl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Draw import MolToFile
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import requests
import plotly.express as px
from IPython.display import HTML
from utility import *
from pyteomics import mgf
from Classic import *
from multiprocessing import Pool
import collections
from typing import List, Tuple
import pickle
import os
import argparse
import random



if __name__ == '__main__':
    #pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('--input', type=str,required=True,default="input_library.txt", help='input libraries')
    args = parser.parse_args()
    input_lib_file = args.input

    #read libraries from input file
    with open(input_lib_file,'r') as f:
        libraries = f.readlines()

    for library in libraries:
        library = library.strip('\n')
        print("starting benchmarking library:"+library)
        summary_file_path = "./data/summary/"+library+"_summary.tsv"
        merged_pairs_file_path = "./data/merged_paris/"+library+"_merged_pairs.tsv"
        cluster_summary_df = pd.read_csv(summary_file_path)
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        print('graph with {} nodes and {} edges'.format(G_all_pairs.number_of_nodes(), G_all_pairs.number_of_edges()))
        print("constructing dic for finger print")
        smiles_list = cluster_summary_df['Smiles'].tolist()
        print(smiles_list)
        data_list = []
        for smiles in tqdm(smiles_list):
            try:
                smiles_string=urllib.parse.quote(smiles.encode("utf-8"))
                api_url = f"https://structure.gnps2.org/classyfire?smiles={smiles_string}"
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    data_list.append(data)
                else:
                    print(f"API request for SMILES {smiles} failed with status code:", response.status_code)
            except Exception as e:
                print(e)

        # Process the collected data and create a DataFrame
        df_list = []
        for data in data_list:
            smiles = data["smiles"]
            inchikey = data["inchikey"]
            description = data.get("description", "N/A")  # Use a default value if "description" is not present
            kingdom_name = data["kingdom"]["name"] if data.get("kingdom") else "N/A"
            superclass_name = data["superclass"]["name"] if data.get("superclass") else "N/A"
            class_name = data["class"]["name"] if data.get("class") else "N/A"
            subclass_name = data["subclass"]["name"] if data.get("subclass") else "N/A"

            data_dict = {
                "SMILES": smiles,
                "InChIKey": inchikey,
                "Description": description,
                "Kingdom": kingdom_name,
                "Superclass": superclass_name,
                "Class": class_name,
                "Subclass": subclass_name
            }

            df_list.append(pd.DataFrame(data_dict, index=[0]))  # Each DataFrame has only one row

        # Concatenate all DataFrames into a single DataFrame
        merged_df = pd.concat(df_list, ignore_index=True)

        # Display the merged DataFrame
        print(merged_df)
        csv_filename = library + "_classifier_results.csv"
        merged_df.to_csv(csv_filename, index=False)

        print(f"Merged DataFrame saved to {csv_filename}")
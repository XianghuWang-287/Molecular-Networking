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

# classifier_df = pd.read_csv("./save_rosetta_ClassyfireOverhaul.csv",sep=',')
# library = "GNPS-SELLECKCHEM-FDA-PART2"
# summary_file_path = "./data/summary/"+library+"_summary.tsv"
# merged_pairs_file_path = "./data/merged_paris/"+library+"_merged_pairs.tsv"
# cluster_summary_df = pd.read_csv(summary_file_path)
# all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')

# for index, row in cluster_summary_df.iterrows():
#     print("Spectrum ID:", row["spectrum_id"])
#     if(classifier_df["Lib_ids"].isin([row["spectrum_id"]]).any()):
#         print("library smiles:",row['Smiles'],"classifier df smiles:",classifier_df.loc[classifier_df["Lib_ids"]==row["spectrum_id"]]["SMILES"].iloc[0])
#     else:
#         print("missing data")
def calculate_correct_classification_percentage(components, node_types_df, threshold=0.7):
    correctly_classified_clusters = 0
    total_clusters = len(components)

    for component in components:
        cluster_nodes = list(component.nodes())
        node_index = [node - 1 for node in cluster_nodes]
        cluster_types = node_types_df.loc[node_index, 'superclass_results']
        cluster_types = cluster_types.replace({np.nan: "Unknown"})
        dominant_type = cluster_types.value_counts().idxmax()
        if dominant_type == "Unknown":
            print("skip")
            continue
        percentage = cluster_types.value_counts()[dominant_type] / len(cluster_nodes)

        if percentage >= threshold:
            correctly_classified_clusters += 1

    return correctly_classified_clusters / total_clusters * 100

def edge_correct_classification_percentage(components,node_types_df):
    correct_num = 0
    total_number = 0
    for component in components:
        total_number = total_number + component.number_of_edges()
        for edge in component.edges():
            node1 = edge[0]
            node2 = edge[1]
            node1_type = node_types_df.iloc[node1-1]['superclass_results']
            node2_type = node_types_df.iloc[node2-1]['superclass_results']
            if pd.isna(node1_type) or pd.isna(node2_type):
                continue
            if node1_type == node2_type:
                correct_num = correct_num+1
    return correct_num/total_number




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
        print("starting benchmarking library:" + library)
        summary_file_path = "./data/summary/" + library + "_summary.tsv"
        merged_pairs_file_path = "./data/merged_paris/" + library + "_merged_pairs.tsv"
        classifer_file_path = "./data/"+library+"_NP.tsv"

        cluster_summary_df = pd.read_csv(summary_file_path)
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        classifer_df = pd.read_csv(classifer_file_path, sep='\t')

        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        print('graph with {} nodes and {} edges'.format(G_all_pairs.number_of_nodes(), G_all_pairs.number_of_edges()))
        print("constructing dic for finger print")
        dic_fp = fingerprint_dic_construct(cluster_summary_df)
        y_weight_avg = []
        components_all_list = []
        results_df_list = []
        for max_k in tqdm(range(1, 40, 2)):
            for max_c in range(2, 102, 5):
                G_all_pairs_filter = G_all_pairs.copy()
                filter_top_k(G_all_pairs_filter, max_k)
                filter_component(G_all_pairs_filter, max_c)
                G_all_pairs_filter_copy = G_all_pairs_filter.copy()
                for node in G_all_pairs_filter_copy.nodes():
                    if G_all_pairs_filter.degree[node] == 0:
                        G_all_pairs_filter.remove_node(node)
                score_all_pairs_filter_list = []
                components = [G_all_pairs_filter.subgraph(c).copy() for c in nx.connected_components(G_all_pairs_filter)]
                # percentage_correctly_classified = calculate_correct_classification_percentage(components,classifer_df,0.7)
                # print(f"Percentage of correctly classified clusters: {percentage_correctly_classified:.2f}%")
                percentage_correct_edges = edge_correct_classification_percentage(components, classifer_df)
                print(percentage_correct_edges)
                results_df_list.append(percentage_correct_edges)
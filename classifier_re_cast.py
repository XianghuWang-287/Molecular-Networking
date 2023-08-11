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
def add_edges_to_mst(original_graph, mst):
    remaining_edges = [(u, v, original_graph[u][v]['Cosine']) for u, v in original_graph.edges() if not mst.has_edge(u, v)]
    remaining_edges.sort(key=lambda x: x[2], reverse=True)

    average_weight = calculate_average_weight(mst)

    for u, v, weight in remaining_edges:
        mst.add_edge(u, v, Cosine=weight)
        new_average_weight = calculate_average_weight(mst)
        if new_average_weight >= average_weight:
            mst.remove_edge(u, v)
            break
        average_weight = new_average_weight

    return mst

def calculate_average_weight(graph):
    total_weight = sum(graph[u][v]['Cosine'] for u, v in graph.edges())
    average_weight = total_weight / graph.number_of_edges()
    return average_weight

def polish_subgraph(G):
    if G.number_of_edges() == 0:
        return G
    maximum_spanning_tree = nx.maximum_spanning_tree(G, weight='Cosine')
    polished_subgraph = add_edges_to_mst(G, maximum_spanning_tree)
    return polished_subgraph
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
            continue
        percentage = cluster_types.value_counts()[dominant_type] / len(cluster_nodes)

        if percentage >= threshold:
            correctly_classified_clusters += 1

    return correctly_classified_clusters / total_clusters

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
        re_align_edge_file_path = "./alignment_results/" + library + "_realignment.pkl"
        classifer_file_path = "./data/" + library + "_NP.tsv"
        cluster_summary_df = pd.read_csv(summary_file_path)
        classifer_df = pd.read_csv(classifer_file_path, sep='\t')
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        print('graph with {} nodes and {} edges'.format(G_all_pairs.number_of_nodes(), G_all_pairs.number_of_edges()))
        print("constructing dic for finger print")
        dic_fp = fingerprint_dic_construct_InCHI(cluster_summary_df)
        print("constructing the re-alignment graph")
        G_all_pairs_realignment = G_all_pairs.copy()
        with open(re_align_edge_file_path, 'rb') as f:
            realignment_edgelist = pickle.load(f)
        new_list = []
        for item in realignment_edgelist:
            if (item != None):
                if item[2] >= 0.65:
                    new_list.append(item)
        print(len(new_list))
        for item in new_list:
            if (item != None):
                G_all_pairs_realignment.add_edge(item[0], item[1], Cosine=item[2])
        results_df_list = []
        thresholds = [x / 100 for x in range(70, 95)]
        for threshold in tqdm(thresholds):
            cast_cluster = CAST_cluster(G_all_pairs_realignment, threshold)
            cast_score_list = []
            cast_components = [G_all_pairs_realignment.subgraph(c).copy() for c in cast_cluster]
            # percentage_correctly_classified = calculate_correct_classification_percentage(cast_components, classifer_df, 0.7)
            # print(f"Percentage of correctly classified clusters: {percentage_correctly_classified:.2f}%")
            benchmark_set = []
            for component in cast_components:
                benchmark_set.append(polish_subgraph(component))
            percentage_correct_edges =  edge_correct_classification_percentage(benchmark_set,classifer_df)
            print(percentage_correct_edges)
            results_df_list.append(percentage_correct_edges)
        print(results_df_list)

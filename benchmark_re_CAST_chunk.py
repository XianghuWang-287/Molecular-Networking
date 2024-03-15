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
def weighted_average(dataframe, value, weight):
    # Filter out rows where the value (score) is 0
    filtered_df = dataframe[dataframe[value] != 0]
    val = filtered_df[value]
    wt = filtered_df[weight]
    # If after filtering, the dataframe is empty or all weights are zero, return NaN or some default value
    if wt.sum() == 0:
        return float('nan')  # or return 0 or any other appropriate value depending on your context
    return (val * wt).sum() / wt.sum()

def cal_N50(df, node_numbers,N_ratio):
    dfnew=df.sort_values('number',ascending=False)
    print(dfnew)
    number=dfnew.values[0][1]
    row_old = dfnew.values[0][1]
    if len(dfnew.values) ==1:
        return row_old
    for row in dfnew.values[1:]:
        if (number >= node_numbers*N_ratio):
            return row_old
        else:
            number=number+ row[1]
            row_old = row[1]
    return 1

def add_edges_to_mst(original_graph, mst):
    remaining_edges = [(u, v, original_graph[u][v]['Cosine']) for u, v in original_graph.edges() if not mst.has_edge(u, v)]
    remaining_edges.sort(key=lambda x: x[2], reverse=True)

    average_weight = calculate_average_weight(mst)

    for u, v, weight in remaining_edges:
        mst.add_edge(u, v, Cosine=weight)
        new_average_weight = calculate_average_weight(mst)
        if new_average_weight <= average_weight:
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

def sample_graph_by_probability(graph, sample_percentage):
    node_list=[]
    for node in graph.nodes():
        if random.random() <= sample_percentage:
            node_list.append(node)
    return graph.subgraph(node_list).copy()
def update_graph_with_alignment_results(G, alignment_results, min_score):
    new_alignment_results = []
    for node1, node2, score in tqdm(alignment_results):
        if score >= min_score:
            G.add_edge(node1, node2, Cosine=score, origin='transitive_alignment')
            new_alignment_results.append((node1, node2, score))
    print(len(new_alignment_results))
    return G
def load_transitive_alignment_results(folder_path):
    all_pairs = []
    for filename in os.listdir(folder_path):
        if filename.startswith("chunk_") and filename.endswith("_realignment.pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                # Load the node pairs and scores
                pairs = pickle.load(file)
                all_pairs.extend(pairs)
    return all_pairs


def plot_cluster_distribution(cast_score_list, cast_number):
    """
    Plots the distribution of cluster sizes against their scores for a given threshold.

    Args:
        cast_score_list (list): List of scores for each cluster.
        cast_number (list): List of sizes for each cluster.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(cast_number, cast_score_list, alpha=0.6)
    plt.title('Cluster Size vs. Score Distribution for Threshold = 0.65')
    plt.xlabel('Cluster Size')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

def polish_subgraph_MST(G):
    # Filter the graph to include only original edges
    original_edges_graph = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d.get('origin') != 'transitive_alignment')
    # Attempt to create MST using only original edges
    try:
        mst = nx.maximum_spanning_tree(original_edges_graph, weight='Cosine')
        if len(mst.nodes()) < len(G.nodes()):
            raise ValueError("MST does not span all nodes with original edges only.")
    except ValueError:
        # Original edges alone do not span all nodes; include transitive alignment edges
        print("MST does not span all nodes with original edges")
        combined_graph = nx.Graph((u, v, d) for u, v, d in G.edges(data=True))  # This includes all edges
        mst = nx.maximum_spanning_tree(combined_graph, weight='Cosine')
        # Incrementally add transitive alignment edges to span all nodes, if necessary

    # Further refinement to add transitive alignment edges only if they improve connectivity can be implemented here

    return mst

if __name__ == '__main__':
    #pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-t', type=str, required=True,
                        default="/home/user/LabData/XianghuData/Transitive_Alignment_Distributed/nf_output/transitive_alignment",
                        help='transitive_alignment folder path')
    parser.add_argument('--input', type=str,required=True,default="input_library.txt", help='input libraries')
    args = parser.parse_args()
    input_lib_file = args.input
    transitive_alignment_folder = args.t

    #read libraries from input file
    with open(input_lib_file,'r') as f:
        libraries = f.readlines()

    for library in libraries:
        library = library.strip('\n')
        print("starting benchmarking library:"+library)
        summary_file_path = "./data/summary/"+library+"_summary.tsv"
        merged_pairs_file_path = "./data/merged_paris/"+library+"_merged_pairs.tsv"
        cluster_summary_df = pd.read_csv(summary_file_path, sep='\t')
        print(cluster_summary_df)
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        print('graph with {} nodes and {} edges'.format(G_all_pairs.number_of_nodes(), G_all_pairs.number_of_edges()))
        print("constructing dic for finger print")
        dic_fp = fingerprint_FBMN_dic_construct(cluster_summary_df)
        print("constructing the re-alignment graph")
        G_all_pairs_realignment = G_all_pairs.copy()
        alignment_results = load_transitive_alignment_results(transitive_alignment_folder)
        G_all_pairs_realignment = update_graph_with_alignment_results(G_all_pairs_realignment, alignment_results, 0.6)
        results_df_list = []
        thresholds = [0.65,0.7,0.75,0.80]
        for threshold in tqdm(thresholds):
            cast_cluster = CAST_cluster(G_all_pairs_realignment, threshold)
            cast_score_list = []
            cast_components = [G_all_pairs_realignment.subgraph(c).copy() for c in cast_cluster]
            benchmark_set = []
            for component in cast_components:
                benchmark_set.append(polish_subgraph_MST(component))
            for component in benchmark_set:
                cast_score_list.append(subgraph_score_dic(component, cluster_summary_df, dic_fp))
            cast_number = [len(x) for x in cast_cluster]
            print(cast_number)
            plot_cluster_distribution(cast_score_list, cast_number)
            df_cast = pd.DataFrame(list(zip(cast_score_list, cast_number)), columns=['score', 'number'])
            results_df_list.append(df_cast)
        result_file_path = "./results-cast/" + library + "_cast_benchmark.pkl"
        x_re_results = np.array([cal_N50(x, 12194, 0.2) for x in results_df_list])
        y_re_results = np.array([weighted_average(x, 'score', 'number') for x in results_df_list])
        print(x_re_results)
        print(y_re_results)
        with open(result_file_path, 'wb') as file:
            pickle.dump(results_df_list, file)






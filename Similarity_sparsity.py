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
import math
import os
import argparse

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
        re_align_edge_file_path = "./alignment_results/"+library+"_realignment.pkl"
        cluster_summary_df = pd.read_csv(summary_file_path)
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        print('graph with {} nodes and {} edges'.format(G_all_pairs.number_of_nodes(), G_all_pairs.number_of_edges()))
        print("constructing dic for finger print")
        dic_fp = fingerprint_dic_construct(cluster_summary_df)
        G_all_pairs_structure = nx.Graph()
        for i in tqdm(range(1,G_all_pairs.number_of_nodes()+1)):
            G_all_pairs_structure.add_node(i)
            for j in range(i+1, G_all_pairs.number_of_nodes()+1):
                similarity = comp_structure_dic(cluster_summary_df,i,j,dic_fp)
                if isinstance(similarity, (int, float, complex)):
                    if similarity> 0.7:
                        G_all_pairs_structure.add_edge(i,j)
                        G_all_pairs_structure[i][j]['stru_similarity'] = similarity
        similarities = []
        for u, v, data in G_all_pairs_structure.edges(data=True):
            similarity = data['stru_similarity']
            similarities.append(similarity)

        # Calculate the mean of the extracted similarities
        mean_similarity = sum(similarities) / len(similarities)

        print("Mean Similarity:", mean_similarity)
        print('structure graph with {} nodes and {} edges'.format(G_all_pairs_structure.number_of_nodes(), G_all_pairs_structure.number_of_edges()))





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
        G_all_pairs_structure = nx.Graph()
        error_num = 0
        final_results = []
        for sample_rate in range(5,10):
            for round in range(10):
                similarity_list=[]
                result_file_path = "./results/"+library+'/' + library +"_"+str(sample_rate*1.0*10)+"_" + str(round)+ "_classic_benchmark.graphml"
                G_sample = nx.read_graphml(result_file_path)
                for edge in G_sample.edges():
                    i = edge[0]
                    j = edge[1]
                    try:
                        similarity = comp_structure_dic(cluster_summary_df,i,j,dic_fp)
                        if isinstance(similarity, (int, float, complex)):
                                similarity_list.append(similarity)
                    except Exception as e:
                        error_num =  error_num + 1
                        print("Warning:", e)


                # Calculate the mean of the extracted similarities
                mean_similarity = sum(similarity_list) / len(similarity_list)
                print("Error number:",error_num)
                print("Mean Similarity:", mean_similarity)
                final_results.append(mean_similarity)
        file_path = "./results-classic-sample/" + library + "structure_similarity.pkl"
        with open(file_path, 'wb') as file:
            pickle.dump(final_results, file)







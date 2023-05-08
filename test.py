import pandas as pd
from gnpsdata import taskresult
from gnpsdata import workflow_classicnetworking
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



classic_networking_task = "9b79d3dd2322454da4b475fbb1081fbb"
# This is the Network Created by Classic Molecular Networking with the Default Layout, this includes all the structure infomration
classic_network_G = workflow_classicnetworking.get_graphml_network(classic_networking_task)
classic_network_G = nx.read_graphml("temp.graphml")

# Getting defautl classic networking data from scratch
# cluster_summary_df = taskresult.get_task_resultview_dataframe(classic_networking_task, "view_all_clusters_withID_beta")
cluster_summary_df = pd.read_csv("summary.tsv")

# classic_df_DB = taskresult.get_task_resultview_dataframe(classic_networking_task, "view_all_annotations_DB")

# Downloading all pairs from BareBones Networking, which calculates all pairs but does not do topology filtering
# all_pairs_task = "0b2207d0925c4568adabfcb063e8ca46"
# all_pairs_df = taskresult.get_task_resultview_dataframe(all_pairs_task, "view_results")
all_pairs_df = pd.read_csv("merged_pairs.tsv",sep='\t')
G_all_pairs =  nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine") #Generate network from all pairs data frame
G_all_pairs_realignment=nx.read_graphml("G_all_pairs_realignment.graphml")
dic_fp=fingerprint_dic_construct(cluster_summary_df)
# dic_fp_classic=fingerprint_dic_construct_networkx(classic_network_G)
components=filt_single_graph(G_all_pairs)
score_all_pair_original=[]
for component in tqdm(components):
    score_all_pair_original.append(subgraph_score_dic(component,classic_network_G,cluster_summary_df,dic_fp))
all_pairs_original_number = [len(x) for x in components]
df_all_pairs_original=pd.DataFrame(list(zip(score_all_pair_original,all_pairs_original_number)), columns=['score','number'])
print(weighted_average(df_all_pairs_original, 'score', 'number'))
# re_alignment(G_all_pairs)

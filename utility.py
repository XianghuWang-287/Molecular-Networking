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


# def subgraph_score(G):
#     score = 0
#     edge_num = G.number_of_edges()
#     error = 0
#     if (edge_num == 0):
#         return 1
#     for edge in G.edges():
#         try:
#             node1 = edge[0]
#             node2 = edge[1]
#             score += float(comp_structure(cluster_summary_df, node1, node2))
#         except Exception:
#             # print(node1, node2) #mute/active node error message
#             error = error + 1
#     if (edge_num - error == 0):
#         return 0
#     if (error):
#         return score / (edge_num - error)
#     else:
#         return score / edge_num
#
#
# def comp_structure(G, node1, node2):
#     if (isinstance(node1, str)):
#         smiles1 = classic_network_G.nodes[node1]["Smiles"]
#         smiles2 = classic_network_G.nodes[node2]["Smiles"]
#     else:
#         smiles1 = G.loc[G['SpectrumID'] == cluster_summary_df.loc[node1 - 1]['SpectrumID'], "Smiles"].values[0]
#         smiles2 = G.loc[G['SpectrumID'] == cluster_summary_df.loc[node2 - 1]['SpectrumID'], "Smiles"].values[0]
#     mol1 = Chem.MolFromSmiles(smiles1.replace('\\\\', '\\'))
#     if not mol1:
#         return {"message": "unable to import structure 1."}, 400
#     fp1 = FingerprintMol(mol1)
#
#     mol2 = Chem.MolFromSmiles(smiles2.replace('\\\\', '\\'))
#     if not mol2:
#         return {"message": "unable to import structure 2."}, 400
#     fp2 = FingerprintMol(mol2)
#     return FingerprintSimilarity(fp1, fp2)


def comp_structure_online(G, node1, node2):
    if (isinstance(node1, str)):
        smiles1 = urllib.parse.quote(G.nodes[node1]["Smiles"])
        smiles2 = urllib.parse.quote(G.nodes[node2]["Smiles"])
    else:
        smiles1 = urllib.parse.quote(G.loc[G['scan']==node1]["Smiles"].values[0])
        smiles2 = urllib.parse.quote(G.loc[G['scan']==node2]["Smiles"].values[0])
    compare_url = "https://gnps-structure.ucsd.edu/structuresimilarity?smiles1={}&smiles2={}".format(smiles1, smiles2)
    r = requests.get(compare_url)
    return r.text

def filt_single_graph(G):
    return [G.subgraph(c).copy() for c in nx.connected_components(G) if G.subgraph(c).number_of_nodes()>1]

def filt_single_cluster(cluster):
    return [c for c in cluster if c.number_of_nodes()>1]

def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()

def fingerprint_dic_construct(G):
    dic={}
    for index in tqdm(range(len(G))):
        try:
            if (G.nodes[str(index)]['Smiles'] == "N/A"):
                inchi = G.nodes[str(index)]['INCHI']
                mol = Chem.MolFromInchi(inchi)
                fp = FingerprintMol(mol)
                dic[inchi] = fp
                continue
            smiles=G.iloc[index]['Smiles']
            #print(smiles)
            mol = Chem.MolFromSmiles(smiles.replace('\\\\','\\'))
            fp=FingerprintMol(mol)
            dic[smiles]=fp
        except Exception:
            #print(index+1)
            continue
    return dic

def fingerprint_dic_construct_networkx(G):
    dic={}
    for node in tqdm(G.nodes()):
        try:
            if (G.nodes[str(node)]['Smiles'] == "N/A"):
                inchi = G.nodes[str(node)]['INCHI']
                mol = Chem.MolFromInchi(inchi)
                fp = FingerprintMol(mol)
                dic[inchi] = fp
                continue
            smiles=G.nodes[str(node)]['Smiles']
            mol = Chem.MolFromSmiles(smiles.replace('\\\\','\\'))
            fp=FingerprintMol(mol)
            dic[smiles]=fp
        except Exception:
            continue
    return dic

def subgraph_score_dic(G,df, dic_fp):
    score = 0
    edge_num = G.number_of_edges()
    error=0
    if(edge_num==0):
        return 1
    for edge in G.edges():
        try:
            node1 = edge[0]
            node2=  edge[1]
            score += float(comp_structure_dic(df,node1,node2, dic_fp))
        except Exception:
            #print(node1, node2) #mute/active node error message
            error=error+1
    if(edge_num-error==0):
        return 0
    if (error):
        return score/(edge_num-error)
    else:
        return score/edge_num

def comp_structure_dic(G, node1, node2, dic_fp):
    if(isinstance(node1,str)):
        smiles1 = G.nodes[node1]["Smiles"]
        smiles2 = G.nodes[node2]["Smiles"]
    elif (G.loc[G['scan']==node1]["Smiles"].values[0]!="N/A"):
        smiles1 =  G.loc[G['scan']==node1]["Smiles"].values[0]
        smiles2 =  G.loc[G['scan']==node2]["Smiles"].values[0]
    else:
        smiles1 =  G.loc[G['scan']==node1]["INCHI"].values[0]
        smiles2 =  G.loc[G['scan']==node2]["INCHI"].values[0]
    try:
        fp1=dic_fp[smiles1]
        fp2=dic_fp[smiles2]
        return FingerprintSimilarity(fp1,fp2)
    except Exception:
        return comp_structure_online(G, node1, node2)

def re_alignment(G):
    for node1, node2 in tqdm(nx.non_edges(G)):
        if nx.has_path(G,node1,node2):
            all_shortest_hops=[p for p in nx.all_shortest_paths(G,node1,node2,weight=None, method='dijkstra')]
            Max_path_weight=0
            Max_path=[]
            for hop in all_shortest_hops:
                Path_weight=0
                for node_i in range(len(hop)-1):
                    Path_weight=Path_weight+G[hop[node_i]][hop[node_i+1]]['Cosine']
                if (Path_weight>Max_path_weight):
                    Max_path_weight=Path_weight
                    Max_path=hop
            #print(Max_path_weight,Max_path)
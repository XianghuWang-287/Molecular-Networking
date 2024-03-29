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
import pickle

SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)
def _cosine_fast(
    spec: SpectrumTuple,
    spec_other: SpectrumTuple,
    fragment_mz_tolerance: float,
    allow_shift: bool,
) -> Tuple[float, List[Tuple[int, int]]]:
    precursor_charge = max(spec.precursor_charge, 1)
    precursor_mass_diff = (
        spec.precursor_mz - spec_other.precursor_mz
    ) * precursor_charge
    # Only take peak shifts into account if the mass difference is relevant.
    num_shifts = 1
    if allow_shift and abs(precursor_mass_diff) >= fragment_mz_tolerance:
        num_shifts += precursor_charge
    other_peak_index = np.zeros(num_shifts, np.uint16)
    mass_diff = np.zeros(num_shifts, np.float32)
    for charge in range(1, num_shifts):
        mass_diff[charge] = precursor_mass_diff / charge

    # Find the matching peaks between both spectra.
    peak_match_scores, peak_match_idx = [], []
    for peak_index, (peak_mz, peak_intensity) in enumerate(
        zip(spec.mz, spec.intensity)
    ):
        # Advance while there is an excessive mass difference.
        for cpi in range(num_shifts):
            while other_peak_index[cpi] < len(spec_other.mz) - 1 and (
                peak_mz - fragment_mz_tolerance
                > spec_other.mz[other_peak_index[cpi]] + mass_diff[cpi]
            ):
                other_peak_index[cpi] += 1
        # Match the peaks within the fragment mass window if possible.
        for cpi in range(num_shifts):
            index = 0
            other_peak_i = other_peak_index[cpi] + index
            while (
                other_peak_i < len(spec_other.mz)
                and abs(
                    peak_mz - (spec_other.mz[other_peak_i] + mass_diff[cpi])
                )
                <= fragment_mz_tolerance
            ):
                peak_match_scores.append(
                    peak_intensity * spec_other.intensity[other_peak_i]
                )
                peak_match_idx.append((peak_index, other_peak_i))
                index += 1
                other_peak_i = other_peak_index[cpi] + index

    score, peak_matches = 0.0, []
    if len(peak_match_scores) > 0:
        # Use the most prominent peak matches to compute the score (sort in
        # descending order).
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(peak_match_idx)[peak_match_order]
        peaks_used, other_peaks_used = set(), set()
        for peak_match_score, peak_i, other_peak_i in zip(
            peak_match_scores_arr,
            peak_match_idx_arr[:, 0],
            peak_match_idx_arr[:, 1],
        ):
            if (
                peak_i not in peaks_used
                and other_peak_i not in other_peaks_used
            ):
                score += peak_match_score
                # Save the matched peaks.
                peak_matches.append((peak_i, other_peak_i))
                # Make sure these peaks are not used anymore.
                peaks_used.add(peak_i)
                other_peaks_used.add(other_peak_i)

    return score, peak_matches

def peak_tuple_to_dic(peakmatches):
    dic={}
    for peakmatch in peakmatches:
        dic[peakmatch[0]]=peakmatch[1]
    return dic

def norm_intensity(intensity):
    return np.copy(intensity)/np.linalg.norm(intensity)
    #return(intensity)

def realign_path(path):
    final_match_list=[]
    spec_1 = spec_dic[path[0]]
    spec_2 = spec_dic[path[0+1]]
    score,peak_matches = _cosine_fast(spec_1,spec_2,0.5,True)
    final_match_list=peak_matches
    idx=1
    while (len(final_match_list)!=0 and idx <len(path)-1):
        temp_peakmatch_list=[]
        spec_1 = spec_dic[path[idx]]
        spec_2 = spec_dic[path[idx+1]]
        score,peak_matches = _cosine_fast(spec_1,spec_2,0.5,True)
        peak_dic1=peak_tuple_to_dic(final_match_list)
        peak_dic2=peak_tuple_to_dic(peak_matches)
        for key, value in peak_dic1.items():
            if (peak_dic2.get(value)):
                temp_peakmatch_list.append((key,peak_dic2[value]))
        final_match_list=temp_peakmatch_list
        idx=idx+1
    spec_start = spec_dic[path[0]]
    spec_end = spec_dic[path[-1]]
    _, matched_peaks = _cosine_fast(spec_start,spec_end,0.5,True)
    peak_match_scores = []
    intesity1=spec_dic[path[0]][3]
    intesity2=spec_dic[path[-1]][3]
    if (len(final_match_list)):
        final_match_list=final_match_list+matched_peaks
        for matched_peak in final_match_list:
            peak_match_scores.append(intesity1[matched_peak[0]]*intesity2[matched_peak[1]])
    score, peak_matches = 0.0, []
    if len(peak_match_scores) > 0:
        # Use the most prominent peak matches to compute the score (sort in
        # descending order).
        peak_match_scores_arr = np.asarray(peak_match_scores)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peak_match_scores_arr = peak_match_scores_arr[peak_match_order]
        peak_match_idx_arr = np.asarray(final_match_list)[peak_match_order]
        peaks_used, other_peaks_used = set(), set()
        for peak_match_score, peak_i, other_peak_i in zip(
            peak_match_scores_arr,
            peak_match_idx_arr[:, 0],
            peak_match_idx_arr[:, 1],
        ):
            if (
                peak_i not in peaks_used
                and other_peak_i not in other_peaks_used
            ):
                score += peak_match_score
                # Save the matched peaks.
                peak_matches.append((peak_i, other_peak_i))
                # Make sure these peaks are not used anymore.
                peaks_used.add(peak_i)
                other_peaks_used.add(other_peak_i)
        return peak_matches, score
    else:
        return "no match",_
def re_alignment_parallel(args):
    node1,node2=args
    if nx.has_path(G_all_pairs, node1, node2):
        all_shortest_hops = [p for p in nx.all_shortest_paths(G_all_pairs, node1, node2, weight=None, method='dijkstra')]
        Max_path_weight = 0
        Max_path = []
        for hop in all_shortest_hops:
            Path_weight = 0
            for node_i in range(len(hop) - 1):
                Path_weight = Path_weight + G_all_pairs[hop[node_i]][hop[node_i + 1]]['Cosine']
            if (Path_weight > Max_path_weight):
                Max_path_weight = Path_weight
                Max_path = hop
        matched_peaks,score = realign_path(Max_path)
        if matched_peaks != "no match":
            G_all_pairs_realignment.add_edge(node1,node2)
            G_all_pairs_realignment[node1][node2]['Cosine']=score
            return (node1,node2,score)
        else:
            return
    else:
        return


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
G_all_pairs_realignment=G_all_pairs.copy()
#dic_fp=fingerprint_dic_construct(cluster_summary_df)
# dic_fp_classic=fingerprint_dic_construct_networkx(classic_network_G)
components=filt_single_graph(G_all_pairs)
# score_all_pair_original=[]
# for component in tqdm(components):
#     score_all_pair_original.append(subgraph_score_dic(component,classic_network_G,cluster_summary_df,dic_fp))
# all_pairs_original_number = [len(x) for x in components]
# df_all_pairs_original=pd.DataFrame(list(zip(score_all_pair_original,all_pairs_original_number)), columns=['score','number'])
# print(weighted_average(df_all_pairs_original, 'score', 'number'))
# re_alignment(G_all_pairs)
if __name__ == '__main__':
    # create a pool of processes
    spec_dic = {}
    print("Start create spectrum dictionary")
    for spectrum in tqdm(mgf.read("converted.mgf")):
        params = spectrum.get('params')
        precursor_mz = cluster_summary_df.loc[int(params['scans'])-1]["Precursor_MZ"]
        charge = cluster_summary_df.loc[int(params['scans'])-1]["Charge"]
        mz = spectrum.get('m/z array')
        intensity = spectrum.get('intensity array')
        spec_dic[int(params['scans'])] = SpectrumTuple(precursor_mz,charge,mz,norm_intensity(intensity))


    with Pool(processes=30,maxtasksperchild=1000) as pool:
        # define the range of values you want to loop over
        values = [[node1,node2] for [node1,node2] in nx.non_edges(G_all_pairs)]
        # apply the function to each value in the loop using imap_unordered
        results = list(tqdm(pool.imap(re_alignment_parallel, values), total=len(values)))
        # print the results
    with open('re_alignment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    nx.write_graphml(G_all_pairs_realignment,"G_all_pairs_realignment.graphml")
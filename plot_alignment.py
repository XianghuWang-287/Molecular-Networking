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
from matplotlib.patches import ConnectionPatch

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

def TupleToPair(tuple):
    peaks = []
    for i in range(len(tuple.mz)):
        peaks.append((tuple.mz[i], tuple.intensity[i]))
    return peaks


def draw_alignment(peaks1, peaks2, matched_peaks, shift=0.1, show_text=False, show_lines=True, scale=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20 * scale, 10 * scale))

    shifted, unshifted = separateShifted(matched_peaks, peaks1, peaks2, 0.5)
    print("shifted", shifted)
    print("unshifted", unshifted)
    max_y = max([_[1] for _ in peaks1])
    peaks1 = [(peaks1[i][0], peaks1[i][1] / max_y) for i in range(len(peaks1))]

    max_y = max([_[1] for _ in peaks2])
    peaks2 = [(peaks2[i][0], peaks2[i][1] / max_y) for i in range(len(peaks2))]

    max_x = max([_[0] for _ in peaks1])
    max_x = max([max_x, max([_[0] for _ in peaks2])])

    ax.set_xlim(0, max_x + 5)

    # print('debugging', peaks2)

    molShifted = [i[0] for i in shifted]
    molUnshifted = [i[0] for i in unshifted]
    modifiedShifted = [i[1] for i in shifted]
    modifiedUnshifted = [i[1] for i in unshifted]

    # draw peaks in site locator main compound
    # make shifted peaks red
    # make unshifted peaks blue

    x = [_[0] for _ in peaks1]
    y = [_[1] for _ in peaks1]
    shift_array = [shift for _ in peaks1]
    ax.bar(x, shift_array, width=0.4 * scale, color="white", alpha=0)
    ax.bar(x, y, width=0.4 * scale, color="gray", bottom=shift_array)

    x_shifted = [peaks1[_][0] for _ in molShifted]
    print("top_shifted", x_shifted)
    y_shifted = [peaks1[_][1] for _ in molShifted]
    shift_array = [shift for _ in molShifted]
    ax.bar(x_shifted, shift_array, width=0.4 * scale, color="white", alpha=0)
    ax.bar(x_shifted, y_shifted, width=0.4 * scale, color="red", bottom=shift_array)

    x_unshifted = [peaks1[_][0] for _ in molUnshifted]
    y_unshifted = [peaks1[_][1] for _ in molUnshifted]
    print("top_unshifted", x_unshifted)
    shift_array = [shift for _ in molUnshifted]
    ax.bar(x_unshifted, shift_array, width=0.4 * scale, color="white", alpha=0)
    ax.bar(x_unshifted, y_unshifted, width=0.4 * scale, color="blue", bottom=shift_array)

    # plot modified peaks as reversed
    x = [_[0] for _ in peaks2]
    y = [-_[1] for _ in peaks2]
    ax.bar(x, y, width=0.4 * scale, color="gray")

    x_shifted = [peaks2[_][0] for _ in modifiedShifted]
    print("top_shifted", x_shifted)
    y_shifted = [-peaks2[_][1] for _ in modifiedShifted]
    ax.bar(x_shifted, y_shifted, width=0.4 * scale, color="red")

    x_unshifted = [peaks2[_][0] for _ in modifiedUnshifted]
    y_unshifted = [-peaks2[_][1] for _ in modifiedUnshifted]
    print("bottom_unshifted", x_unshifted)
    ax.bar(x_unshifted, y_unshifted, width=0.4 * scale, color="blue")

    if show_lines:
        for peak in shifted:
            x1 = peaks1[peak[0]][0]
            x2 = peaks2[peak[1]][0]
            # draw line with text in the middle
            ax.plot([x1, x2], [shift, 0], color="red", linewidth=0.2 * scale, linestyle="--")
            val = abs(peaks1[peak[0]][0] - peaks2[peak[1]][0])
            val = round(val, 2)
            if show_text:
                ax.text((x1 + x2) / 2, shift / 2, str(val), fontsize=10, horizontalalignment='center')

        for peak in unshifted:
            x1 = peaks1[peak[0]][0]
            x2 = peaks2[peak[1]][0]
            ax.plot([x1, x2], [shift, 0], color="blue", linewidth=0.2 * scale, linestyle="--")

    # draw horizontal line
    ax.plot([5, max_x], [shift, shift], color="gray", linewidth=0.4 * scale, linestyle="-")
    ax.plot([5, max_x], [0, 0], color="gray", linewidth=0.4 * scale, linestyle="-")

    # custom y axis ticks
    y_ticks1 = [i / 10 + shift for i in range(0, 11, 2)]
    y_ticks2 = [-i / 10 for i in range(0, 11, 2)]
    # reverse y ticks2
    y_ticks2 = y_ticks2[::-1]
    y_ticks = y_ticks2 + y_ticks1
    print(y_ticks)
    y_ticks_labels1 = [i for i in range(0, 110, 20)]
    y_ticks_labels2 = [i for i in range(0, 110, 20)]
    # reverse y ticks2
    y_ticks_labels2 = y_ticks_labels2[::-1]
    y_ticks_labels = y_ticks_labels2 + y_ticks_labels1
    print(y_ticks_labels)
    ax.set_yticks(y_ticks, y_ticks_labels)

    # set font size
    ax.tick_params(axis='both', which='major', labelsize=20 * scale)

    # add legend
    ax.plot([], [], color="red", linewidth=2 * scale, linestyle="-", label="Shifted Matched Peaks")
    ax.plot([], [], color="blue", linewidth=2 * scale, linestyle="-", label="Unshifted Matched Peaks")
    ax.plot([], [], color="gray", linewidth=2 * scale, linestyle="-", label="Unmatched Peaks")
    ax.legend(loc='upper left')
    # legend font size
    ax.legend(prop={'size': 20 * scale})

    plt.show()
    return ax


def separateShifted(matchedPeaks, mol1peaks, mol2peaks, eps = 0.1):
    """
    Separates the shifted and unshifted peaks.
    """
    shifted = []
    unshifted = []
    for peak in matchedPeaks:
        if abs(mol1peaks[peak[0]][0] - mol2peaks[peak[1]][0]) > eps:
            shifted.append(peak)
        else:
            unshifted.append(peak)
    return shifted, unshifted


if __name__ == '__main__':
    #pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('--input', type=str,required=True,default="input_library.txt", help='input libraries')
    args = parser.parse_args()
    input_lib_file = args.input
    #read libraries from input file
    with open(input_lib_file,'r') as f:
        libraries = f.readlines()

    temp_count = 0
    for library in libraries:
        library = library.strip('\n')
        print("starting align library:"+library)
        summary_file_path = "./data/summary/"+library+"_summary.tsv"
        merged_pairs_file_path = "./data/merged_paris/"+library+"_merged_pairs.tsv"
        mgf_file_path = "./data/converted/"+library+"_converted.mgf"
        cluster_summary_df = pd.read_csv(summary_file_path)
        all_pairs_df = pd.read_csv(merged_pairs_file_path, sep='\t')
        G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
        G_all_pairs_realignment = G_all_pairs.copy()
        spec_dic = {}
        print("Start create spectrum dictionary")
        for spectrum in tqdm(mgf.read(mgf_file_path)):
            params = spectrum.get('params')
            precursor_mz = cluster_summary_df.loc[int(params['scans']) - 1]["Precursor_MZ"]
            charge = cluster_summary_df.loc[int(params['scans']) - 1]["Charge"]
            mz_array = spectrum.get('m/z array')
            intensity_array = spectrum.get('intensity array')
            filtered_mz = []
            filtered_intensities = []
            precursor_value = float(cluster_summary_df.loc[cluster_summary_df['scan'] == int(params['scans'])]["Precursor_MZ"].values[0])
            for i, mz in enumerate(mz_array):
                peak_range = [j for j in range(len(mz_array)) if abs(mz_array[j] - mz) <= 25]
                sorted_range = sorted(peak_range, key=lambda j: intensity_array[j], reverse=True)
                if i in sorted_range[:6]:
                    if abs(mz - precursor_value) > 17:
                        filtered_mz.append(mz)
                        filtered_intensities.append(intensity_array[i])
            filtered_intensities = [math.sqrt(x) for x in filtered_intensities]
            spec_dic[int(params['scans'])] = SpectrumTuple(precursor_value, charge, filtered_mz, norm_intensity(filtered_intensities))

            temp_count+=1
            if temp_count > 2:
                break
        spectrum_A = spec_dic[1]
        spectrum_B = spec_dic[2]
        spectrum_C = spec_dic[3]
        _,matched_peaks_A_B = _cosine_fast(spectrum_A,spectrum_B,0.5,True)
        _,matched_peaks_B_C = _cosine_fast(spectrum_B,spectrum_C,0.5,True)
        print(matched_peaks_A_B)
        print(matched_peaks_B_C)

        peaks1= TupleToPair(spec_dic[1])
        peaks2 = TupleToPair(spec_dic[2])
        peaks3 = TupleToPair(spec_dic[3])

        shiftedAB, unshiftedAB = separateShifted(matched_peaks_A_B,peaks1,peaks2,0.5)
        shiftedBC, unshiftedBC = separateShifted(matched_peaks_B_C, peaks2, peaks3, 0.5)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        ax1.bar(spectrum_A.mz, spectrum_A.intensity, label='Spectrum A', linestyle="-",width=2)
        ax1.set_ylabel('Intensity')
        ax1.legend()


        ax2.bar(spectrum_B.mz, spectrum_B.intensity, label='Spectrum B', linestyle="-",width=2)
        ax2.set_ylabel('Intensity')
        ax2.legend()


        ax3.bar(spectrum_C.mz, spectrum_C.intensity, label='Spectrum C', linestyle="-",width=2)
        ax3.set_xlabel('m/z')
        ax3.set_ylabel('Intensity')
        ax3.legend()


        print(shiftedAB)
        for (idx_A, idx_B) in shiftedAB:
            xyA = (spectrum_A.mz[idx_A], 0)
            xyB = (spectrum_B.mz[idx_B], 0)
            coordsA = "data"
            coordsB = "data"
            con_AB = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB, axesA=ax2, axesB=ax1,
                                     color='red',linestyle='--')
            ax2.add_artist(con_AB)



        for (idx_B, idx_C) in shiftedBC:
            xyB = (spectrum_B.mz[idx_B], 0)
            xyC = (spectrum_C.mz[idx_C], 0)
            coordsA = "data"
            coordsB = "data"
            con_BC = ConnectionPatch(xyA=xyB, xyB=xyC, coordsA=coordsA, coordsB=coordsB, axesA=ax3, axesB=ax2,
                                     color='green',linestyle='--')
            ax3.add_artist(con_BC)


        plt.show()
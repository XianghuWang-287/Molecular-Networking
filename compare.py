from pyteomics import mgf
# from Classic import *
from tqdm import tqdm
import numpy as np
import networkx as nx
from multiprocessing import Pool
import collections
from typing import List, Tuple
import csv
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-m', type=str, required=False, default="trans_align_result.tsv", help='raw pairs filename')

    args = parser.parse_args()
    raw_pairs_filename = args.m


    # read the raw pairs file
    all_pairs_df = pd.read_csv(raw_pairs_filename, sep='\t')
    # constructed network from edge list
    G_all_pairs = nx.from_pandas_edgelist(all_pairs_df, "CLUSTERID1", "CLUSTERID2", "Cosine")
    print(G_all_pairs.number_of_nodes())
    components = [G_all_pairs.subgraph(c).copy() for c in nx.connected_components(G_all_pairs)]
    components_num = len(components)
    plt.figure(figsize=(18, 100))
    for component in components:
        plt.subplot(components_num // 8 + 1, 8, components.index(component) + 1)
        nx.draw(component, node_size=2)
    plt.show()
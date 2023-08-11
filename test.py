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

from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint

# Load molecules
mol1 = Chem.MolFromSmiles('CCO')  # Replace with your SMILES strings or molecule files
mol2 = Chem.MolFromSmiles('CCC')

# Generate RDKFingerprints
# fp1 = RDKFingerprint(mol1)
# fp2 = RDKFingerprint(mol2)
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

# Calculate different similarity metrics
tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
dice = DataStructs.DiceSimilarity(fp1, fp2)
cosine = DataStructs.CosineSimilarity(fp1, fp2)
sokal = DataStructs.SokalSimilarity(fp1, fp2)
russel = DataStructs.RusselSimilarity(fp1, fp2)
kulczynski = DataStructs.KulczynskiSimilarity(fp1, fp2)
mcconnaughey = DataStructs.McConnaugheySimilarity(fp1, fp2)


# Print similarity values
print("Tanimoto Similarity:", tanimoto)
print("Dice Similarity:", dice)
print("Cosine Similarity:", cosine)
print("Sokal Similarity:", sokal)
print("Russel Similarity:", russel)
print("Kulczynski Similarity:", kulczynski)
print("McConnaughey Similarity:", mcconnaughey)


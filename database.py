from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import re


def initialize_dataframe(molecule_list):
    """
    Constructs a pandas DataFrame from a list of molecules (defined by RDKit)
    :param molecule_list: List of RDKit defined molecules
    :return: nmr_df: Dataframe of all molecular properties
    """
    nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in molecule_list if x is not None])
    nmr_df['Name'] = [x.GetProp('_Name') for x in molecule_list if x is not None]
    nmr_df['Smiles'] = [Chem.MolToSmiles(x) for x in molecule_list if x is not None]
    return nmr_df


def import_nmrshiftdb2_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    return [x for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd')]


def create_proton_carbon_spectra(nmr_df):
    """
    Isolates the first 1H and 13C spectrum available from all spectra
    :param nmr_df: Dataframe of all molecular properties
    :return: nmr_df:  Dataframe of all molecular properties and two clean 1H and 13C spectra
    """
    nmr_df['Spectrum 13C'] = nmr_df.filter(regex='Spectrum 13C.*').values.tolist()
    nmr_df['Spectrum 1H'] = nmr_df.filter(regex='Spectrum 1H.*').values.tolist()
    clean_13C_list = []
    clean_1H_list = []
    for mol_13C in nmr_df['Spectrum 13C'].values:
        clean_13C_list.append(next(filter(None, mol_13C), np.nan))
    for mol_1H in nmr_df['Spectrum 1H'].values:
        clean_1H_list.append(next(filter(None, mol_1H), np.nan))
    nmr_df['Spectrum 13C'] = clean_13C_list
    nmr_df['Spectrum 1H'] = clean_1H_list
    return nmr_df


def trim_dataframe_no_two_spectra(nmr_df):
    """
    Keeps only the records that have both 1H and 13C spectra
    :param nmr_df:
    :return: dataframe containing records that have both 1H and 13C spectra
    """
    return nmr_df[~(pd.isna(nmr_df['Spectrum 13C'])) & ~(pd.isna(nmr_df['Spectrum 1H']))]


def drop_unnecessary_columns(nmr_df):
    """
    Keeps only the relevant columns of the dataframe (Name, SMILES, 1H and 13C spectra)
    :param nmr_df:
    :return:
    """
    return nmr_df.iloc[:, -4:]


def get_morgan_fingerprints(mol_name):
    """
    Get Bit Array of Morgan molecular fingerprint
    :param mol_name:
    :return:
    """
    return AllChem.GetMorganFingerprintAsBitVect(mol_name, 2).ToBitString()


def simplify_spectra(nmr_df):
    """
    Convert the string format of the spectra into manageable list
    :param nmr_df:
    :return:
    """
    nmr_df['Spectrum 1H'] = [re.findall(r'^\d+\.?\d*|\|\d+\.?\d*', x) for x in nmr_df['Spectrum 1H']]
    nmr_df['Spectrum 1H'] = [list(map(lambda y: re.sub(r'\|', '', y), x)) for x in nmr_df['Spectrum 1H']]
    nmr_df['Spectrum 13C'] = [re.findall(r'^\d+\.?\d*|\|\d+\.?\d*', x) for x in nmr_df['Spectrum 13C']]
    nmr_df['Spectrum 13C'] = [list(map(lambda y: re.sub(r'\|', '', y), x)) for x in nmr_df['Spectrum 13C']]
    return nmr_df


def import_database_as_df():
    nmr_df = initialize_dataframe(import_nmrshiftdb2_database())
    nmr_df = create_proton_carbon_spectra(nmr_df)
    nmr_df = trim_dataframe_no_two_spectra(nmr_df)
    nmr_df = drop_unnecessary_columns(nmr_df)
    nmr_df = simplify_spectra(nmr_df)
    return nmr_df

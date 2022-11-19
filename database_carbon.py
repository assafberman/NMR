from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit import RDLogger
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
    nmr_df['Morgan'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=512) for x in molecule_list if x is not None]
    return nmr_df


def import_nmrshiftdb2_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    return [x for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd')]


def create_proton_carbon_spectra(nmr_df):
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C 0']
    return nmr_df


def create_proton_carbon_spectra_2(nmr_df):
    """
    Isolates the first 1H and 13C spectrum available from all spectra
    :param nmr_df: Dataframe of all molecular properties
    :return: nmr_df:  Dataframe of all molecular properties and two clean 1H and 13C spectra
    """
    nmr_df['Spectrum 13C'] = nmr_df.filter(like='Spectrum 13C').values.tolist()
    print('13C:', nmr_df['Spectrum 13C'].count())
    print('1H:', nmr_df['Spectrum 1H'].count())
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(lambda x: next(filter(None, x), np.nan))
    nmr_df['Spectrum 1H'] = nmr_df['Spectrum 1H'].apply(lambda x: next(filter(None, x), np.nan))
    print('13C:', nmr_df['Spectrum 13C'].count())
    print('1H:', nmr_df['Spectrum 1H'].count())
    return nmr_df


def trim_dataframe_no_two_spectra(nmr_df):
    """
    Keeps only the records that have both 1H and 13C spectra
    :param nmr_df:
    :return: dataframe containing records that have both 1H and 13C spectra
    """
    return nmr_df[~(pd.isna(nmr_df['Spectrum 13C']))]


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
    nmr_df['Spectrum 13C'] = [re.findall(r'^\d+\.?\d*|\|\d+\.?\d*|[a-zA-Z]+', x) for x in nmr_df['Spectrum 13C']]
    nmr_df['Spectrum 13C'] = [list(map(lambda y: re.sub(r'\|', '', y), x)) for x in nmr_df['Spectrum 13C']]
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_shift_multiplicity_association)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_correct_tuples)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_frequency_list)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_num_multiplicity)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(pad_spectrum)
    nmr_df['Input'] = nmr_df.apply(lambda x: list(x['Spectrum 13C']), axis=1)
    return nmr_df


def aux_correct_tuples(spec_list):
    corrected_list = []
    for item in spec_list:
        if isinstance(item, tuple):
            if aux_is_number(item[0]) and item[1].isalpha():
                corrected_list.append(item)
    return corrected_list


def aux_num_multiplicity(spec_list):
    replace_dict = {'S': 1, 's': 1, 'D': 2, 'd': 2, 'T': 3, 't': 3, 'Q': 4, 'q': 4, 'CH': 5, 'u': 6, 'p': 7}
    return [replace_dict[x] if str(x).isalpha() else x for x in spec_list]


def aux_frequency_list(spec_list):
    freq = {}
    freq_list = []
    for item in spec_list:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    for key, val in freq.items():
        if isinstance(key, str):
            freq_list.append(float(key))
        else:
            freq_list.append(float(key[0]))
            freq_list.append(key[1])
        freq_list.append(val)
    return freq_list


def aux_shift_multiplicity_association(spec_list):
    associated_list = []
    for item in zip(spec_list[::2], spec_list[1::2]):
        associated_list.append(item)
    return associated_list


def aux_is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def pad_spectrum(spec_list, size=256):
    lst = np.array(spec_list, dtype=float)
    lst = np.concatenate([lst, np.zeros(size - len(spec_list))], dtype=float)
    return lst


def concat_spectra(spec_list1, spec_list2):
    return spec_list1 + spec_list2


def import_database_as_df():
    RDLogger.DisableLog('rdApp.*')
    nmr_df = initialize_dataframe(import_nmrshiftdb2_database())
    nmr_df = create_proton_carbon_spectra(nmr_df)
    print('Before trim:', nmr_df.shape)
    nmr_df = trim_dataframe_no_two_spectra(nmr_df)
    print('After trim:', nmr_df.shape)
    nmr_df = drop_unnecessary_columns(nmr_df)
    nmr_df = simplify_spectra(nmr_df)
    return nmr_df

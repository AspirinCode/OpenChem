# TODO: packed variable length sequence

import numpy as np
import pickle

from torch.utils.data import Dataset

from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import sanitize_smiles, pad_sequences, seq2tensor
from openchem.data.utils import get_tokens


class SmilesProteinDataset(Dataset):
    def __init__(self, filename, tokenized=False, cols_to_read=None,
                 delimiter=',', mol_tokens=None, prot_tokens=None, pad=True):
        super(SmilesProteinDataset, self).__init__()
        if not tokenized:
            data = read_smiles_property_file(filename, cols_to_read, delimiter)
            smiles = data[0]
            proteins = np.array(data[1])
            target = np.array(data[2], dtype='float')
            clean_smiles, clean_idx = sanitize_smiles(smiles)
            self.target = target[clean_idx]
            proteins = list(proteins[clean_idx])
            if pad:
                clean_smiles, self.mol_lengths = pad_sequences(clean_smiles)
                proteins, self.prot_lengths = pad_sequences(proteins)
            mol_tokens, _, _ = get_tokens(clean_smiles, mol_tokens)
            prot_tokens, _, _ = get_tokens(proteins, prot_tokens)
            clean_smiles, self.mol_tokens = seq2tensor(clean_smiles, mol_tokens)
            proteins, self.prot_tokens = seq2tensor(proteins, prot_tokens)
            self.mol_num_tokens = len(self.mol_tokens)
            self.prot_num_tokens = len(self.prot_tokens)
            self.molecules = clean_smiles
            self.proteins = proteins
        else:
            f = open(filename, 'rb')
            data = pickle.load(f)
            self.mol_tokens = data['smiles_tokens']
            self.prot_tokens = data['proteins_tokens']
            self.mol_num_tokens = len(data['smiles_tokens'])
            self.prot_num_tokens = len(data['proteins_tokens'])
            self.molecules = data['smiles']
            self.proteins = data['proteins']
            self.target = data['labels'] 
        assert len(self.molecules) == len(self.proteins)
        assert len(self.molecules) == len(self.target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'tokenized_smiles': self.molecules[index],
                  'tokenized_protein': self.proteins[index],
                  'labels': self.target[index],
                  'mol_length': self.mol_lengths[index],
                  'prot_length': self.prot_lengths[index]}
        return sample

import sys
sys.path.append("../")
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
import pandas as pd
import itertools


# Training data provided by Yoseph's lab: https://drive.google.com/file/d/1lN6EhcfB8Bot-rFMTID1EC_rKCVJ7n0M/view (SENT IN BASECAMP)
# To pre-train the model, dowload 'sequences_training_data_gencode.tsv' from 
#     https://drive.google.com/file/d/1lN6EhcfB8Bot-rFMTID1EC_rKCVJ7n0M/view 
#     and modify PATH_TO_TRAINING_DATA to point to correct path to correct local path to sequences_training_data_gencode.tsv
# To run BOS, this pretraining is not needed as the pre-trained model weights are provided 
PATH_TO_TRAINING_DATA = 'sequences_training_data_gencode.tsv'

class DataModuleKmers(pl.LightningDataModule):
    def __init__(self, batch_size=128, k=3, load_train_data=False, test_seqs_list=None, load_test_data=False, ): 
        super().__init__() 
        self.batch_size = batch_size 
        self.train  = DatasetKmers(dataset='train', k=k, load_data=load_train_data) 
        self.test   = DatasetKmers(
            dataset='test', 
            k=k, 
            seqs_list=test_seqs_list,
            vocab=self.train.vocab, 
            vocab2idx=self.train.vocab2idx, 
            load_data=load_test_data,
        )
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)


class DatasetKmers(Dataset): 
    def __init__(self, dataset='train', seqs_list=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        if seqs_list is None:
            if load_data and (dataset=="train"): 
                df = pd.read_csv(PATH_TO_TRAINING_DATA, sep='\t')
                seqs_list = df['seq'].values  # (794872,), confirmed (794872,) 
            else:
                seqs_list = ['A', 'C', 'T', 'G', 'N']

        self.dataset = dataset
        self.k = k
        regular_data = [] 
        for seq in seqs_list: 
            regular_data.append([token for token in seq]) # list of tokens
        
        # first get initial vocab set 
        if vocab is None:
            self.regular_vocab = set((token for seq in regular_data for token in seq))
            self.regular_vocab.discard(".") 
            if '-' not in self.regular_vocab: 
                self.regular_vocab.add('-') 
            self.vocab = ["".join(kmer) for kmer in itertools.product(self.regular_vocab, repeat=k)] 
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))] 
        else: 
            self.vocab = vocab 

        if vocab2idx is None:
            self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }
        else:
            self.vocab2idx = vocab2idx
        
        self.data = []
        if load_data:
            for seq in regular_data:
                token_num = 0
                kmer_tokens = []
                while token_num < len(seq):
                    kmer = seq[token_num:token_num+k]
                    while len(kmer) < k:
                        kmer += '-' # padd so we always have length k 
                    kmer_tokens.append("".join(kmer)) 
                    token_num += k 
                self.data.append(kmer_tokens) 
        


    def tokenize_sequences(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num + self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k  
                if type(kmer) == list: kmer = "".join(kmer)
                kmer_tokens.append(kmer) 
                token_num += self.k 
            tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 


    def encode(self, tokenized_sequence):
        return torch.tensor([self.vocab2idx[s] for s in [*tokenized_sequence, '<stop>']])

    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given seq (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded sequence string (ie ACTGGCATT...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        sequence = dec[0:stop] # cut off stop tokens
        while "<start>" in sequence: # start at last start token 
            start = (1+dec.index("<start>")) 
            sequence = sequence[start:]
        sequence = "".join(sequence) # combine into single string 
        sequence = sequence.replace("N", "T") # Only want ACTG, replace N's with T
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    @property
    def vocab_size(self):
        return len(self.vocab)


def collate_fn(data):
    # Length of longest string in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )

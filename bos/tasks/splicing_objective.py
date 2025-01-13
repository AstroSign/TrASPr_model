import torch 
import numpy as np 
import random 
import sys 
sys.path.append("../")
ORACLE_IMPORT_SUCCESSFUL = True 
try: 
    from tasks.oracle import DNABertPredictor
except:
    ORACLE_IMPORT_SUCCESSFUL = False 
    print("\n\n\n FAILED TO LOAD ORACLE, USING DUMMY RANDOM ORACLE FOR TESTING, RESULTS WILL BE INVALID \n\n\n")
from utils.vae.data import DataModuleKmers, collate_fn
from utils.vae.transformer_vae_unbounded import InfoTransformerVAE
from utils.seq_utils import convert_seqs_list_to_quads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SplicingObjective:
    def __init__(
        self,
        tissue="brain",
        minimize_psi=False,
        dim=256*4,
        num_calls=0,
        dtype=torch.float32,
        lb=None, # None is important for changing vae w/ lolbo... 
        ub=None, # None is important for changing vae w/ lolbo... 
        path_to_vae_statedict="../utils/vae/saved_model_weights/hearty-gorge-28_model_state.pkl",
        max_string_length=1024,
        oracle_model_weights_path="../tasks/best_gtex_checkpoint",
        **kwargs,
    ):
        # if minimize_psi is True, we seek to decrease psi, 
        #   otherwise, the goal is to increase psi (maximize psi)
        self.minimize_psi = minimize_psi
        # target tissue for which we want to increase (or decrease) psi 
        self.tissue = tissue 
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim 
        self.dim = dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        if ORACLE_IMPORT_SUCCESSFUL and (oracle_model_weights_path is not None):
            self.bert = DNABertPredictor(model_path=oracle_model_weights_path)
        else:
            # use dummy oracle instead just to test that the repo env is set up properly 
            self.bert = None 
        self.initialize_vae() 

    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), 
            each item in enumerable type must be a float tensor of shape (dim,) 
            (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving objective value obtained by passing each x in xs into f(x).
        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        assert xs.shape[-1] == self.dim 
        xs = xs.to(device) # torch.Size([bsz, 256*4])
        n_inputs = len(xs)
        seqs_list = self.vae_decode(z=xs) # length bsz, list of lists of four splice site sequences 
        four_seqs_list = convert_seqs_list_to_quads(seqs_list) # from [s1, s2, ..., sn] --> [[s1,s2,s3,s4],[s5,s6,s7,s8],...]
        ys = self.four_seqs_list_to_scores(four_seqs_list)
        assert ys.shape[0] == n_inputs
        self.num_calls += n_inputs
        return_dict = {
            "ys":ys,
            "strings":four_seqs_list,
        }
        return return_dict

    def dummy_oracle(self, seqs, tissue):
        ''' Optimize the number of G's in the sequence (rather than psi) 
            for testing code/env setup without oracle 
        ''' 
        assert type(tissue) == str 
        assert type(seqs) == list 
        ys = []
        for seq_quad in seqs:
            assert type(seq_quad) == list 
            assert len(seq_quad) == 4
            n_gs = 0 
            len_seq = 0 
            for seq in seq_quad:
                assert type(seq) == str 
                for char in seq:
                    char in ['A', 'C', 'T', 'G', '-']
                    if char == "G":
                        n_gs += 1 
                    if char != "-":
                        len_seq += 1 
            normed_n_gs = n_gs/len_seq
            ys.append(normed_n_gs)
        assert len(ys) == len(seqs)
        return ys 

    def four_seqs_list_to_scores(self, four_seqs_list):
        if self.bert is None:
            ys = self.dummy_oracle(seqs=four_seqs_list, tissue=self.tissue)
        else:
            ys = self.bert.predict(seqs=four_seqs_list, tissue=self.tissue) # list of length bsz of psis 
        ys = torch.tensor(ys).unsqueeze(-1).to(dtype=self.dtype) # (bsz,1) tensor of scores 
        if self.minimize_psi:
            ys = ys*-1 # want to minimize psi, turn minimization problem into maximization problem by using *-1
        return ys 

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        # create data module object 
        data_module = DataModuleKmers() 
        self.dataobj = data_module.train 
        # initialize vae 
        self.vae = InfoTransformerVAE(dataset=self.dataobj).to(device)
        # load state dict 
        state_dict = torch.load(self.path_to_vae_statedict, map_location=torch.device(device)) 
        self.vae.load_state_dict(state_dict, strict=True) 
        # set max string length 
        self.vae.max_string_length = self.max_string_length 
        # put model in eval mode 
        self.vae.eval()


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).to(dtype=self.dtype)
        self.vae.eval()
        self.vae.to(device)
        # sample strings form VAE decoder
        with torch.no_grad():
            sample = self.vae.sample(z=z.to(device).reshape(-1, 2, 128))
        # grab decoded strings
        decoded_aa_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        return decoded_aa_seqs


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        if type(xs_batch[0]) == list:
            # each item in xs_batch is a list of four splice site sequences 
            xs_as_individual_seqs = []
            for quad in xs_batch:
                for seq in quad:
                    xs_as_individual_seqs.append(seq)
            assert len(xs_as_individual_seqs) == len(xs_batch)*4
            xs_batch = xs_as_individual_seqs
        assert type(xs_batch[0]) == str 
        tokenized_seqs = self.dataobj.tokenize_sequences(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)
        out_dict = self.vae(X.to(device))
        vae_loss, z = out_dict['loss'], out_dict['z']
        z = z.reshape(-1,self.dim)
        return z, vae_loss


if __name__ == "__main__":
    obj = SplicingObjective()
    x = torch.randn(3, obj.dim).to(dtype=obj.dtype)
    out_dict = obj(x)
    y = out_dict["ys"]
    print(y.shape)
    print(f"y: {y}") 
    sequences_out = out_dict["strings"]
    print("example seq out:", sequences_out[0])


import torch 
import copy 
import random 
import math 

ALL_BASES = ['A', 'C', 'T', 'G']

def get_mutated_seqs(starter_four_seqs_list, n_seqs_total=10_000, max_n_edits=10,):
    # get randomly mutated versions of starter sequence 
    mutated_seqs = []
    num_edits = []
    for _ in range(n_seqs_total):
        new_four_seqs_list = copy.deepcopy(starter_four_seqs_list)
        # make a random number of edits between 1 and max_n_edits
        n_edits_single_seq = random.randint(1,max_n_edits) 
        num_edits.append(n_edits_single_seq)
        for _ in range(n_edits_single_seq):
            # edit in a random one of the four splice-site seqs 
            edit_quad = random.randint(0,3) 
            # edit in a random location in that splice-site seq 
            edit_loc = random.randint(0,len(starter_four_seqs_list[edit_quad]) - 1) 
            new_seq = copy.deepcopy(new_four_seqs_list[edit_quad])
            base_new_seq = [char for char in new_seq]
            # make the edit by swapping in a random new base 
            base_new_seq[edit_loc] = random.choice(ALL_BASES)
            new_seq = "".join(base_new_seq) 
            new_four_seqs_list[edit_quad] = new_seq
        mutated_seqs.append(new_four_seqs_list)

    return mutated_seqs, num_edits


def get_random_init_data(
    num_initialization_points, 
    objective, 
    starter_sequence, 
    edit_distance_threshold,
    dtype,
    vae_forward_bsz=64,
):
    ''' randomly initialize num_initialization_points
        total initial data points to kick-off optimization 
        by making random edits to the starter sequences 
    '''
    # use random mutations of starter as init data 
    init_train_strings, num_edits = get_mutated_seqs(
        starter_four_seqs_list=starter_sequence, 
        n_seqs_total=num_initialization_points-1, 
        max_n_edits=edit_distance_threshold,
    )
    # also include starter in init data 
    init_train_strings.append(starter_sequence)
    num_edits.append(0)
    # use number of edits to train constraint surrogate model 
    init_train_c = torch.tensor(num_edits).unsqueeze(-1).to(dtype=dtype)
    # encode train strings w/ vae to get train_x (latent space points for init)
    num_batches = math.ceil(len(init_train_strings) / vae_forward_bsz)
    with torch.no_grad():
        init_train_x = []
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*vae_forward_bsz, (batch_ix+1)*vae_forward_bsz
            train_strings_batch = init_train_strings[start_idx:stop_idx]
            train_x_batch, _ = objective.vae_forward(train_strings_batch) 
            init_train_x.append(train_x_batch.detach().cpu())
        init_train_x = torch.cat(init_train_x)
    # get scores for each sequence 
    init_train_y = objective.four_seqs_list_to_scores(init_train_strings)

    # train_x, train_y, train_c, train_strings
    return init_train_x, init_train_y, init_train_c, init_train_strings

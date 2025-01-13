import torch 

def convert_seqs_list_to_quads(seqs):
    quads_train_x_seqs = []
    idx = 0 
    for seq in seqs:
        if idx == 0:
            quad = [seq]
            idx += 1
        else:
            quad.append(seq)
            idx += 1
        if idx == 4:
            quads_train_x_seqs.append(quad)
            idx = 0 
    return quads_train_x_seqs


def get_edit_distances_from_starter(
    sequences_list,
    starter_seq,
    dtype,
):
    edit_dists = []
    for seq_quad in sequences_list:
        edit_dist = get_total_edit_dist_between_quads(
            four_seqs_1=seq_quad, 
            four_seqs_2=starter_seq,
        )
        edit_dists.append(edit_dist)
    return torch.tensor(edit_dists).unsqueeze(-1).to(dtype=dtype)


def get_total_edit_dist_between_quads(four_seqs_1, four_seqs_2):
    # Want to compute total edit distance by summing edit distance
    #   between each pair of the four subseqneces.
    edit_dist = 0
    for i in range(4):
        edit_dist += levenshtein_edit_distance(four_seqs_1[i], four_seqs_2[i])
    return edit_dist


def levenshtein_edit_distance(s1, s2): 
    ''' Returns Levenshtein Edit Distance btwn two strings'''
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1 
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]  


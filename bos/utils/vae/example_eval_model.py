import torch
from .data import DataModuleKmers 
from .transformer_vae_unbounded import InfoTransformerVAE 

def main(batch_size=128):
    test_seqs_list =  [
        'CTTAAAAAAAAAAAAACCAACAATCCACCCTACTTACTCTTTTTTCTGTATGACTGTCAATCATGGAACTTGTTAACTTGCTGTTTCTAGAACAAATTGGATGGAACAGTATGGACTGAAATTGATGATACCAAAGTCTTCAAAATTCTAGACCTTGAAGACCTAGAAAGAACGTTCTCTGCCTACCAAAGACAGCAGGTAACAACATGAGCCTTCCAGATGCCCCAAAGTCCTGTGTCCTTGGGCATGGCTCCTGGGCCATGCTGCTTGGTCTCCCCAAGGCGTTCCTGTAAAGCAGCACAGCTACCTGTCCCTGTGCGTGTCCAGTGGCCCATCAGTGCATGTGAACCACTGGAAGGCATGTGGACTCTTTGTGGAGAAGCTTCCTATG', 
        'CTTACACTGTGATCTTCACTGTATCTTCAGCCAGACTGGCCCACTTGAGCTAAGGACTTAAAGGGCAGGAAACATCTCCTGCCCTTTTGTCCTGCACCACCAATGTCCTTAACTGCTTTGACTTTCCCTGTTTGCTGCTGTGGTTTCTGAATTGCATGTATCTGATTTGCTTTCTCTATGCCTCTGCCATGGCTGTAGGAGTTCTTTGTGAACAACTCCAAGCAGGTAAGCAAGCTACTATATAACCCTTATTTAACACAGCTCCTTGGCTCACTGTAAGCTTGAAATCCCTTGGTTATCTTGTTAGCTAATGTAAGACTAGGTGATTCCACTGTGGGATTCAACTCTTCCTGGACTGTCTCTCCAGAGTATGCCTGAAATCCGAGCACAA', 
        'CAGCCAGACTGGCCCACTTGAGCTAAGGACTTAAAGGGCAGGAAACATCTCCTGCCCTTTTGTCCTGCACCACCAATGTCCTTAACTGCTTTGACTTTCCCTGTTTGCTGCTGTGGTTTCTGAATTGCATGTATCTGATTTGCTTTCTCTATGCCTCTGCCATGGCTGTAGGAGTTCTTTGTGAACAACTCCAAGCAGGTAAGCAAGCTACTATATAACCCTTATTTAACACAGCTCCTTGGCTCACTGTAAGCTTGAAATCCCTTGGTTATCTTGTTAGCTAATGTAAGACTAGGTGATTCCACTGTGGGATTCAACTCTTCCTGGACTGTCTCTCCAGAGTATGCCTGAAATCCGAGCACAAGAGAGAGACCTTTTCTCGTTTAGCCCC', 
        'GGTCAGCTTTGTCTTAGCCTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTGACTGCACCACCTTCTGAACATTGCCTCATCATCTCAGTTTGGCCAGGGTATGGGTATGTGCACAGAGAATGTGGAGGAAAAATGTGTGTGTGTGGAGCCTGACTGTCCACATTTGACTTTGATGGATGTCTCTTGTGTTCTTTCCAGAAAGAAGCAGATGCCATTGATGACACACTGAGTTCCAAACTTAAAGTCAAAGAGCTGTCAGTGATTGATGGTCGGAGAGCTCAGAATTGCAACATCCTTCTGTCGAGGTACTGTTTGGTGCTGAATAAACATGGCTGGGAGGACCAGCTCTGCTCTTAGCTTAGAACCAGAGGGCTGCTGTTCAATTTTACTCA',
    ]
    state_dict_path = 'saved_model_weights/hearty-gorge-28_model_state.pkl'
    datamodule = DataModuleKmers(batch_size, k=3, test_seqs_list=test_seqs_list ) 

    model = InfoTransformerVAE(dataset=datamodule.train)
    state_dict = torch.load(state_dict_path) # load state dict 
    model.load_state_dict(state_dict, strict=True) 
    
    test_loader = datamodule.test_dataloader() 
    model = model.cuda()    
    model = model.eval()  
    sum_loss = 0.0 
    sum_token_acc = 0.0
    sum_str_acc = 0.0
    num_iters = 0 
    for data in test_loader:
        input1 = data.cuda() 
        out_dict = model(input1)
        sum_loss += out_dict['loss'].item() 
        sum_token_acc += out_dict['recon_token_acc'].item() 
        sum_str_acc += out_dict['recon_string_acc'].item() 
        num_iters += 1 

    avg_loss = sum_loss/num_iters 
    avg_token_acc = sum_token_acc/num_iters 
    avg_str_acc = sum_str_acc/num_iters 

    print(f"Mean loss: {avg_loss}, Mean token acc {avg_token_acc}, Mean string acc {avg_str_acc}")


if __name__ == "__main__":
    main() 

import sys 
sys.path.append("../")
import math 
import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_single_model(
    model,
    mll,
    train_x,
    train_y,
    lr=0.01,
    n_epochs=30,
    train_bsz=32,
    grad_clip=1.0,
    train_to_convergence=True, 
    max_allowed_n_failures_improve_loss=10,
    max_allowed_n_epochs=100,
):
    model.train()
    params_list = [{"params": model.parameters(), "lr":lr}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    train_bsz = min(len(train_y),train_bsz)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    lowest_loss = torch.inf 
    n_failures_improve_loss = 0
    epochs_trained = 0
    continue_training_condition = True 
    while continue_training_condition:
        total_loss = 0
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = -mll(output, scores.squeeze().to(device))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        epochs_trained += 1
        if total_loss < lowest_loss:
            lowest_loss = total_loss
        else:
            n_failures_improve_loss += 1
        if train_to_convergence:
            continue_training_condition = n_failures_improve_loss < max_allowed_n_failures_improve_loss
            if epochs_trained > max_allowed_n_epochs:
                continue_training_condition = False 
        else:
            continue_training_condition = epochs_trained < n_epochs
    
    model.eval()
    return model, mll 

def update_models_end_to_end_with_vae(
    objective, # lsbo objective with vae object 
    gp_model,
    gp_mll,
    starter_seq,
    train_strings,
    train_y,
    constraint_func, 
    train_c=None,
    c_model=None,
    c_mll=None,
    lr=0.0001,
    n_epochs=30,
    train_bsz=32,
    grad_clip=1.0,
    dtype=torch.float32,
):
    # First train end to end 
    train_bsz = min(len(train_y),train_bsz)
    gp_model.train()
    objective.vae.train()
    all_models_list = [gp_model, objective.vae]
    if c_model is not None:
        c_model.train()
        all_models_list.append(c_model)
    torch_models_list = torch.nn.ModuleList(all_models_list)
    params_list = [{"params": torch_models_list.parameters(), "lr":lr}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    num_batches = math.ceil(len(train_strings) / train_bsz)
    for _ in range(n_epochs):
        for batch_ix in range(num_batches):
            optimizer.zero_grad()
            start_idx, stop_idx = batch_ix*train_bsz, (batch_ix+1)*train_bsz
            # get vae loss 
            batch_strings = train_strings[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch_strings)
            # add gp models loss 
            batch_y = train_y[start_idx:stop_idx]
            output = gp_model(z)
            gp_loss = -gp_mll(output, batch_y.squeeze().to(device))
            loss = vae_loss + gp_loss
            # add constraint model loss 
            if c_model is not None:
                batch_c = train_c[start_idx:stop_idx]
                output = c_model(z)
                loss_c = -c_mll(output, batch_c.squeeze().to(device))
                loss = loss + loss_c 
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(torch_models_list.parameters(), max_norm=grad_clip)
            optimizer.step()

    # ________________________
    # Then do lolbo-recentering 
    objective.vae.eval()
    torch_models_list = torch.nn.ModuleList([gp_model])
    # no vae updates, just updating models on new data locations 
    if c_model is not None:
        torch_models_list = torch.nn.ModuleList([gp_model, c_model])
    params_list = [{"params": torch_models_list.parameters(), "lr":lr}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    all_new_ys, all_new_xs, all_new_seqs, all_new_cs = [], [], [], []
    for _ in range(n_epochs):
        for batch_ix in range(num_batches):
            optimizer.zero_grad()
            start_idx, stop_idx = batch_ix*train_bsz, (batch_ix+1)*train_bsz
            # pass through vae to get new z 
            with torch.no_grad(): 
                batch_strings = train_strings[start_idx:stop_idx]
                z, _ = objective.vae_forward(batch_strings)
            out_dict = objective(z)
            new_ys = out_dict["ys"]
            new_strings = out_dict["strings"]
            all_new_seqs = all_new_seqs + new_strings
 
            # add gp models loss 
            output = gp_model(z)
            recentering_loss = -gp_mll(output, new_ys.squeeze().to(device))
            # add constraint model loss 
            if c_model is not None:
                new_cs = constraint_func(
                    sequences_list=new_strings,
                    starter_seq=starter_seq,
                    dtype=dtype,
                )
                output = c_model(z)
                loss_c = -c_mll(output, new_cs.squeeze().to(device))
                recentering_loss = recentering_loss + loss_c 
                all_new_cs.append(new_cs.detach().cpu())
            recentering_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(torch_models_list.parameters(), max_norm=grad_clip)
            optimizer.step()
            all_new_ys.append(new_ys.detach().cpu())
            all_new_xs.append(z.detach().cpu()) #asdfghjkl

    if c_model is not None:
        all_new_cs = torch.cat(all_new_cs)
    else:
        all_new_cs = None 

    gp_model.eval()
    if c_model is not None:
        c_model.eval()

    return_dict = {}
    return_dict["gp_model"] = gp_model
    return_dict["gp_mll"] = gp_mll
    return_dict["c_model"] = c_model 
    return_dict["c_mll"] = c_mll 
    return_dict["objective"] = objective

    return_dict["new_ys"] = torch.cat(all_new_ys, dim=-2)
    return_dict["new_xs"] = torch.cat(all_new_xs, dim=-2)
    return_dict["new_seqs"] = all_new_seqs
    return_dict["new_cs"] = all_new_cs

    return return_dict

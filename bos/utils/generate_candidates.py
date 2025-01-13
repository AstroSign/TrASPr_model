import sys 
sys.path.append("../")
import torch
from torch.quasirandom import SobolEngine
# from botorch.generation.sampling import MaxPosteriorSampling
from utils.botorch_sampling import MaxPosteriorSampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(
    gp_model,
    X,
    batch_size,
    tr_center,
    tr_length,
    n_candidates=None,
    dtype=torch.float32,
    device=torch.device('cuda'),
    lb=None,
    ub=None,
    # constraint args (SCBO)
    c_model=None,
    edit_distance_threshold=30,
    avg_over_n_samples_c_model=20,
):
    if n_candidates is None: 
        n_candidates = min(5000, max(2000, 200 * X.shape[-1])) 

    weights = torch.ones_like(tr_center)

    if (lb is None): # if no absolute bounds 
        lb = X.min().item() 
        ub = X.max().item()
        temp_range_bounds = ub - lb 
        lb = lb - temp_range_bounds*0.1 # add small buffer lower than lowest we've seen 
        ub = ub + temp_range_bounds*0.1 # add small buffer higher than highest we've seen 

    weights = weights * (ub - lb)
    tr_lb = torch.clamp(tr_center - weights * tr_length / 2.0, lb, ub) 
    tr_ub = torch.clamp(tr_center + weights * tr_length / 2.0, lb, ub) 
    lb = tr_lb 
    ub = tr_ub 

    # create ts cands (turbo implementation)
    dim = X.shape[-1]
    lb = lb.to(device)
    ub = ub.to(device)
    sobol = SobolEngine(dim, scramble=True) 
    pert = sobol.draw(n_candidates).to(dtype=dtype).to(device)
    pert = lb + (ub - lb) * pert
    lb = lb.to(device)
    ub = ub.to(device)
    # Create a perturbation mask 
    prob_perturb = min(20.0 / dim, 1.0)
    mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
    mask = mask.to(device)
    # Create candidate points from the perturbations and the mask
    X_cand = tr_center.expand(n_candidates, dim).clone()
    X_cand = X_cand.to(device)
    X_cand[mask] = pert[mask]

    if c_model is not None:
        # if constrained OPT with SCBO, remove x cands that c_model predicts will not meet constraint 
        with torch.no_grad():
            X_cand = remove_samples_unlikely_to_meet_constraint(
                c_model=c_model,
                X_cand=X_cand, # (N_cand, dim)
                avg_over_n_samples=avg_over_n_samples_c_model, # S
                edit_distance_threshold=edit_distance_threshold,
                batch_size=batch_size,
            )
        if X_cand.shape[0] == batch_size: # if we have exactly bsz candidates that meet constraints, we are done 
            return X_cand.detach().cpu()
    
    with torch.no_grad():
        thompson_sampling = MaxPosteriorSampling(
            model=gp_model,
            replacement=False,
        ) 
        with torch.no_grad():
            X_next = thompson_sampling(X_cand.to(device), num_samples=batch_size )

    return X_next.detach().cpu()


def remove_samples_unlikely_to_meet_constraint(
    c_model,
    X_cand, # (N_cand, dim)
    avg_over_n_samples=20, # S
    edit_distance_threshold=30,
    batch_size=20,
):
    N_cand = X_cand.shape[0]
    dim = X_cand.shape[1]
    all_y_samples = []  # compresed y samples
    posterior = c_model.posterior(X_cand, observation_noise=False)
    all_y_samples = posterior.rsample(sample_shape=torch.Size([avg_over_n_samples])) # (avg_over_n_samples, N_cand, 1)
    all_y_samples = all_y_samples.squeeze() # (avg_over_n_samples, N_cand) i.e. torch.Size([10, 500])

    # average over S samples to get avg c_val per candidate 
    avg_pred_c_val = all_y_samples.mean(0) # (N_cand,)
    assert avg_pred_c_val.shape[0] == N_cand
    assert len(avg_pred_c_val.shape) == 1 
    feasible_x_cand = X_cand[avg_pred_c_val <= edit_distance_threshold] # (N_feasible, dim)
    # if none/ too few of the samples meet the constraints, pick the batch_size candidates that minimize violation (SCBO paper)
    if feasible_x_cand.shape[0] < batch_size:
        # equivalently, take bottom k predicted edit distances from starter m
        min_violator_indicies = torch.topk(avg_pred_c_val*-1, k=batch_size).indices # torch.Size([batch_size]) of ints 
        feasible_x_cand = X_cand[min_violator_indicies] # (batch_size, dim)
        assert feasible_x_cand.shape[0] == batch_size
    assert feasible_x_cand.shape[1] == dim 
    assert len(feasible_x_cand.shape) == 2

    # return only X cands that are likley to be feasible according to c_model 
    return feasible_x_cand # (N_feasible, dim)

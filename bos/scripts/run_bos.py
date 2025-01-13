import sys 
sys.path.append("../")
import torch
import numpy as np
import fire 
import pandas as pd
import warnings
import copy
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood
from utils.gp_dkl import GPModelDKL 
from utils.train_model import (
    update_single_model,
    update_models_end_to_end_with_vae
)
from utils.create_wandb_tracker import create_wandb_tracker
from utils.set_seed import set_seed 
from utils.get_random_init_data import get_random_init_data
from utils.turbo import TurboState, update_state
from tasks.splicing_objective import SplicingObjective
from utils.seq_utils import get_edit_distances_from_starter
import botorch
from utils.generate_candidates import generate_batch


SEQ_NUM_TO_ID = {
    1:"seq_E1+I1",
    2:"seq_I1+A",
    3:"seq_A+I2",
    4:"seq_I2+E2",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Optimize(object):
    """
    Run Multi Objective COverage Bayesian Optimization (MOCOBO)
    Args:
        tissue: String id for target tissue  
        minimize_psi: if True, we find sequences to minimize psi, otherwise, we find sequences to maximize psi 
        edit_distance_threshold: int, max allowed edit distance from starter sequence 
        path_to_starter_sequence: path to tsv file with desired starter sequence 
        starter_sequence_id: int, id for particular desired starter sequence in tsv file 
        oracle_model_weights_path: path to oracle model weights (i.e. "home/bos/best_gtex_checkpoint"), these are weights for the pre-trained DNABERT_light oracle used to evaluate PSI 
        seed: Random seed to be set. If None, no particular random seed is set
        wandb_entity: Username for your wandb account for wandb logging
        wandb_project_name: Name of wandb project where results will be logged, if none specified, will use default f"bos-{tissue}"
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        bsz: Acquisition batch size (how many new datapoints are selected to evaluate on each step of optimization)
        train_bsz: batch size used for model training/updates
        num_initialization_points: Number of initial random sequences used to kick off optimization
        lr: Learning rate for surrogate model updates
        n_update_epochs: Number of epochs to update the model for on each optimization step
        n_inducing_pts: Number of inducing points for GP surrogate model 
        grad_clip: clip the gradeint at this value during model training 
        train_to_convergence: if true, train until loss stops improving instead of only for n_update_epochs
        max_allowed_n_failures_improve_loss: We train model until the loss fails to improve for this many epochs
        max_allowed_n_epochs: When we train to convergence, we also cap the number of epochs to this max allowed value
        update_on_n_pts: Update model on this many data points on each iteration.
        e2e_freq: number of concecutive failures to make progress between each end-to-end update (vae + surrogate model joint update) 
        n_e2e_update_epochs: number of epochs to update vae and surrogate models end-to-end 
        e2e_lr: learning rate for end to end updates (lower for more stable vae+gp updates)
        float_dtype_as_int: specify integer either 32 or 64, dictates whether to use torch.float32 or torch.float64 
        verbose: if True, print optimization progress updates 
        run_id: Optional string run id. Only use is for wandb logging to identify a specific run
    """
    def __init__(
        self,
        tissue: str="brain",
        minimize_psi=False,
        edit_distance_threshold: int=30,
        path_to_starter_sequence: str="../starter_sequences/dev_3.tsv",
        starter_sequence_id: str="chr14_+_modulizer_nonchg_nonskip_00000100_nonchgCase_-31.65_G36M_S0fpoc",
        oracle_model_weights_path: str=None,
        seed: int=None,
        wandb_entity: str="nmaus-penn",
        wandb_project_name: str="",
        max_n_oracle_calls: int=500_000,
        bsz: int=10,
        train_bsz: int=32,
        num_initialization_points: int=2_000,
        lr: float=0.001,
        n_update_epochs: int=2,
        n_inducing_pts: int=1024,
        grad_clip: float=1.0,
        train_to_convergence=True,
        max_allowed_n_failures_improve_loss: int=3,
        max_allowed_n_epochs: int=30, 
        update_on_n_pts: int=1_000,
        e2e_freq: int=10,
        float_dtype_as_int: int=32,
        verbose=True,
        n_e2e_update_epochs: int=2,
        e2e_lr: float=0.001,
        run_id: str="",
    ):
        if float_dtype_as_int == 32:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)
        elif float_dtype_as_int == 64:
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            assert 0, f"float_dtype_as_int must be one of: [32, 64], instead got {float_dtype_as_int}"
        
        # log all args to wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        wandb_config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}

        self.minimize_psi = minimize_psi
        self.oracle_model_weights_path = oracle_model_weights_path
        self.tissue = tissue 
        self.edit_distance_threshold = edit_distance_threshold
        self.path_to_starter_sequence = path_to_starter_sequence
        self.starter_sequence_id = starter_sequence_id
        self.e2e_lr = e2e_lr
        self.n_e2e_update_epochs = n_e2e_update_epochs
        self.e2e_freq = e2e_freq
        self.run_id = run_id
        self.init_training_complete = False
        self.update_on_n_pts = update_on_n_pts
        self.verbose = verbose
        self.max_allowed_n_failures_improve_loss = max_allowed_n_failures_improve_loss
        self.max_allowed_n_epochs = max_allowed_n_epochs
        self.max_n_oracle_calls = max_n_oracle_calls
        self.n_inducing_pts = n_inducing_pts
        self.lr = lr
        self.n_update_epochs = n_update_epochs
        self.train_bsz = train_bsz
        self.grad_clip = grad_clip
        self.bsz = bsz
        self.train_to_convergence = train_to_convergence
        self.num_initialization_points = num_initialization_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        # start wandb tracker
        if not wandb_project_name:
            wandb_project_name = f"bos-{tissue}"
        # tracker, wandb_run_name
        self.tracker, self.wandb_run_name = create_wandb_tracker(
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            config_dict=wandb_config_dict,
        )
        signal.signal(signal.SIGINT, self.handler)

        # read in starter sequence 
        self.get_starter_sequence()

        # initialize objective 
        self.objective = SplicingObjective(
            dtype=self.dtype,
            tissue=self.tissue,
            minimize_psi=self.minimize_psi,
            oracle_model_weights_path=self.oracle_model_weights_path,
        )

        # get initialization data 
        self.get_initialization_data()
        assert self.train_y.shape[0] == self.num_initialization_points
        assert self.train_y.shape[1] == 1 
        assert self.train_c.shape[0] == self.num_initialization_points
        assert self.train_c.shape[1] == 1
        assert self.train_x.shape[0] == self.num_initialization_points
        assert self.train_x.shape[1] == self.objective.dim 
        assert type(self.train_strings) == list 
        assert len(self.train_strings) == self.num_initialization_points 
 
        # None indicicates they must be initialized still 
        self.tr_state = None 
 
        # get inducing points
        inducing_points = self.train_x[0:self.n_inducing_pts,:]

        # Define GP surrogate model
        self.gp_model = GPModelDKL(
            inducing_points=inducing_points.to(self.device),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        ).to(device)
        self.gp_mll = PredictiveLogLikelihood(
            self.gp_model.likelihood, self.gp_model, num_data=self.train_x.shape[0]
        )
        # Define GP surrogate constraint model
        self.c_model = GPModelDKL(
            inducing_points=inducing_points.to(self.device),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        ).to(device)
        self.c_mll = PredictiveLogLikelihood(
            self.c_model.likelihood, self.c_model, num_data=self.train_x.shape[0]
        )

    def get_starter_sequence(self,):
        df = pd.read_csv(self.path_to_starter_sequence, sep = '\t')
        df = df[df["ID"] == self.starter_sequence_id]
        seq1 = df['seq_E1+I1'].values[0]
        seq2 = df['seq_I1+A'].values[0]
        seq3 = df['seq_A+I2'].values[0]
        seq4 = df['seq_I2+E2'].values[0]
        self.starter_sequence = [seq1, seq2, seq3, seq4]
        for seq in self.starter_sequence:
            assert type(seq) == str
        return self 

    def get_initialization_data(self,):
        # get random init training data 
        train_x, train_y, train_c, train_strings = get_random_init_data(
            starter_sequence=self.starter_sequence,
            edit_distance_threshold=self.edit_distance_threshold,
            num_initialization_points=self.num_initialization_points,
            objective=self.objective,
            dtype=self.dtype,
        )

        if self.verbose:
            print("train x shape:", train_x.shape)
            print("train y shape:", train_y.shape)
            print(f"N={len(train_strings)} train sequences, i.e.", train_strings[0])

        self.train_x = train_x
        self.train_y = train_y 
        self.train_c = train_c 
        self.all_train_c_w_infeasible = train_c 
        self.all_train_x_w_infeasible = train_x
        self.train_strings = train_strings 

        return self 


    def grab_data_for_update(self,):
        if not self.init_training_complete:
            x_update_on = self.train_x
            y_update_on = self.train_y.squeeze()
            c_update_on = self.train_c
            c_update_on_w_infeasilbe = self.all_train_c_w_infeasible
            x_update_on_w_infeasilbe = self.all_train_x_w_infeasible
            self.init_training_complete = True
            strings_update_on = self.train_strings 
        else:
            # update on latest collected update_on_n_pts data, plus best covering set k seen so far 
            total_n_data = self.train_x.shape[0]
            top_indicies = torch.topk(self.train_y.squeeze(), k=self.update_on_n_pts).indices.tolist()
            latest_data_idxs = np.arange(total_n_data)[-self.bsz:].tolist()
            idxs_update_on = latest_data_idxs + top_indicies
            idxs_update_on = list(set(idxs_update_on)) # removes duplicates 
            x_update_on = self.train_x[idxs_update_on]
            y_update_on = self.train_y.squeeze()[idxs_update_on]
            if self.train_c is None:
                c_update_on = None 
            else:
                c_update_on = self.train_c[idxs_update_on]

            if self.train_strings is None:
                strings_update_on = None 
            else:
                strings_update_on = np.array(self.train_strings)[idxs_update_on].tolist()

            if self.all_train_c_w_infeasible is None:
                c_update_on_w_infeasilbe = None 
                x_update_on_w_infeasilbe = None 
            else:
                # best indexes don't apply to full train c/x with infeasilbe data included 
                c_update_on_w_infeasilbe = self.all_train_c_w_infeasible[-self.bsz:]
                x_update_on_w_infeasilbe = self.all_train_x_w_infeasible[-self.bsz:]

        self.x_update_on = x_update_on
        self.y_update_on = y_update_on
        self.c_update_on = c_update_on
        self.c_update_on_w_infeasilbe = c_update_on_w_infeasilbe
        self.x_update_on_w_infeasilbe = x_update_on_w_infeasilbe
        self.strings_update_on = strings_update_on

        return self 

    def create_save_data_dir(self,):
        save_data_dir = f"../save_opt_data/{self.tissue}/"
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        save_data_dir = save_data_dir + f"{self.wandb_run_name}/"
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
        self.save_data_dir = save_data_dir
        return self 
    

    def save_run_data(self,):
        save_df = {}
        psis_collected = self.train_y.squeeze()
        if self.minimize_psi:
            # if minimizing psi, bos tries to increase -psi, 
            # so multiply by -1 again here to save true psi values 
            psis_collected = psis_collected*-1 
        save_df[f"psi_{self.tissue}"] = psis_collected.tolist()
        # seq_E1+I1,seq_I1+A,seq_A+I2,seq_I2+E2
        for seq_n in SEQ_NUM_TO_ID:
            save_df[SEQ_NUM_TO_ID[seq_n]] = []
        for four_seqs in self.train_strings:
            for i in range(4):
                save_df[SEQ_NUM_TO_ID[i+1]].append(four_seqs[i])
        save_df = pd.DataFrame.from_dict(save_df)
        save_df.to_csv(f"{self.save_data_dir}all_sequences_and_psis_found.csv", index=False)
        return self 


    def run(self):
        ''' Main optimization loop
        '''
        self.total_n_infeasible_thrown_out = 0 # count num candidates thrown out due to not meeting constraint(s)
        self.progress_fails_since_last_e2e = 0 # for LOL-BO track of when to do end-to-end updates only 
        self.count_n_e2e_updates = 0 
        prev_best_y = -np.inf 
        self.create_save_data_dir()
        while self.objective.num_calls < self.max_n_oracle_calls:
            # compute best strings and best ys 
            best_ix = self.train_y.argmax().item()
            best_y = self.train_y[best_ix].item()
            best_x = self.train_x[best_ix]

            if best_y > prev_best_y:
                # if we improved, save data  
                self.save_run_data()
                prev_best_y = best_y 
            else: 
                # if no imporvement, count one failure to imporve 
                self.progress_fails_since_last_e2e += 1 

            # Print progress update and update wandb with optimization progress
            n_calls_ = self.objective.num_calls
            if self.verbose:
                print(f"Optimizing psi for tissue: {self.tissue}, Wandb run: {self.wandb_run_name}, After {n_calls_} oracle calls, best score found = {best_y}")
            dict_log = {
                "best_y":best_y,
                "n_oracle_calls":n_calls_,
                "total_n_infeasible_thrown_out":self.total_n_infeasible_thrown_out,
                "count_n_e2e_updates":self.count_n_e2e_updates,
            }
            self.tracker.log(dict_log)
            
            # Update model on data 
            self.grab_data_for_update()
            if self.progress_fails_since_last_e2e >= self.e2e_freq:
                # Do end-to-end vae+gp updates (LOL-BO)
                self.update_surrogate_models_and_vae_end_to_end() 
            else: 
                # otherwise, just update the surrogate models on data
                self.update_surrogate_models()

            # update trust region state 
            self.update_trust_region(best_x, best_y)

            # Generate a batch of candidates 
            x_next = generate_batch(
                gp_model=self.gp_model,
                X=self.train_x,
                batch_size=self.bsz,
                tr_center=self.tr_state.center,
                tr_length=self.tr_state.length,
                dtype=self.dtype,
                device=device,
                lb=self.objective.lb,
                ub=self.objective.ub,
                # constraint args (SCBO)
                c_model=self.c_model, # init 
                edit_distance_threshold=self.edit_distance_threshold,
            )
            assert x_next.shape[-1] == self.objective.dim 

            # Evaluate candidates
            out_dict = self.objective(x_next)
            y_next = out_dict["ys"]
            strings_next = out_dict["strings"]
            if self.c_model is not None:
                # Compute constraint values and remove new data that doesn't meet constraint 
                c_next = get_edit_distances_from_starter(
                    sequences_list=strings_next,
                    starter_seq=self.starter_sequence,
                    dtype=self.dtype,
                ) # c_next is a tensor (n_seqs,1)
            else:
                c_next = None 

            self.update_datasets(x_next, y_next, c_next, strings_next)

        # final save for all run data 
        self.save_run_data()
        # terminate wandb tracker 
        self.tracker.finish()
        return self


    def update_datasets(self, x_next, y_next, c_next, strings_next):
        if c_next is not None:
            # Remove new data that doesn't meet constraint 
            self.all_train_c_w_infeasible = torch.cat((self.all_train_c_w_infeasible, c_next))
            self.all_train_x_w_infeasible = torch.cat((self.all_train_x_w_infeasible, x_next), dim=-2)
            feasible_next_bool_array = c_next.squeeze() <= self.edit_distance_threshold
            # Subtract num infeasible from total num oracle calls bc we don't actually need to eval oracle on these 
            num_infeasible = (feasible_next_bool_array == False).sum().item() 
            self.objective.num_calls = self.objective.num_calls - num_infeasible 
            self.total_n_infeasible_thrown_out += num_infeasible
            # update next feasible candidates only 
            x_next = x_next[feasible_next_bool_array] 
            y_next = y_next[feasible_next_bool_array]
            c_next = c_next[feasible_next_bool_array]
            strings_next = np.array(strings_next)[feasible_next_bool_array.numpy()].tolist()

        # Update data
        self.train_x = torch.cat((self.train_x, x_next), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next), dim=-2)
        if self.train_c is not None:
            self.train_c = torch.cat((self.train_c, c_next))
        if strings_next is not None:
            self.train_strings = self.train_strings + strings_next 
        
        return self 

    def update_trust_region(self, best_x, best_y):
        if self.tr_state is not None:
            self.tr_state = update_state(
                state=self.tr_state,
                new_best=best_y,
            )
        if (self.tr_state is None) or self.tr_state.restart_triggered:
            self.tr_state = TurboState(
                dim=self.train_x.shape[-1],
                batch_size=self.bsz,
                best_value=best_y,
                center=best_x,
            )
        self.tr_state.center = best_x
        return self

    def update_surrogate_models_and_vae_end_to_end(self,):
        # includes recentering after e2e updates 
        gp_model_state_before_update = copy.deepcopy(self.gp_model.state_dict())
        gp_mll_state_before_update = copy.deepcopy(self.gp_mll.state_dict())
        if self.c_model is not None:
            c_model_state_before_update = copy.deepcopy(self.c_model.state_dict())
            c_mll_state_before_update = copy.deepcopy(self.c_mll.state_dict())
        # load state dict for vae corresponding to task 
        vae_state_before_update = copy.deepcopy(self.objective.vae.state_dict())
        try: 
            if len(self.y_update_on.shape) == 1:
                self.y_update_on = self.y_update_on.unsqueeze(-1) # (N_update, 1)
            assert self.y_update_on.shape[-1] == 1
            assert self.y_update_on.shape[0] == len(self.strings_update_on)
            assert len(self.y_update_on.shape) == 2 # # (N_update, 1)
            out_dict = update_models_end_to_end_with_vae(
                objective=self.objective,
                gp_model=self.gp_model,
                gp_mll=self.gp_mll,
                starter_seq=self.starter_sequence,
                train_strings=self.strings_update_on,
                train_y=self.y_update_on,
                constraint_func=get_edit_distances_from_starter, 
                train_c=self.c_update_on,
                c_model=self.c_model,
                c_mll=self.c_mll,
                lr=self.e2e_lr,
                n_epochs=self.n_e2e_update_epochs,
                train_bsz=self.train_bsz,
                grad_clip=self.grad_clip,
                dtype=self.dtype,
            )
            # update models 
            self.gp_model = out_dict["gp_model"]
            self.gp_mll = out_dict["gp_mll"]
            self.c_model = out_dict["c_model"]
            self.c_mll = out_dict["c_mll"]
            self.objective = out_dict["objective"] 
            # update datasets of recentered data 
            self.update_datasets(
                x_next=out_dict["new_xs"], 
                y_next=out_dict["new_ys"], 
                c_next=out_dict["new_cs"] , 
                strings_next=out_dict["new_seqs"],
            )
            # reset progress fails since last e2e update to 0
            self.progress_fails_since_last_e2e = 0 
            # count e2e update 
            self.count_n_e2e_updates += 1
        except:
            torch.cuda.empty_cache()
            # In case of nan loss due to unstable trianing, 
            # 1. reset models to prev state dicts before unstable training 
            self.gp_model.load_state_dict(gp_model_state_before_update)
            self.gp_mll.load_state_dict(gp_mll_state_before_update)
            self.objective.vae.load_state_dict(vae_state_before_update)
            if self.c_model is not None:
                self.c_model.load_state_dict(c_model_state_before_update)
                self.c_mll.load_state_dict(c_mll_state_before_update)
            # 2. reduce lr for next time to reduce prob of training instability
            self.e2e_lr = self.e2e_lr/2 
        torch.cuda.empty_cache()
        return self 

    def update_surrogate_models(self,):
        if self.c_update_on_w_infeasilbe is not None:
            # Update constraint model on all xs and constraint values 
            #   (including infeasible data that doesn't meet constraints)
            self.c_model, self.c_mll = update_single_model(
                model=self.c_model,
                mll=self.c_mll,
                train_x=self.x_update_on_w_infeasilbe,
                train_y=self.c_update_on_w_infeasilbe,
                lr=self.lr,
                n_epochs=self.n_update_epochs,
                train_bsz=self.train_bsz,
                grad_clip=self.grad_clip,
                train_to_convergence=self.train_to_convergence, 
                max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                max_allowed_n_epochs=self.max_allowed_n_epochs, 
            )
        
        self.gp_model, self.gp_mll = update_single_model(
            model=self.gp_model,
            mll=self.gp_mll,
            train_x=self.x_update_on,
            train_y=self.y_update_on,
            lr=self.lr,
            n_epochs=self.n_update_epochs,
            train_bsz=self.train_bsz,
            grad_clip=self.grad_clip,
            train_to_convergence=self.train_to_convergence, 
            max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
            max_allowed_n_epochs=self.max_allowed_n_epochs, 
        )

        return self 

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we terminate wandb tracker
        print("Ctrl-c hass been pressed, terminating wandb tracker...")
        self.tracker.finish()
        msg = "tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)
        return None 

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)


if __name__ == "__main__":
    fire.Fire(Optimize)
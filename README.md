# Transformer for Alternative Splice Predictions(TrASPr)
This repository includes the implementation of 'Generative modeling for RNA splicing
predictions and design'. Please cite our paper if you use the models or codes. The repo is still actively under development, so please kindly report if there is any issue encountered.

 In this package, we provides resources including: source codes of the TrASPr model, BOS, and scripts for reproducing figures in the paper. This package is still under development, as more features will be included gradually.

## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version >= 410.48 (compatible with CUDA 10.0). 

#### 1.1 Create and activate a new virtual environment

```
conda create -n traspr python=3.8
conda activate traspr
```

#### 1.2 Install the packages and other requirements for TrASPr model 


(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

cd TrASPr_model
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

## 2. Pre-train

```
cd examples

export KMER=6
export TRAIN_FILE=PATH_TO_TRASPR_TRAIN_DATA
export TEST_FILE=PATH_TO_TRASPR_TEST_DATA
export SOURCE=PATH_TO_TRASPR_REPO
export OUTPUT_PATH=output$KMER

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 15 \
    --per_gpu_train_batch_size 40 \
    --per_gpu_eval_batch_size 24 \
    --save_steps 5000 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 5000 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 400 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.15 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --embedding_method RoPE \
    --n_process 24

```

## 3. Fine-tuning

```
export KMER=6
export MODEL_PATH=PRETRAIN_MODEL_PATH
export DATA_PATH=DATA_PATH
export OUTPUT_PATH=OUTPUT_PATH

export TF_MODE=multi_tf_two_mlp
export TASK_NAME=dnacass
export MODEL_TYPE=dnamultitrans
export MAX_LEN=410

python finetune_psi.py \
    --model_type $MODEL_TYPE \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length $MAX_LEN \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8   \
    --learning_rate 1e-5 \
    --max_grad_norm 20.0 \
    --num_train_epochs 20.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --overwrite_output_dir \
    --logging_steps 5000 \
    --save_steps 5000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.2 \
    --overwrite_output \
    --weight_decay 0.01 \
    --consval \
    --use_features \
    --sampler ss \
    --tf_mode $TF_MODE \
    --multi_weight_dpsi 2 \
    --n_process 8

```

## 4. TrASPr Output

TrASPr will save a .npy file with predicted PSI, dPSI+ and dPSI- for each sequences in the dev.tsv file from the DATA_PATH. For each checkpoint, there will be corresponding prediction results. The final results and checkpoint results will be stored under OUTPUT_PATH.

## 5. BOS

Please refer to bos folder for details.
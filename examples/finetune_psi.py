# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertForSequenceMultiClassification,
    BertForSequenceMultiClassificationMultiTransformer,
    BertForSequenceMultiClassificationMultiTransformerAlt,
    BertForSequenceMultiClassificationMultiTransformerAltFull,
    BertTokenizer,
    DNATokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    LSTMForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import mse_pearson_and_spearman as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "dnamulti": (BertConfig, BertForSequenceMultiClassification, DNATokenizer),
    "dnamultitrans": (BertConfig, BertForSequenceMultiClassificationMultiTransformer, DNATokenizer),
    "dnamultitransalt": (BertConfig, BertForSequenceMultiClassificationMultiTransformerAlt, DNATokenizer),
    "dnamultitransaltfull": (BertConfig, BertForSequenceMultiClassificationMultiTransformerAltFull, DNATokenizer),
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
    "lstm": (BertConfig, LSTMForSequenceClassification, DNATokenizer),
}

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"] 

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # if args.output_mode == "multi_regression":
    if args.output_mode == "multi_regression_random_sample_now":
        POS_THRESHOLD_ABS = 0.15
        class Balanced_Sampler():
            #should be moved to other part later
            def __init__(self, pos_index, neg_index, batch_size, pos_perc):
                self.pos_size = int(batch_size * pos_perc)
                self.neg_size = int(batch_size - self.pos_size)
                self.pos_index = pos_index
                self.neg_index = neg_index

            def __iter__(self):        
                pos_n_batch = len(self.pos_index) // self.pos_size
                neg_n_batch = len(self.neg_index) // self.neg_size
                
                #shuffled samples:
                batches_pos = np.random.choice(
                    self.pos_index,
                    size=(self.pos_size, pos_n_batch),
                    replace=False)

                # Indexes of batches with non-changing events.
                batches_neg = np.random.choice(
                    self.neg_index,
                    size=(self.neg_size, neg_n_batch),
                    replace=False)

                #in order samples:
                # batches_pos = []
                # for i in range(pos_n_batch):
                #     cur_index = i * self.pos_size
                #     batches_pos.append(self.pos_index[cur_index:cur_index+self.pos_size])
                # if cur_index+self.pos_size < len(self.pos_index):
                #     batches_pos.append(self.pos_index[-self.pos_size:])
                
                # batches_neg = []
                # for i in range(neg_n_batch):
                #     cur_index = i * self.neg_size
                #     batches_neg.append(self.neg_index[cur_index:cur_index+self.neg_size])
                # if cur_index+self.neg_size < len(self.neg_index):
                #     batches_neg.append(self.neg_index[-self.neg_size:])
                # batches_pos = np.array(batches_pos).T
                # batches_neg = np.array(batches_neg).T


                batches = []
                count = 0
                for xx in range(neg_n_batch):
                    if xx % pos_n_batch == 0:
                        count = 0
                    batch = np.concatenate((batches_neg[:, xx], batches_pos[:, count]))
                    np.random.shuffle(batch)
                    batches.append(batch)
                    count += 1
                return iter(batches)
            def __len__(self):
                return len(self.neg_index) // self.neg_size

        class Balanced_Sampler_category():
            #should be moved to other part later
            def __init__(self, chg_chg_dict, chg_nonchg_index, nonchg_index, batch_size, chg_perc=0.5, chg_nonchg_perc=0.25, data_size = 500000):
                self.chg_case_size = int(batch_size * chg_perc)
                self.chg_nonchg_size = int(batch_size * (1 - chg_perc) * chg_nonchg_perc)
                self.nonchg_size = batch_size - self.chg_case_size - self.chg_nonchg_size
                self.chg_chg_dict = chg_chg_dict
                self.chg_nonchg_index = chg_nonchg_index
                self.nonchg_index = nonchg_index
                self.n_batch = data_size // batch_size
            def __iter__(self):
                
                #shuffled samples:
                batches = []
                for _ in range(self.n_batch):
                    chg_batch = []
                    for _ in range(self.chg_case_size):
                        # t1 = np.random.randint(16)
                        # t2 = np.random.randint(16)
                        # up_down = np.random.randint(1,3)
                        t1 = np.random.choice(list(self.chg_chg_dict.keys()))
                        t2 = np.random.choice(list(self.chg_chg_dict[t1].keys()))
                        up_down = np.random.choice(list(self.chg_chg_dict[t1][t2].keys()))
                        chg_batch.append(np.random.choice(self.chg_chg_dict[t1][t2][up_down]))
                    chg_nonchg_batch = np.random.choice(self.chg_nonchg_index, self.chg_nonchg_size)
                    nonchg_batch = np.random.choice(self.nonchg_index, self.nonchg_size)
                    batch = np.concatenate((chg_batch, chg_nonchg_batch, nonchg_batch))
                    np.random.shuffle(batch)
                    batches.append(batch)
                
                return iter(batches)
            def __len__(self):
                return self.n_batch
        if args.sampler == 'bsc':
            # balanced sampling between different categories
            chg_chg_dict = {}
            chg_nonchg_index = []
            nonchg_index = []
            for ii in range(len(train_dataset)):
                cats = train_dataset[ii][5].tolist()
                if cats[0] == 0:
                    nonchg_index.append(ii)
                elif cats[0] == 1:
                    chg_nonchg_index.append(ii)
                else:
                    t1 = cats[1]
                    t2 = cats[2]
                    up_down = cats[3]
                    if t1 not in chg_chg_dict:
                        chg_chg_dict[t1] = {t2:{up_down:[ii]}}
                    elif t2 not in chg_chg_dict[t1]:
                        chg_chg_dict[t1][t2] = {up_down:[ii]}
                    elif up_down not in chg_chg_dict[t1][t2]:
                        chg_chg_dict[t1][t2][up_down] = [ii]
                    else:
                        chg_chg_dict[t1][t2][up_down].append(ii)
            # # print(chg_chg_dict.keys())
            # # for i in chg_chg_dict:
            # #     print(i, chg_chg_dict[i].keys(), len(chg_chg_dict[i].keys()))
            # # quit()
            train_sampler = Balanced_Sampler_category(chg_chg_dict, chg_nonchg_index, nonchg_index, args.train_batch_size, chg_perc=0.5, chg_nonchg_perc=0.25, data_size = 500000)
            train_dataset = TensorDataset(*train_dataset[:][:-1])
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
        elif args.sampler == 'bs':
            # used with Balanced_Sampler()
            pos_index = []
            neg_index = []
            for ii in range(len(train_dataset)):
                labels = train_dataset[ii][3]
                if labels[1] < POS_THRESHOLD_ABS and labels[2] < POS_THRESHOLD_ABS:
                    neg_index.append(ii)
                else:
                    pos_index.append(ii)
            train_sampler = Balanced_Sampler(pos_index, neg_index, args.train_batch_size, 0.5)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
        elif args.sampler == 'ss':
            # train_dataset = TensorDataset(*train_dataset[:][:-1])
            train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        else:
            raise ValueError("Error with wrong sampler name:{}".format(args.sampler))
    else:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if "bert" not in n and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {
            "params": [p for n, p in model.named_parameters() if "bert" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": 0.00001,
        },
        {"params": [p for n, p in model.named_parameters() if "bert" in n and any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": 0.00001},

        # {
        #     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #     "weight_decay": args.weight_decay,
        # },
        # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    # print([t.shape for a in optimizer_grouped_parameters for t in a['params'] if 4111 in t.shape])
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # quit()
    
    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = 0
    last_auc = 0
    stop_count = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            if args.task_name == "dnacass_pokedex":
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6], "tissue_rep": batch[7]}
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6]}
            
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            #print(inputs["input_ids"].shape, inputs["attention_mask"].shape, inputs["labels"], inputs["token_type_ids"])
            #print(args.model_type)
            #quit()
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, global_step = global_step)


                        if args.task_name == "dna690":
                            # record the best auc
                            if results["auc"] > best_auc:
                                best_auc = results["auc"]

                        if args.early_stop != 0:
                            # record current auc to perform early stop
                            if results["auc"] < last_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_auc = results["auc"]
                            
                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step


                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.task_name == "dna690" and results["auc"] < best_auc:
                        continue
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", evaluate=True, global_step = 0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)
        

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.task_name == "dnacass_pokedex":
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6], "tissue_rep": batch[7]}
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6]}
            
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                #print(batch[0].shape, batch[1].shape, batch[3].shape)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                # tmp_eval_loss, logits, ori_psis = outputs[:3]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        

        print("args.output_mode",args.output_mode)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "multi_regression":
            # preds = preds
            preds = torch.sigmoid(torch.tensor(preds))
        # print("sample result from evaluation set prediction:\n")
        print(preds[:1000:100])
        print(out_label_ids[:1000:100])
        # print(ori_psis[:1000:100])
        # print(preds[:10])
        # print(out_label_ids[:10])
        # print(ori_psis[:10])
        if args.do_ensemble_pred:
            result = compute_metrics(preds, out_label_ids)
        else:
            result = compute_metrics(preds, out_label_ids)
        result['loss'] = eval_loss
        results.update(result)
        
        if args.task_name == "dna690":
            eval_output_dir = args.result_dir
            if not os.path.exists(args.result_dir): 
                os.makedirs(args.result_dir)
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

        if global_step !=0:
            output_pred_file = os.path.join(eval_output_dir, "pred_results-{}.npy".format(global_step))
            np.save(output_pred_file, preds)
            
        if not os.path.isfile(output_eval_file):
            with open(output_eval_file, "w") as writer:
                writer.write('name ' + ' '.join(result.keys()) + '\n')

        with open(output_eval_file, "a") as writer:

            if args.task_name[:3] == "dna":
                eval_result = args.data_dir.split('/')[-1] + " "
            else:
                eval_result = ""

            logger.info("***** Eval results {} *****".format(prefix))
            header = " " * len(eval_result) + " ".join(list(sorted(result.keys())))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                eval_result = eval_result + str(result[key])[:5] + " "
            writer.write(eval_result + "\n")

    if args.do_ensemble_pred:
        return results, eval_task, preds, out_label_ids, probs
    else:
        return results



def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    output_rep = False
    representations = None


    # import pickle as pkl
    # temp_name_weight = {}
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, torch.norm(param)/torch.numel(param),torch.topk(param.flatten(),3).values, torch.topk(param.flatten(),3,largest=False).values, param.shape)
    #         temp_name_weight[name] = param
    # with open("weight_matrix-4000.pkl", 'wb') as w:
    #     pkl.dump(temp_name_weight, w)
    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

        if pred_dataset == -1:
            print(f"Empty file: {args.data_dir}")
            continue
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.task_name == "dnacass_pokedex":
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6], "tissue_rep": batch[7]}
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "consval": batch[4], "features": batch[6]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, logits = outputs[:2]
                if output_rep:
                    representation = outputs[2].detach().cpu().numpy()
                    if representations is None:
                        representations = representation
                    else:
                        np.append(representations, representation, axis=0)
                        

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "multi_regression":
            preds = torch.sigmoid(torch.tensor(preds))
            probs = preds

        if args.do_ensemble_pred:
            result = compute_metrics(preds, out_label_ids)
        else:
            result = compute_metrics(preds, out_label_ids)
        
        pred_output_dir = args.predict_dir
        if not os.path.exists(pred_output_dir):
               os.makedir(pred_output_dir)
        output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")
        logger.info("***** Pred results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        np.save(output_pred_file, probs)
        if output_rep:
            representations = np.array(representations)
            print(representations.shape)
            np.save(os.path.join(pred_output_dir, "representations.npy"), representations)




def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed).unsqueeze(0)


def visualize(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
            
            
        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                attention = outputs[-1][-1]
                _, logits = outputs[:2]

                
                preds[index*batch_size:index*batch_size+len(batch[0]),:] = logits.detach().cpu().numpy()
                attention_scores[index*batch_size:index*batch_size+len(batch[0]),:,:,:] = attention.cpu().numpy()
                # if preds is None:
                #     preds = logits.detach().cpu().numpy()
                # else:
                #     preds = np.concatenate((preds, logits.detach().cpu().numpy()), axis=0)

                # if attention_scores is not None:
                #     attention_scores = np.concatenate((attention_scores, attention.cpu().numpy()), 0)
                # else:
                #     attention_scores = attention.cpu().numpy()
        
        if args.task_name != "dnasplice":
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
        else:
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()

        scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

        for index, attention_score in enumerate(attention_scores):
            attn_score = []
            for i in range(1, attention_score.shape[-1]-kmer+2):
                attn_score.append(float(attention_score[:,0,i].sum()))

            for i in range(len(attn_score)-1):
                if attn_score[i+1] == 0:
                    attn_score[i] = 0
                    break

            # attn_score[0] = 0    
            counts = np.zeros([len(attn_score)+kmer-1])
            real_scores = np.zeros([len(attn_score)+kmer-1])
            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i+j] += 1.0
                    real_scores[i+j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)
            
        
            # print(index)
            # print(real_scores)
            # print(len(real_scores))

            scores[index] = real_scores
        

    return scores, probs



def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if args.do_predict:
        if os.path.isfile(args.data_dir):
            cached_features_file = os.path.join(
            '/'.join(args.data_dir.split('/')[:-1]),
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                str(args.max_seq_length),
                str(task),
            ),
        )
        else:
            cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                str(args.max_seq_length),
                str(task),
            ),
        )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )

        
        print("finish loading examples")
        
        if len(examples) == 0:
            return -1
        print("size of examples:", len(examples),examples[0])

        

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0


        if args.n_process == 1:
            features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_length,
            output_mode=output_mode,
            pad_on_left=pad_on_left,  # pad on the left for xlnet
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,)
                
        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc/4),1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples)/n_proc)
            for i in range(1, n_proc+1):
                if i != n_proc:
                    indexes.append(len_slice*(i))
                else:
                    indexes.append(len(examples))
           
            results = []
            
            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(examples[indexes[i]:indexes[i+1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
                print(str(i+1) + ' processor started !')
            
            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if output_mode == "multi_regression":
        all_input_ids = torch.tensor([[f.input_ids for f in x] for x in features], dtype=torch.long)
        all_attention_mask = torch.tensor([[f.attention_mask for f in x] for x in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([[f.token_type_ids for f in x] for x in features], dtype=torch.long)
        all_consval = torch.tensor([[f.consval for f in x] for x in features], dtype=torch.long)
        all_cat = torch.tensor([f[0].cat for f in features], dtype=torch.long)
        all_features = torch.tensor([f[0].features for f in features], dtype=torch.long)
        all_tissue_reps = torch.tensor([f[0].tissue_rep for f in features], dtype=torch.float) if features[0][0].tissue_rep else None

    else:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    elif output_mode == "multi_regression":
        all_labels = torch.tensor([f[0].label for f in features])

    if output_mode == "multi_regression":
        if task == "dnacass_pokedex":
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_consval, all_cat, all_features,all_tissue_reps)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_consval, all_cat, all_features)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--n_process",
        default=2,
        type=int,
        help="number of processes used for data process",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    
    # Other parameters
    parser.add_argument(
        "--visualize_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        help="The directory where the dna690 and mouse will save results.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--predict_dir",
        default=None,
        type=str,
        help="The output directory of predicted result. (when do_predict)",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")
    parser.add_argument("--visualize_train", action="store_true", help="Whether to visualize train.tsv or dev.tsv.")
    parser.add_argument("--do_ensemble_pred", action="store_true", help="Whether to do ensemble prediction with kmer 3456.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",
    )
    parser.add_argument(
        "--early_stop", default=0, type=int, help="set this to a positive integet if you want to perfrom early stop. The model will stop \
                                                    if the auc keep decreasing early_stop times",
    )
    parser.add_argument(
        "--predict_scan_size",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn_dropout", default=0.0, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--fc_hidden", default=768, type=int, help="Number of hidden unit in a fully connected layer.")
    parser.add_argument("--sampler", default="bsc", type=str, help="Sampler to use for sampling batches")




    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--visualize_models", type=int, default=None, help="The model used to do visualization. If None, use 3456.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--consval",
        action="store_true",
        help="Whether to use conservation values as input.",
    )

    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Whether to use features values as input.",
    )

    parser.add_argument(
        "--pokedex_rep",
        action="store_true",
        help="Whether to use pokedex representation for tissues.",
    )

    parser.add_argument(
        "--tf_mode",
        default=None,
        type=str,
        help="tensorflow model type",
    )

    parser.add_argument(
        "--embedding_method",
        default=None,
        type=str,
        help="Transformer embedding method. If None or 'RoPE', RoPE embedding will be used. Otherwise, absolute positional embedding will be used.",
    )

    parser.add_argument(
        "--rotary_value",
        default=False,
        action="store_true",
        help="Whether to apply rotary embedding for value vector.",
    )
    

    parser.add_argument(
        "--multi_weight_dpsi",
        default=None,
        type=int,
        help="higher weights on dpsi items in loss function",
    )

    parser.add_argument(
        "--continous_pred",
        action="store_true",
        help="Whether to predict multiple files without reloading the model",
    )
    args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    logger.info("  Embedding Method = %s", args.embedding_method)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if not args.do_visualize and not args.do_ensemble_pred:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        config.split = int(args.max_seq_length/512)
        config.rnn = args.rnn
        config.num_rnn_layer = args.num_rnn_layer
        config.rnn_dropout = args.rnn_dropout
        config.rnn_hidden = args.rnn_hidden
        config.consval = args.consval
        config.use_features = args.use_features
        config.pokedex_rep = args.pokedex_rep
        config.fc_hidden = args.fc_hidden
        config.embedding_method = args.embedding_method

        config.model_type = args.model_type

        config.rotary_value = args.rotary_value

        config.tf_mode = args.tf_mode
        config.multi_weight_dpsi = args.multi_weight_dpsi
        config.kmer = int(args.tokenizer_name[-1]) 
        #tokenizer = tokenizer_class.from_pretrained(
        #    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        #    do_lower_case=args.do_lower_case,
        #    cache_dir=args.cache_dir if args.cache_dir else None,
        #)
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path if args.model_name_or_path else args.tokenizer_name,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        if args.model_name_or_path and args.model_type != "lstm":
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            logger.info('finish loading model')
        else:
            logger.info("Training new model from scratch")
            model = model_class(config=config)


        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.task_name != "dna690":
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        logger.info("Predict using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        if args.continous_pred and os.path.isdir(args.data_dir):
            file_list = os.listdir(args.data_dir)
            tmp_data_dir = args.data_dir
            tmp_predict_dir = args.predict_dir
            for filename in file_list:
                if "cached" in filename or not filename.endswith(".tsv"):
                    continue
                if filename.split('_')[0].isdigit():
                    filenum = filename.split('_')[0]
                else:
                    filenum = filename.split('.')[0].split('_')[-1]
                args.data_dir = os.path.join(tmp_data_dir,filename)
                args.predict_dir = os.path.join(tmp_predict_dir,filenum)
                # if os.path.exists(args.predict_dir): #to avoid rerun things. remove later
                #     continue
                predict(args, model, tokenizer, prefix=prefix)
        else:
            prediction = predict(args, model, tokenizer, prefix=prefix)

    # Visualize
    if args.do_visualize and args.local_rank in [-1, 0]:
        visualization_models = [3,4,5,6] if not args.visualize_models else [args.visualize_models]

        scores = None
        all_probs = None

        for kmer in visualization_models:
            output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
            #checkpoint_name = os.listdir(output_dir)[0]
            #output_dir = os.path.join(output_dir, checkpoint_name)
            
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                checkpoint,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            attention_scores, probs = visualize(args, model, tokenizer, prefix=prefix, kmer=kmer)
            if scores is not None:
                all_probs += probs
                scores += attention_scores
            else:
                all_probs = deepcopy(probs)
                scores = deepcopy(attention_scores)

        all_probs = all_probs/float(len(visualization_models))
        np.save(os.path.join(args.predict_dir, "atten.npy"), scores)
        np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # ensemble prediction
    if args.do_ensemble_pred and args.local_rank in [-1, 0]:

        for kmer in range(3,7):
            output_dir = os.path.join(args.output_dir, str(kmer))
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            if kmer == 3:
                args.data_dir = os.path.join(args.data_dir, str(kmer))
            else:
                args.data_dir = args.data_dir.replace("/"+str(kmer-1), "/"+str(kmer))

            if args.result_dir.split('/')[-1] == "test.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix)
            elif args.result_dir.split('/')[-1] == "train.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False)
            else:
                raise ValueError("file name in result_dir should be either test.npy or train.npy")

            if kmer == 3:
                all_probs = deepcopy(probs)
                cat_probs = deepcopy(probs)
            else:
                all_probs += probs
                cat_probs = np.concatenate((cat_probs, probs), axis=1)
            print(cat_probs[0])
        

        all_probs = all_probs / 4.0
        all_preds = np.argmax(all_probs, axis=1)
        
        # save label and data for stuck ensemble
        labels = np.array(out_label_ids)
        labels = labels.reshape(labels.shape[0],1)
        data = np.concatenate((cat_probs, labels), axis=1)
        random.shuffle(data)
        root_path = args.result_dir.replace(args.result_dir.split('/')[-1],'')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # data_path = os.path.join(root_path, "data")
        # pred_path = os.path.join(root_path, "pred")
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)
        # if not os.path.exists(pred_path):
        #     os.makedirs(pred_path)
        # np.save(os.path.join(data_path, args.result_dir.split('/')[-1]), data)
        # np.save(os.path.join(pred_path, "pred_results.npy", all_probs[:,1]))
        np.save(args.result_dir, data)
        ensemble_results = compute_metrics(all_preds, out_label_ids)
        logger.info("***** Ensemble results {} *****".format(prefix))
        for key in sorted(ensemble_results.keys()):
            logger.info("  %s = %s", key, str(ensemble_results[key]))    


            


    return results


if __name__ == "__main__":
    main()

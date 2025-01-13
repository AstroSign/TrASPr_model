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
""" GLUE processors and helpers """

import logging
import os
from itertools import cycle, islice
from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))


        if isinstance(example, InputExample):
            example = [example]
        temp = []

        
        for each_example in example:

            

            inputs = tokenizer.encode_plus(each_example.text_a, each_example.text_b, add_special_tokens=True, max_length=max_length,)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            consval = each_example.consval
            if each_example.cat:
                cat = [int(x) for x in each_example.cat.split(',')]
            else:
                cat = [-1]
            if each_example.features:
                example_features = each_example.features
            else:
                example_features = [-1]
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                if consval is not None:
                    consval = [float(x) for x in consval.split(' ')]
                    consval = [0] * (max_length - len(consval)) + consval
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                if consval is not None:
                    if consval == '':
                        print(each_example.text_a)
                        quit()
                    try:
                        consval = [float(x) for x in consval.split(' ')]
                        consval = consval + [0] * (max_length - len(consval))
                    except:
                        print(each_example, consval)
                    
                    
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            
            #pad tissue rep
            tissue_rep = each_example.tissue_rep
            
            # if tissue_rep is not None:
            #     hidden_size = 768
            #     #repeat padding
            #     # tissue_rep = list(islice(cycle(tissue_rep), hidden_size))
                
            #     # padding with zeros
            #     tissue_rep = tissue_rep + [0] * (hidden_size - len(tissue_rep))

                # attention_mask.extend([1 if mask_padding_with_zero else 0] * 1)

                # padding with zeros separate tissues
                # for i in range(len(tissue_rep)):
                #     tissue_rep[i] = tissue_rep[i] + [0] * (hidden_size - len(tissue_rep[i]))

                #     attention_mask.extend([1 if mask_padding_with_zero else 0] * 1)


            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length or len(attention_mask) == max_length and tissue_rep is not None, "Error with input length {} vs {}".format(
                len(attention_mask), max_length
            )
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), max_length
            )
            assert len(consval) == max_length, "Error with input length {} vs {},{}".format(
                len(consval), max_length,consval
            )

            if output_mode == "classification":
                label = label_map[each_example.label]
            elif output_mode == "regression":
                label = float(each_example.label)
            elif output_mode == "multi_regression":
                label = [float(i) for i in each_example.label.split(',')]
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (each_example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                if consval:
                    logger.info("consval: %s" % " ".join([str(x) for x in consval]))
                if cat:
                    logger.info("cat: %s" % " ".join([str(x) for x in cat]))
                if example_features:
                    logger.info("example_features: %s" % " ".join([str(x) for x in example_features]))
                if tissue_rep is not None:
                    logger.info("tissue_rep: %s" % " ".join([str(x) for x in tissue_rep]))
                logger.info("label: %s (id = %s)" % (each_example.label, str(label)))
                

            temp.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label, consval=consval, cat=cat, features=example_features,tissue_rep=tissue_rep
                )
            )
        if len(temp) == 1:
            features.append(temp[0])
        else:
            features.append(temp)
    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                        "consval": ex.consval,
                        "cat": ex.cat,
                        "features": example_features,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32, "consval": tf.int32, "cat": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "consval": tf.TensorShape([None]),
                    "cat": tf.TensorShape([None]),
                    "features": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class DnaPromProcessor(DataProcessor):
    """Processor for the DNA promoter data"""

    def get_labels(self):
        return ["0", "1"]

    def get_train_examples(self, data_dir):
        if os.path.isfile(data_dir):
            logger.info("LOOKING AT {}".format(data_dir))
            return self._create_examples(self._read_tsv(data_dir), "train")
        else:
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        if os.path.isfile(data_dir):
            return self._create_examples(self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DnaCassSingleTransProcessor(DnaPromProcessor):
    """Processor for the DNA cassette exons data with single transformer"""

    def get_labels(self):
        return [None, None, None]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        mark = 0
        cat = None
        features = None
        species = None
        norm_len_feat = False
        offset = 0 
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'id_cat' in line[0]:
                    mark = 1
                elif 'id' in line[0]:
                    mark = 2
                elif 'ID' in line[0]:
                    mark = 3
                if "norm_len" in line:
                    norm_len_feat = True
                # if "species" in line:
                #     offset = 1
                continue
            if mark == 1:
                cat = line[0].split('_')[-1]  #category: chg/non-chg case, tissue1, tissu2, up/down
                line = line[1:]
            elif mark == 2:
                line = line[1:]
            elif mark == 3:
                features = [int(x) for x in line[0].split('_')[5]]  #coding features: denove, coding, noncoding, start-coding, end-coding, mix, non-coding gene
                line = line[1:]
            
            temp = []
            if line[4] in ["HUMAN", "MOUSE"]:
                offset = 1
                species = line[4]
            tissue1 = line[4 + offset]
            tissue2 = line[5 + offset]
            tissues = tissue1 + ' ' + tissue2
            label = line[-1]

            if len(line) > 11:
                length_feats = line[6 + offset:10 + offset]
                length_feat = ' '.join(length_feats[0].split(',') + length_feats[2].split(',') + length_feats[3].split(',')[1:])
                if norm_len_feat:
                    norm_len = [float(x) for x in line[10].split(",")]
                    features += norm_len
                    cons_feats = line[11 + offset:15 + offset]

                else:
                    cons_feats = line[10 + offset:14 + offset]
                for j in range(4):
                    # length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    if species:
                        text_a = line[j] + " [SEP] JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + species + ' ' + tissues
                    else:
                        text_a = line[j] + " [SEP] JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues
                    # text_a = line[j] + " [SEP] JUNC0" + str(j+1) + ' [SEP] ' + tissues
                    # text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, consval=cons_feats[j], cat=cat, features=features))
            elif len(line) > 7:
                length_feats = line[6 + offset:10 + offset]
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues
                    text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                for j in range(4):
                    guid = "%s-%s-%s" % (set_type, i, j)
                    text_a = line[j] + " [SEP] JUNC0" + str(j+1) + ' ' + tissues
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            examples.append(temp)
        return examples

class DnaCassProcessor(DnaPromProcessor):
    """Processor for the DNA cassette exons data with multi transformers"""

    def get_labels(self):
        return [None, None, None]
        # return [None, None]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        mark = 0
        cat = None
        features = None
        species = None
        norm_len_feat = False
        offset = 0 
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'id_cat' in line[0]:
                    mark = 1
                elif 'id' in line[0]:
                    mark = 2
                elif 'ID' in line[0]:
                    mark = 3
                if "norm_len" in line:
                    norm_len_feat = True
                continue
            if mark == 1:
                cat = line[0].split('_')[-1]  #category: chg/non-chg case, tissue1, tissu2, up/down
                line = line[1:]
            elif mark == 2:
                line = line[1:]
            elif mark == 3:
                features = [int(x) for x in line[0].split('_')[5]]  #coding features: denove, coding, noncoding, start-coding, end-coding, mix, non-coding gene
                line = line[1:]
            temp = []

            if line[4] in ["HUMAN", "MOUSE"]:
                offset = 1
                # species = line[4]
                species = None
            tissue1 = line[4 + offset]
            tissue2 = line[5 + offset]

            tissues = tissue1 + ' ' + tissue2
            label = line[-1]

            if len(line) > 11:
                length_feats = line[6 + offset:10 + offset]
                length_feat = ' '.join(length_feats[0].split(',') + length_feats[2].split(',') + length_feats[3].split(',')[0:1])
                if norm_len_feat:
                    # norm_len = [float(x) for x in line[10].split(",")]
                    # features += norm_len
                    cons_feats = line[11 + offset:15 + offset]
                else:
                    cons_feats = line[10 + offset:14 + offset]
                for j in range(4):
                    # length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                    if species:
                        text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + species + ' ' + tissues
                    else:
                        text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                    # text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, consval=cons_feats[j], cat=cat, features=features))

            elif len(line) > 7:
                length_feats = line[6 + offset:10 + offset]
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] "  + length_feat + ' [SEP] ' + tissues
                    text_a = length_feat + " [SEP] " + tissues + " [SEP] "  + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    text_a = line[j] + " [SEP] " + tissues
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            examples.append(temp)
        return examples

class DnaCassProcessorAlt(DnaPromProcessor):
    """Processor for the DNA cassette exons data with multi transformers"""

    def get_labels(self):
        return [None, None, None]
        # return [None, None]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        mark = 0
        cat = None
        features = None
        species = None
        norm_len_feat = False
        offset = 0 
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'id_cat' in line[0]:
                    mark = 1
                elif 'id' in line[0]:
                    mark = 2
                elif 'ID' in line[0]:
                    mark = 3
                if "norm_len" in line:
                    norm_len_feat = True
                continue
            if mark == 1:
                cat = line[0].split('_')[-1]  #category: chg/non-chg case, tissue1, tissu2, up/down
                line = line[1:]
            elif mark == 2:
                line = line[1:]
            elif mark == 3:
                features = [int(x) for x in line[0].split('_')[5]]  #coding features: denove, coding, noncoding, start-coding, end-coding, mix, non-coding gene
                line = line[1:]
            temp = []

            if line[4] in ["HUMAN", "MOUSE"]:
                offset = 1
                # species = line[4]
                species = None
            tissue1 = line[3 + offset]
            tissue2 = line[4 + offset]

            tissues = tissue1 + ' ' + tissue2
            label = line[-1]

            if len(line) > 9:
                length_feat = ' '.join(line[5].split(','))
                
                cons_feats = line[6 + offset:9 + offset]
                for j in range(3):
                    # length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                    if species:
                        text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + species + ' ' + tissues
                    else:
                        text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                    # text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, consval=cons_feats[j], cat=cat, features=features))

            elif len(line) > 7:
                length_feats = line[6 + offset:10 + offset]
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] "  + length_feat + ' [SEP] ' + tissues
                    text_a = length_feat + " [SEP] " + tissues + " [SEP] "  + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    text_a = line[j] + " [SEP] " + tissues
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            examples.append(temp)
        return examples

class DnaCassProcessorAltFull(DnaPromProcessor):
    """Processor for the DNA cassette exons data with multi transformers"""

    def get_labels(self):
        return [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        # return [None, None]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        def label_helper(labels):
            psis = []
            dpsis = []
            for label in labels.split('|'):
                label = [x for x in label.split(',')]
                psis.append(label[0])
                dpsis.extend(label[1:])
            return ','.join(psis + dpsis)

        examples = []
        mark = 0
        cat = None
        features = None
        species = None
        norm_len_feat = False
        offset = 0 
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'id_cat' in line[0]:
                    mark = 1
                elif 'id' in line[0]:
                    mark = 2
                elif 'ID' in line[0]:
                    mark = 3
                if "norm_len" in line:
                    norm_len_feat = True
                continue
            if mark == 1:
                cat = line[0].split('_')[-1]  #category: chg/non-chg case, tissue1, tissu2, up/down
                line = line[1:]
            elif mark == 2:
                line = line[1:]
            elif mark == 3:
                features = [int(x) for x in line[0].split('_')[5]]  #coding features: denove, coding, noncoding, start-coding, end-coding, mix, non-coding gene
                line = line[1:]
            temp = []

            if line[4] in ["HUMAN", "MOUSE"]:
                offset = 1
                # species = line[4]
                species = None
            tissue1 = line[6 + offset]
            tissue2 = line[7 + offset]

            tissues = tissue1 + ' ' + tissue2
            label = label_helper(line[-1])


            length_feat = ' '.join(line[8].split(','))
            
            cons_feats = line[9 + offset:15 + offset]
            for j in range(6):
                # length_feat = ' '.join(length_feats[j].split(','))
                guid = "%s-%s-%s" % (set_type, i, j)
                # text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                if species:
                    text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + species + ' ' + tissues
                else:
                    text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                # text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, consval=cons_feats[j], cat=cat, features=features))

            
            examples.append(temp)
        return examples


class DnaCassPokedexProcessor(DnaPromProcessor):
    """Processor for the DNA cassette exons data and pokedex representatin for tissues with multi transformers"""

    def get_labels(self):
        return [None, None, None]
        # return [None, None]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        mark = 0
        cat = None
        features = None
        species = None
        norm_len_feat = False
        offset = 0 
        for (i, line) in enumerate(lines):
            if i == 0:
                if 'id_cat' in line[0]:
                    mark = 1
                elif 'id' in line[0]:
                    mark = 2
                elif 'ID' in line[0]:
                    mark = 3
                if "norm_len" in line:
                    norm_len_feat = True
                continue
            if mark == 1:
                cat = line[0].split('_')[-1]  #category: chg/non-chg case, tissue1, tissu2, up/down
                line = line[1:]
            elif mark == 2:
                line = line[1:]
            elif mark == 3:
                features = [int(x) for x in line[0].split('_')[5]]  #coding features: denove, coding, noncoding, start-coding, end-coding, mix, non-coding gene
                line = line[1:]
            temp = []

            if line[4] in ["HUMAN", "MOUSE"]:
                offset = 1
                # species = line[4]
                species = None
            tissue1 = [float(x) for x in line[4 + offset].split(' ')]
            tissue2 = [float(x) for x in line[5 + offset].split(' ')]

            tissues = tissue1 + tissue2
            # tissues = [tissue1,tissue2]
            label = line[-1]

            if len(line) > 11:
                length_feats = line[6 + offset:10 + offset]
                length_feat = ' '.join(length_feats[0].split(',') + length_feats[2].split(',') + length_feats[3].split(',')[0:1])
                if norm_len_feat:
                    # norm_len = [float(x) for x in line[10].split(",")]
                    # features += norm_len
                    cons_feats = line[11 + offset:15 + offset]
                else:
                    cons_feats = line[10 + offset:14 + offset]
                for j in range(4):
                    # length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + tissues
                    if species:
                        text_a = line[j] + " [SEP] " + length_feat + ' [SEP] ' + species
                    else:
                        text_a = line[j] + " [SEP] " + length_feat
                    # text_a = "JUNC0" + str(j+1) + ' ' + length_feat + ' [SEP] ' + tissues + ' [SEP] ' + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, consval=cons_feats[j], cat=cat, features=features, tissue_rep=tissues))

            elif len(line) > 7:
                length_feats = line[6 + offset:10 + offset]
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    # text_a = line[j] + " [SEP] "  + length_feat + ' [SEP] ' + tissues
                    text_a = length_feat + " [SEP] " + tissues + " [SEP] "  + line[j]
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            else:
                for j in range(4):
                    length_feat = ' '.join(length_feats[j].split(','))
                    guid = "%s-%s-%s" % (set_type, i, j)
                    text_a = line[j] + " [SEP] " + tissues
                    temp.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            examples.append(temp)
        return examples


class DnaSpliceProcessor(DataProcessor):
    """Processor for the DNA promoter data"""

    def get_labels(self):
        return ["0", "1", "2"] 

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DnaPairProcessor(DataProcessor):
    """Processor for the DNA promoter data"""

    def get_labels(self):
        return ["0", "1"] 

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "dnaprom": 2,
    "dnacass": 3,
    "dnacasssingle": 3,
    "dnacass_alt": 3,
    "dnacass_alt_full": 18,
    "dna690":2,
    "dnapair":2,
    "dnasplice":3,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "dnaprom": DnaPromProcessor,
    "dnacass": DnaCassProcessor,
    "dnacass_alt": DnaCassProcessorAlt,
    "dnacass_alt_full": DnaCassProcessorAltFull,
    "dnacass_pokedex": DnaCassPokedexProcessor,
    "dnacasssingle": DnaCassSingleTransProcessor,
    "dna690": DnaPromProcessor,
    "dnapair": DnaPairProcessor,
    "dnasplice": DnaSpliceProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "dnaprom": "classification",
    "dnacass": "multi_regression",
    "dnacass_alt": "multi_regression",
    "dnacass_alt_full": "multi_regression",
    "dnacass_pokedex": "multi_regression",
    "dnacasssingle": "multi_regression",
    "dna690": "classification",
    "dnapair": "classification",
    "dnasplice": "classification",
}

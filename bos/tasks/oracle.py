import tempfile
import torch
from transformers import (
    BertForSequenceMultiClassificationMultiTransformer,
    DNATokenizer,
    glue_convert_examples_to_features,
)
from transformers.data.processors.glue import DnaCassProcessor

class DNABertPredictor:
    TISSUES_DICT = {
        "lung": "TISS00",
        "heart": "TISS02",
        "brain": "TISS03",
        "liver": "TISS07",
        "spleen": "TISS11",
        "cells_EBV_transformed_lymphocytes": "TISS14",
        "K562_WT": "K562",
        "HepG2_WT": "HepG2",
    }

    def __init__(self, model_path: str, max_seq_length: int = 410, seed: int = 42):
        """Initialize the DNABERT predictor with a pre-trained model.

        Args:
            model_path: Path to the pre-trained model directory
            max_seq_length: Maximum sequence length for the model
            seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DNATokenizer.from_pretrained(model_path)
        self.model = BertForSequenceMultiClassificationMultiTransformer.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def _seq2kmer(self, seq: str, k: int = 6):
        """Convert a sequence to k-mer representation."""
        kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
        n_kmers = len(kmer)
        kmers = " ".join(kmer)
        return kmers, n_kmers

    def _seq2input(self, seqs, tissue1: str, tissue2: str, k: int = 6) -> str:
        """Convert sequences to input format for DNABERT model."""
        header = "ID\tseq_E1+I1\tseq_I1+A\tseq_A+I2\tseq_I2+E2\tTissue1\tTissue2\tlength_E1+I1\tlength_I1+A\tlength_A+I2\tlength_I2+E2\tlabel"
        all_seqs_output = ""

        for four_seqs_list in seqs:
            # Then write file to be ready by oracle model 
            all_seqs_output += "\n"
            output = "13_+_modulizer_chg_nonskip_00000101_chgCase_5f3iE4L8iYQ_HepG2_QKI_WT\t"
            length_features_and_consvals = "EXLEN4,ITLEN4\tEXLEN1,ITLEN4\tEXLEN1,ITLEN5\tEXLEN3,ITLEN5\t"
            for seq in four_seqs_list:
                seq = seq.replace("-", "") # remove any '-' tokens (output by vae to create k-mers)
                seq_kmer, n_kmers = self._seq2kmer(seq, k)
                for i in range(n_kmers):
                    length_features_and_consvals += "0"
                    if i < (n_kmers - 1):
                        length_features_and_consvals += " "
                length_features_and_consvals += "\t"
                output += seq_kmer + "\t"

            end_string = f"{tissue1}\t{tissue2}\t{length_features_and_consvals}0,0,0"
            output += end_string
            all_seqs_output += output

        return header + all_seqs_output

    def _load_and_cache_examples(self, data_dir: str):
        """Load and preprocess examples from data directory."""

        processor = DnaCassProcessor()
        examples = processor.get_dev_examples(data_dir)

        if not examples:
            raise ValueError(f"No examples found in {data_dir}")

        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

        features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=processor.get_labels(),
            max_length=self.max_seq_length,
            output_mode="multi_regression",
            pad_on_left=False,
            pad_token=pad_token,
            pad_token_segment_id=0,
        )

        tensors = {
            "input_ids": torch.tensor([[f.input_ids for f in x] for x in features], dtype=torch.long),
            "attention_mask": torch.tensor([[f.attention_mask for f in x] for x in features], dtype=torch.long),
            "consval": torch.tensor([[f.consval for f in x] for x in features], dtype=torch.long),
            "features": torch.tensor([f[0].features for f in features], dtype=torch.long),
            "labels": torch.tensor([f[0].label for f in features]),
        }

        return {name: tensor.to(self.device) for name, tensor in tensors.items()}

    @torch.inference_mode()
    def predict(self, seqs, tissue: str = "brain"):
        """
        seqs: list of tuples of 4 sequences
        """
        if tissue not in self.TISSUES_DICT:
            raise ValueError(f"Invalid tissue. Must be one of: {list(self.TISSUES_DICT.keys())}")

        tissue_id = self.TISSUES_DICT[tissue]

        with tempfile.NamedTemporaryFile("w") as temp_data_dir:
            input_data = self._seq2input(seqs=seqs, tissue1=tissue_id, tissue2=tissue_id)
            temp_data_dir.write(input_data)
            temp_data_dir.flush()

            inputs = self._load_and_cache_examples(temp_data_dir.name)

            inputs["token_type_ids"] = None
            outputs = self.model(**inputs)
            _, logits = outputs[:2]

            psis = torch.sigmoid(logits).cpu().numpy()
            return psis[:, 0].tolist()  # type: ignore


if __name__ == "__main__":
    model_path = "../tasks/best_gtex_checkpoint"
    bert = DNABertPredictor(model_path=model_path)
    seqs1 = [
        "GCAAATCTGTCCAAGGCTGCCTGGGGAGCATTGGAGAAGAATGGCACCCAGCTGATGATCCGCTCCTACGAGCTCGGGGTCCTTTTCCTCCCTTCAGCATTTGTAAGTTTACACTTCCAACTGCACAGTGGGTCACATGGGAGATGAATTGGGCTTTGTGTTCTCTTCCTTTAGTCACAACAGCTTGCCTAAGGTAGGCAGGAA",
        "TTTATTTAGGTACCATAAAGATATTATTGCTTCTTGAGGCCTTCTGGTTTTTTTTTTTAAACCCAGAAAATGGGGTGCAATCATAGATTTCCTCTGTGTGTCTCCGTATGCTAGACAGTTTCAAAGTGAAACAGAAGTTCTTCGCTGGCAGCCAGGAGCCAATGGCCACCTTTCCTGTGCCATATGATTTGCCTCCAGAACTGTATGGAAGTAAAG",
        "GTCTAGACAGTTTCAAAGTGAAACAGAAGTTCTTCGCTGGCAGCCAGGAGCCAATGGCCACCTTTCCTGTGCCATATGATTTGCCTCCAGAACTGTATGGAAGTAAAGGTGAGACACAGATAAAGGAAAACCACGGGTGGATATGCATAAGAAAAACAAACAGAGCCCAGGAGAAGCCTTTGGTCTCAGTGGGATCCTGGCTCATGGAACCCTCTG",
        "TTCACTCACCTAGTGTCCTAATAGGAGCCTTCGCTAAAAGCTGTTGTCTGAAGGGGGAGAATCTTCTGGGCTTGGTTCAGAAAATACTCTACCACAGTTTAGAGAGTCAAGCACATAAGTGTTTTTATGCCATCAGGTGAGTGCATTTTCTACCATCCTATTCAAATAAATATTACGTAATGTGTTTTTCCCCCAGATCGGCCATGGATATGGAACATTCCTTATGTCAAAGCACCGGATACGCATGGGAACATGTGGGTGCCCTCCTGAGAATCTTGAGGCACTGTGAAATTTAAGTGTAAGACATTGAGCCACAAACATGGAATCTCTTCTTTGTACTGGATGTCCACTTCCCTTAAAGTCTTATTTGCACCCTTACAAAATCTTTCCAAAG",
    ]
    seqs2 = [
        "GGTACAGTCAAGGGGAGCCCTTCTCCCACAGGAGGGCATTGGGGTGTGGGGCCTGGGGCACTGGTTCAGGTACCGCCTTATCCCGGGCCAGGAATGAGCTCAGTGACCACTTGGATGCTATGGACTCCAACCTGGATAACCTGCAGACCATGCTGAGCAGCCACGGCTTCAGCGTGGACACCAGTGCCCTGCTGGACGTGAGTGGAGCCCCGCCGCCCCGCCTCCCCGCCCCGCCTCCCCGCCGCGCCGCCCCGCCTCCCCGCCCCGCCTCCCCGCCTCCCCGCGCCGCCGCCCCGCCTCCCCGCCCCGCCTCCCCGCGCCTCCCCGCCTCCCCGCCCCGCCTCCCCGCCTCCCCGCCCCGCCTCCCCGCCCCGCCTCCCCGCCCCGCCCCCGG",
        "CTCCCCGCCCCGCCTCCCCGCCCCGCCCCCGGGTGCTGTTCTGACTTCCCTCCCTCCTCCGTCCCTCTTCAGCCCCTCGGTGACCGTGCCCGACATGAGCCTGCCTGACCTTGACAGCAGCCTGGCCAGT",
        "TGTTCAGCCCCTCGGTGACCGTGCCCGACATGAGCCTGCCTGACCTTGACAGCAGCCTGGCCAGTGTGCGTAGGCGGGCGGGGGGTGAGGGGGAACGAGACCAGCGGGAGTGCTCACAATACCGTCTCCA",
        "TAGGCGGGCGGGGGGTGAGGGGGAACGAGACCAGCGGGAGTGCTCACAATACCGTCTCCACCCCACAGATCCAAGAGCTCCTGTCTCCCCAGGAGCCCCCCAGGCCTCCCGAGGCAGAGAACAGCAGCCCGGATTCAG",
    ]
    psis_brain = bert.predict(seqs=[seqs1, seqs2], tissue="brain")
    psis_heart = bert.predict(seqs=[seqs1, seqs2], tissue="heart")
    print(f"Psi brain values for two example inputs: {psis_brain}")
    print(f"Psi heart values for two example inputs: {psis_heart}")
    # Output expected: 
    # Psi brain values for two example inputs: [0.38091161847114563, 0.3480629026889801]
    # Psi heart values for two example inputs: [0.4863601326942444, 0.6968383193016052]

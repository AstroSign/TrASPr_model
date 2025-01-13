## Data Description

In this text, we will include the details of each data file and the meaning of columns. Repeat columns will only be introduced at the first time it appearing.

1. GTEx_data.tsv: includes test data from GTEx dataset for chromosomes 1,3,5,7,9. 
    - ID: Unique hash ID for each splicing event.	
    - Chr: Chromosome of the event.
    - Strand: Strand of the event.
    - Exon_start, Exon_end: the start and end coordinates of the middle exon in each cassettte exon event.
    - Tissue: In which tissue the event is measured/predicted.
    - Change_case: If it is a changing event(dPSI between at least one tissue pair is larger than 0.15).
    - Label: PSI label for corresponding case.
    - TrASPr_pred, Pangolin_pred, SpliceAI_pred, SpliceTF_pred: Prediction results for 4 models compared in the paper: TrASPr, Pangolin, SpliceAI and SpliceTransformer.

2. GTEx_TrASPr_dPSI_preds.tsv: include tissue pair prediction results of TrASPr for tissue specific comparison. In this file, dPSI+ and dPSI- will be included in Label and TrASPr_pred.


3. ENCODE_data_RBPAE.tsv: TrASPr prediction results with token/RBP-AE as tissue representation for training on GTEx and testing ENCODE dataset. 
    - SS1_pos, SS2_pos, SS3_pos, SS4_pos: Splice site coordiantes of cassette exon events.
    - AE_pred: prediction results with RBP-AE as the tissue representation.
    - TrASPr_pred: prediction results with one-hot token as the tissue representation.

4. alternative_5.tsv / alternative_5.tsv: TrASPr prediction results for alternative splice site dataset. 

5. TrASPr_swap_ss_results.tsv: TrASPr prediction results for swap splice site experiment. 
    - dScores: delta MaxEnt score for splice sites before and after the swap.
    - TrASPr_pred_dPSI: delta PSI score for the case before and after the swap. 

6. TrASPr_rbpkd_results.tsv: TrASPr prediction results for RBP Knockdown experiment. 
    - RBP_KD: The target RBP whose motifs are mutated in the sequences.
    - Label_dPSI: dPSI before and after the motif mutations.
    - TrASPr_pred_dPSI, Pangolin_pred_dPSI, SpliceAI_pred_dPSI: prediction changes before and after the motif mutations for TrASPr, Pangolin and SpliceAI

7. CD19_mutation_pos.txt: mutation positions of CD19 dataset. The first column is the index and the rest are the coordinates of muations.

8. CD19_crossVal.tsv: Cross validation results for CD19 dataset.
    - Index: Index in the CD19 dataset

9. CD19_single_filter.tsv: Single filtering experiment results for CD19 dataset.

10. lsvseq_data.tsv: Low expression events validated by LSV-seq and also predicted by TrASPr.
    - gene_id: gene ID of the event.
    - ref_exon_start, ref_exon_end: start and end coordinates for the middle(reference) exon.
    - change_label: if the event is a changing one based on LSV-seq validation results.
    - gene_name: gene name for the event.
    - splicing_code_passed, splicing_code_passed_stringent, splicing_code_passed_very_stringent: if TrASPr predicted it as a changing event based on different thresholds.

11. disrupt_ss: folder includes generated sequences by BOS, Genetic Algorithm and Random Algorithm for disrupt splice site experiment. 
    - *.cvs: generated sequence file with predicted PSI and generated sequences.
    - *_edit_index.txt: index file includes information of editting location of each algorithm.
    - mutate_ss_seq.score: MaxEnt score for each splice site.
    - mutate_ss_seq_ids.output: include predicted PSI and changed sequences for each algorithm.

12. BOS_results_edit_index_CD19.txt: BOS generated sequences for CD19 dataset. It includes the dPSI after editting and the editting locations.

13. brain_specific_10starters: folder includes generated sequences by BOS, Genetic Algorithm and Random Algorithm for brain specific experiment with 10 different start sequences.
    - all_low_bos_samples.tsv: includes the label for each sequences which has low PSI for all tissues as starters.
    - *.cvs: generated sequenecs by three generative algorithms.

14. BOS_Daam1: folder includes generated sequences by BOS for Daam1 gene.
    - edit_index_2k.txt: editting locations of BOS.
    - minimize_psi_N2A_daam1_parental_recentered.csv: generated sequences by BOS for Daam1 gene with target function to minimize PSI.



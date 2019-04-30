# Configuration File Reference

## Example config files

```yaml
# file paths
annotation_dir: genome/hg38
genome_dir: genome/hg38
output_dir: output/cfRNA
temp_dir: tmp
data_dir: data/cfRNA

# general parameters
threads_compress: 1

# mapping parameters
aligner: bowtie2
paired_end: false
small_rna: true
count_method: [mirna_and_domains_rna, mirna_only]
batch_indices: [1]
normalization_method: ["TMM"]
batch_removal_method: ["limma", "null"]
rna_types: [univec, rRNA, lncRNA, miRNA, mRNA, piRNA, snoRNA, snRNA, srpRNA, tRNA, tucpRNA, Y_RNA]
```


## Default config files with all parameters
```yaml
# RNA types for sequential mapping in small-RNA pipeline
rna_types: [rRNA, lncRNA, miRNA, mRNA, piRNA, snoRNA, 
  snRNA, srpRNA, tRNA, tucpRNA, Y_RNA]
# Adjusted p-value threshold for defining domains
call_domain_pvalue: "05"
# Distribution to use to model read coverage in each bin
distribution: ZeroTruncatedNegativeBinomial
# Size of each bin to compute read coverage
bin_size: 20
# Define recurrent domain as domains called in fraction of samples above this value
cov_threshold: 0.2
# Method to scale features
scale_method: robust
# Classifier for feature selection
classifiers: random_forest
# Number of features to select
n_selects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
resample_method: bootstrap
# Feature selection methods
select_methods: [robust]

# Parameters for classifiers
classifier_params:
  logistic_regression:
    penalty: l2
  random_forest:
    n_estimators: 10
# Parameters for grid search for classifier parameters
grid_search:
  logistic_regression:
    C: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5]
  linear_svm:
    C: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5]
  random_forest:
    n_estimators: [25, 50, 75]
    max_depth: [3, 4, 5, 6, 7, 8]
# Splitter for cross-validation
cv_splitter: stratified_shuffle_split
# Number of cross-validation folds for grid search
grid_search_n_splits: 5
# Number of train-test splits for cross-validation
cv_n_splits: 50
# Fraction/number of test samples in cross-validation
cv_test_size: 0.2
# Compute sample weight for imbalanced classes
compute_sample_weight: true
# Fold change filter in feature selection (up, down, both)
fold_change_direction: up
# Fold change filter threshold
fold_change_threshold: 1
# Fraction of features to eliminate in each RFE step
rfe_step: 0.1
# Number of cross-validation splits in RFE
rfe_n_splits: 10
# Number of cross-validation splits in robust feature selection
robust_feature_selection_n_splits: 10
# Splitter for robust feature selection
robust_feature_selection_splitter: stratified_shuffle_split

# Number of random train-test splits for cross-validation
cross_validation_splits: 50
# Type of counts for feature selection
#   domains_combined: combine miRNA/piRNA with long RNA domains
#   transcript: transcript-level features
#   featurecounts: gene-level features counted using featureCounts
count_method: domains_combined
# Define low expression value as read counts below this value
filtercount: 10
# Keep features with high expression in fraction of samples above this value
filtersample: 0.2
# Imputation methods to try (set to "null" to skip imputation)
imputation_methods: ["scimpute_count", "viper_count", "null"]
# Read depth normalization methods to try
normalization_methods: ["SCnorm", "TMM", "RLE", "CPM", "CPM_top", "CPM_rm"]
# Batch effect removal methods to try (set "null" to skip batch effect removal)
batch_removal_methods: ["null", "Combat", "RUV"]
# Column index of batch effect in batch_info.txt to considier for Combat
batch_indices: []

# Root directory
root_dir: "."
# Directory for sequences and annotations
genome_dir: "genome/hg38"
# Temporary directory (e.g. samtools sort, sort)
temp_dir: "tmp"
# Directory for third-party tools
tools_dir: "tools"
# Directory for exSeek scripts
bin_dir: "bin"

# Number of threads for uncompression and compression
threads_compress: 1
# Default number of threads to use
threads: 1
# alignment software to use (valie choices: bowtie, star)
aligner: bowtie2
# Remove 3'-end adaptor sequence from single-end reads
adaptor: ""
# Remove 5'-end adaptor sequence from single-end reads
adaptor_5p: ""
# Remove 3'-end adaptor sequence from the first read in a pair
adaptor1: ""
# Remove 3'-end adaptor sequence from the second read in a pair
adaptor2: ""
# Remove 5'-end adaptor sequence from the first read in a pair
adaptor1_5p: ""
# Remove 5'-end adaptor sequence from the second in a pair
adaptor2_5p: ""
# Exact number of bases to trim from 5'-end
trim_5p: 0
# Exact number of bases to trim from 3'-end
trim_3p: 0
# Discard reads of length below this value
min_read_length: 16
# Maximum read length
max_read_length: 100
# Trim bases with quality below this value from 3'-end
min_base_quality: 30
# Trim bases with quality below this value from 5'-end
min_base_quality_5p: 30
# Trim bases with quality below this value from 3'-end
min_base_quality_3p: 30
# Quality encoding in FASTQ files
quality_base: 33
# Strandness (valid choices: forward, reverse, no)
strandness: forward
# Filter out reads with mapping quality below this value
min_mapping_quality: 0
# Only considier longest transcript for transcriptome mapping
use_longest_transcript: true
# Expected read length for mapping using STAR
star_sjdboverhang: 100
# Number of threads for mapping
threads_mapping: 4
# Remove duplicates for long RNA-seq before feature counting
remove_duplicates_long: false
# Input reads are paired-end
paired_end: false
# Use small RNA-seq pipeline (sequential mapping)
small_rna: true
# Remove UMI tags (leading nucleotides)
umi_tags: false
# Length of the UMI barcode
umi_length: 0
# Evaluate published biomarkers
evaluate_features_preprocess_methods: []
# Differential expression method
diffexp_method: deseq2
# Count multi-mapping reads
count_multimap_reads: true
# Count overlapping features
count_overlapping_features: true
```


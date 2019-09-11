# Introduction

**A bioinformatics tool for exRNA biomarker discovery**

[![Build Status](https://travis-ci.com/lulab/exSeek-dev.svg?token=CyRgUWsqWCctKvAxMXto&branch=master)](https://travis-ci.com/lulab/exSeek-dev)

## Table of Contents

- [Installation](#installation)
- [Usage](#basic-usage-of-exseek)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Copyright and License Information](#copyright-and-license-information)


## Installation

For easy installation, you can use the docker image we provide: [exSEEK Docker Image](https://hub.docker.com/r/ltbyshi/exseek)

Alternatively, you can use use singularity or udocker to run the container for Linux kernel < 3 or if you don't have permission to use docker.


## Usage

Run `exseek.py --help` to get basic usage:

```text
usage: exseek.py [-h] --dataset DATASET [--config-dir CONFIG_DIR] [--cluster]
                 [--cluster-config CLUSTER_CONFIG]
                 [--cluster-command CLUSTER_COMMAND]
                 [--singularity SINGULARITY]
                 [--singularity-wrapper-dir SINGULARITY_WRAPPER_DIR]
                 {quality_control,prepare_genome,mapping,count_matrix,call_domains,normalization,feature_selection,update_sequential_mapping,update_singularity_wrappers}

exSeek main program

positional arguments:
  {quality_control,quality_control_clean,cutadapt,rename_fastq,fastq_to_fasta,prepare_genome,bigwig,
  mapping,count_matrix,call_domains,merge_domains,combine_domains,normalization,feature_selection,
  differential_expression,evaluate_features,igv,update_sequential_mapping,update_singularity_wrappers}

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        dataset name
  --config-dir CONFIG_DIR, -c CONFIG_DIR
                        directory for configuration files
  --cluster             submit to cluster
  --cluster-config CLUSTER_CONFIG
                        cluster configuration file ({config_dir}/cluster.yaml
                        by default)
  --cluster-command CLUSTER_COMMAND
                        command for submitting job to cluster (default read
                        from {config_dir}/cluster_command.txt
  --singularity SINGULARITY
                        singularity image file
  --singularity-wrapper-dir SINGULARITY_WRAPPER_DIR
                        directory for singularity wrappers
```


The [main program of exSEEK](https://github.com/lulab/exSEEK/tree/master/exSEEK) starts from a data matrix of gene expression (read counts of each gene in each sample). Meanwhile, we provide some pipelines and QC steps for the [pre-process](https://github.com/lulab/exSEEK/tree/master/pre-process) of exRNA-seq (including long and short  cfRNA-seq/exoRNA-seq) raw data. You can decide where to begin and run the related commands.


For detailed commands instruction and introduction, please check:
- [Preprocess](pre-process)
  - [genome and annotations](pre-process/genome_and_annotations.md)
  - [small RNA-seq mapping](pre-process/small_rna_mapping.md)
  - [long RNA-seq mapping](pre-process/long_rna_mapping.md)
- [exSEEK](exseek)
  - [config file](exseek/configuration.md)
  - [matrix processing](exseek/matrix_processing.md)
  - [feature selection](exseek/feature_selection.md)
  - [cluster configuration](exseek/cluster_configuration.md)
  



> **Note**
> We also recommend other alternatives for the pre-process, such as [exceRpt](https://github.com/gersteinlab/exceRpt), that is specifically developed for the process of exRNA-seq raw reads.
> * Other arguments are passed to *snakemake*
> * Specify number of processes to run in parallel with *-j*


## Frequently Asked Questions

[FAQs](https://github.com/lulab/exSEEK_docs/tree/dd93c0deb8978e7aa0276d6fdf40ae288e5d42fa/FAQ.md)


## Copyright and License Information
Copyright (C) 2019 Tsinghua University, Beijing, China 

Authors: Binbin Shi, Xupeng Chen, Jingyi Cao and Zhi John Lu 

This program is licensed with commercial restriction use license. Please see the [LICENSE](https://github.com/lulab/exSEEK_docs/blob/master/LICENSE) file for details.

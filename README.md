# Introduction

**A bioinformatics tool for exRNA biomarker discovery**

[![Build Status](https://travis-ci.com/lulab/exSeek-dev.svg?token=CyRgUWsqWCctKvAxMXto&branch=master)](https://travis-ci.com/lulab/exSeek-dev)

## Table of Contents

- [Workflow](#Workflow)
- [Installation](#installation)
- [Usage](#basic-usage-of-exseek)
- [Contact](#contact)
- [Frequently asked Questions](#frequently-asked-questions)
- [Copyright and License Information](#copyright-and-license-information)

## Workflow

![workflow](.gitbook/assets/whole_pipe.png)

## Installation

For easy intallation, you can use the docker image we provide: [exSEEK Docker Image](https://hub.docker.com/r/ltbyshi/exseek)

If you would like to manually set up the environment, please follow the [requirements](https://exseek.gitbook.io/docs/installation)


## exSEEK modules

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


The [main program of exSEEK](https://github.com/lulab/exSEEK/tree/master/exSEEK) starts from a data matrix of gene expression (read counts of each gene in each sample). Meanwhile, we provide some pipelines and QC steps for the [pre-process](https://github.com/lulab/exSEEK/tree/master/pre-process) of exRNA-seq (including long and short  cfRNA-seq/exoRNA-seq) raw data. 

You can refer to this picture to understand the function and the processing and analyzing steps of exSEEK. Then you can decide where to begin and run the related commands.


![exSEEK modules](.gitbook/assets/exseek_modules.png)


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
> We also recommend other alternatives for the pre-process, such as [exceRpt](https://github.com/gersteinlab/exceRpt) and ?, that are specifically developed for the process of exRNA-seq raw reads.
> * Other arguments are passed to *snakemake*
> * Specify number of processes to run in parallel with *-j*


## Frequently asked Questions

[FAQs](https://github.com/lulab/exSEEK_docs/tree/dd93c0deb8978e7aa0276d6fdf40ae288e5d42fa/FAQ.md)


## Copyright and License Information
Copyright (C) 2019 Tsinghua University, Beijing, China 

Authors: Binbin Shi, Jingyi Cao, Xupeng Chen, and Zhi John Lu 

This program is licensed with commercial restriction use license. Please see the [LICENSE](http://exseek.ncrnalab.org/LICEN) file for details.

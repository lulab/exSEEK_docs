# Installation

- exSEEK contains a main program exseek.py which wraps several snakemake scripts. It relies on many softwares and environment. The easiest way to use exSEEK is downloading the docker image we provide: [exSEEK Docker Image](). 

- You can also follow the instructions to manually set up the environment:

## Install exSEEK

```bash
git clone https://github.com/lulab/exSeek-dev.git
```

## Install required software

* Python 3.6 \(miniconda\)
* Python 2.7 \(miniconda\)
* Java 8
* R 3.4 \([https://mran.microsoft.com/download](https://mran.microsoft.com/download)\)

### Configure conda channels

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/mro/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
```

### Install Python packages using conda

```bash
conda install -y numpy scipy scikit-learn 'openssl<1.1'
conda install -y pandas matplotlib seaborn
conda install -y tqdm snakemake h5py bokeh
conda install -y umap jinja2
pip install scikit-rebate
pip install mlxtend 
pip install flask Flask-AutoIndex
pip install multiqc
```

### Install Bioconda packages

List of all available Bioconda packages: \([https://bioconda.github.io/recipes.html](https://bioconda.github.io/recipes.html)\)

```bash
conda install -c bioconda -y bedtools samtools star subread bowtie2
conda install -c bioconda -y rsem bamtools cutadapt picard gffread gffcompare
conda install -c bioconda -y ucsc-bedtogenepred ucsc-genepredtogtf ucsc-bedgraphtobigwig ucsc-bigwigtobedgraph
conda install -c bioconda -y htseq fastx_toolkit biopython
conda install -c bioconda -y flexbar
```

### Install Ubuntu packages

```bash
sudo apt-get install -y gzip pigz openjdk-8-jdk libgraphviz-dev uuid-dev zlib1g-dev libpng-dev gawk
```

### Install R packages

Install by running the following code in an R interactive session:

```r
options("repos" = c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(BioC_mirror="https://mirrors.tuna.tsinghua.edu.cn/bioconductor")
# From CRAN
install.packages(c('devtools', 'sva', 'VGAM', 'argparse', 'magrittr', 'readr', 'mvoutlier', 
    'ggpubr', 'fastqr', 'ggfortify', 'survminer', 'SIS'))
# From Bioconductor
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
#BiocManager::install("TCGAbiolinks", version = "3.8")
#source('https://bioconductor.org/biocLite.R')
BiocManager::install(c('SingleCellExperiment', 'scater', 'scran', 'SCnorm',
    'EDASeq', 'RUVSeq', 'DESeq2', 'edgeR', 'sva', 'apeglm', 'gglasso', 'ggbio', 'TCGAbiolinks'))
# From R-forge
install.packages('countreg', repos = c('http://download.r-forge.r-project.org', 'https://mirrors.tuna.tsinghua.edu.cn/CRAN/'), dep = TRUE)
# From GitHub
library(devtools)
install_github('ChenMengjie/VIPER')
install_github('kassambara/easyGgplot2')
install_github("Vivianstats/scImpute")
install_github("hemberg-lab/scRNA.seq.funcs")
install_github('theislab/kBET')
```

### Other packages

* [find\_circ 1.2](https://github.com/marvin-jens/find_circ) \(depends on Python 2.7\)
* [GTFTools](http://www.genemine.org/codes/GTFtools_0.6.5.zip) \(depends on Python\)
* [Prinseq](http://prinseq.sourceforge.net/) \(requires Perl\)

## Singularity

### Build image

```bash
singularity build singularity/exseek.img singularity/Singularity
```

### Make wrappers for singularity executables

```bash
bin/make_singularity_wrappers.py \
    --image ~/singularity/simg/exseek.simg \
    --list-file singularity/exports.txt \
    --singularity-path $(which singularity) \
    -o ~/singularity/wrappers/exseek
```

### Add wrappers to PATH

```bash
export PATH="$HOME/singularity/wrappers/exseek:$PATH"
```

## Build Pykent

```bash
wget -O tools/ucsc-tools.tar.gz http://hgdownload.soe.ucsc.edu/admin/exe/userApps.src.tgz
tar -C tools -zxf tools/ucsc-tools.tar.gz
(cd tools/userApps/kent/src/htslib/
    CFLAGS="-fPIC -DUCSC_CRAM=0 -DKNETFILE_HOOKS=1" ./configure
    make
)
(cd tools/userApps/kent/src/lib/
echo '
%.o: %.c
        ${CC} -fPIC ${COPT} ${CFLAGS} ${HG_DEFS} ${LOWELAB_DEFS} ${HG_WARN} ${HG_INC} ${XINC} -o $@ -c $<

$(MACHTYPE)/libjkweb.so: $(O) $(MACHTYPE)
    gcc -fPIC -shared -o $(MACHTYPE)/libjkweb.so $(O) -Wl,-z,defs -L../htslib -lhts -lm -lz -lpthread -lpng -lcrypto -lssl -luuid
' > makefile
make x86_64/libjkweb.so
)
cp tools/userApps/kent/src/lib/x86_64/libjkweb.so lib/
```


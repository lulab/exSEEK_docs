# Feature Selection

## Recommended machine learning packages

*scikit-learn*

Link: (https://scikit-learn.org/stable)

scikit-learn has a unified and simple interface for machine learning algorithms.
You can also write your own algorithms under the scikit-learn

*skrebate*

Link: (https://epistasislab.github.io/scikit-rebate/)

skrebate is RELIEF-based feature selection confining the *scikit-learn* interface.
Implemented algorithms: ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, TuRF

*mlextend*

Link: (http://rasbt.github.io/mlxtend/)

Additional machine learning algorithms using scikit-learn framework. For example,

* Feature selectors: ColumnSelector, ExhaustiveFeatureSelector, SequentialFeatureSelector (forward selection, backward selection)
* Classifiers: LogisticRegression, Perceptron, MultiLayerPerceptron, EnsembleVoteClassifier


*SciKit-Learn Laboratory (SKLL)*

Command line wrapper for scikit-learn. You can quickly test machine learning algorithms without writing Python code.

Link: (https://skll.readthedocs.io/en/latest/index.html)

*imbalanced-learn*

Techniques for handling datasets with imbalanced class ratio. Including over-sampling, under-sampling and ensemble methods.

Link: (https://imbalanced-learn.readthedocs.io/en/stable/)

*Other machine learning packages*

Machine learning packages similar to scikit-learn: (https://scikit-learn.org/stable/related_projects.html)

## Example input files

**Expression matrix**

Example filename: `data/example/matrix.txt`.

A tab-separated text file: columns are samples and rows are features.
The sample ids are in the first row and feature names are in the first column.
The expression matrix is usually generated from the original read counts matrix after normalization and batch-effect removal.

Part of the example file:
```
17402567-B      249136-B        385247-B        497411-B        498221-B
hsa-let-7a-3p|miRNA|hsa-let-7a-3p|hsa-let-7a-3p|hsa-let-7a-3p|0|21      241.252239438145        242.671840235123
189.623814753394        374.659743308841
hsa-let-7a-5p|miRNA|hsa-let-7a-5p|hsa-let-7a-5p|hsa-let-7a-5p|0|22      7511.96433719066        8362.29912221603
8330.21347167472        6863.8031596435
hsa-let-7b-3p|miRNA|hsa-let-7b-3p|hsa-let-7b-3p|hsa-let-7b-3p|0|22      50.1959586577797        52.0361345280116
40.9852127416823        100.398094556227
hsa-let-7b-5p|miRNA|hsa-let-7b-5p|hsa-let-7b-5p|hsa-let-7b-5p|0|22      7789.76724807717        9632.71346424331
10061.869728083 16473.7010179541
```

**Class labels**

Example filename : `data/example/sample_classes.txt`

A tab-separated text file with two columns: sample_id, label. The first line is treated as a header.
In this example, class labels are disease status of each sample: stage_A, stage_B, stage_C, Normal.
All samples except samples with "Normal" labels are from HCC patients.

Part of the example file:
```
sample_id       label
17402567-B      stage_A
249136-B        stage_A
385247-B        stage_A
497411-B        stage_A
498221-B        stage_A
507450-B        stage_A
507511-B        stage_A
507867-B        stage_A
507887-B        stage_A
```

## Feature selection and cross-validation with a single combination of parameters

The objective of feature selection is to find a small subset of features that robustly distinguishes between normal and cancer samples.
exSEEK provides a script `machine_learning.py` for various machine-learning tasks including feature selection, cross validation and prediction.

Command-line usage of `machine_learning.py`:

```
usage: machine_learning.py run_pipeline [-h] --matrix FILE --sample-classes
                                        FILE [--positive-class STRING]
                                        [--negative-class STRING]
                                        [--features FILE] --config FILE
                                        --output-dir DIR
                                        [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --matrix FILE, -i FILE
                        input feature matrix (rows are samples and columns are
                        features
  --sample-classes FILE
                        input file containing sample classes with 2 columns:
                        sample_id, sample_class
  --positive-class STRING
                        comma-separated list of sample classes to use as
                        positive class
  --negative-class STRING
                        comma-separates list of sample classes to use as
                        negative class
  --features FILE       input file containing subset of feature names
  --config FILE, -c FILE
                        configuration file of parameters in YAML format
  --output-dir DIR, -o DIR
                        output directory
  --log-level LOG_LEVEL
                        logging level
```

The basic workflow of machine learning is:

**Read input data matrix and class labels**

`machine_learning.py` accepts data matrix and class labels with the `--matrix/-i` and `--sample-classes` option.

`machine_learning.py` assumes that rows are samples and columns are features (different from count matrix or differential expression).
We can transpose the data matrix to swap columns and rows if the data matrix is different from this using the option: `--transpose`.

**Define positive and negative class**

The class labels in `sample_classes.txt` should contain at least 2 categories. If there are more than 2 categories in class labels, we
can specify the positive class and the negative class using the `--positive` and `--negative` option. Each option can be a single label
or a comma-separated list of labels. For example, `--positive-class stage_A,stage_B,stage_C --negative-class Normal`.  
This defines *stage_A*, *stage_B* and *stage_C* as positive class and *Normal* as negative class.

> Note that samples not belonging to any of the specified labels are removed before further steps.

**Specify subset of features (optional)**

To use only a subset of features instead of the whole matrix, we can provide a list of features through option
`--features features.txt`. Each line in `features.txt` is a feature name.

The remaining parameters are specified through a configuration file in YAML format through the `--config/-c` option.
The options described above can also be provided as a YAML configuration file.

YAML is a simple text format that is suitable for exchanging data between programs supported by many programming languages.
YAML is more flexible and readable than JSON (does not support comments).
For more information about YAML format, please refer to [Wikipedia](https://en.wikipedia.org/wiki/YAML)
and [YAML official site](https://yaml.org/).

```yaml
# numbers
n: 1
m: 1.1
# strings
s: abcde
# strings with blank characters
t: "abcde 12345"
# list
- listitem1
- listitem2
- listitem3
# inline list
list1: [a, b, c, d, e, f]
# key-value pairs (dict or hash)
d:
    a: 1
    b: 2
    c: 3
# inline key-value pairs
- {a: 1, b: 2, c: 3}
# nested list
users:
    - username: a
      email: a@example.com
    - username: b
      email: b@example.com
    - username: c
      email: c@example.com
```

**Feature preprocessing**

Before building a machine learning model, the data matrix passes several steps of preprocessing.
There are 4 types of feature preprocessors: scaler, transformer and selector.

Preprocessors as specified in the `preprocess_steps` varible in the configuration file. 
Under `preprocess_steps`, each preprocessor are specified as a list item (`- example_preprocessor`).
The data matrix is preprocessed in the order they are specified in the configuration file.

**feature transformation**

A transformer transforms all elements in the data matrix using the same function.
For expression matrix in RPKM or RPM, we usually transform the matrix by applying the log2 function:

```
log2(x + pseudocount)
```

The configuration section for log transformation is:
```yaml
- log_transform:
    # name of the preprocessor
    name: LogTransform
    # type of the preprocessor
    type: transformer
    # whether to enable the preprocessor
    enabled: true
    params:
        # base of the log function
        base: 2
        # pseudo_count to add to each element to prevent ill-conditioning
        pseudo_count: 1
```

In this example, `log_transform` is the name of the preprocessor.
Two keys are required under the `log_transform` section: `name` and `type`:

* `name` is the name of the predefined transformer. Currently, only one transformer is available: LogTransform.
* `type` is type of the preprocessor. `type` can only take values: `transformer`, `selector`, `scaler`.
* `enabled` is optional. The preprocessor is ignored unless `enabled` is set to `true`.
* `params`: extra parameters passed to the preprocessor. For `LogTransform`, two parameters can be specified: `base` and `pseudo_count`.

**feature scaling**

A scaler is usually applied independently for each feature. 
The most commonly used scaler is z-score scaler, which is also called standard scaler.

The configuration section for standard scaler is:

```yaml
- scale_features:
    name: StandardScaler
    type: scaler
    enabled: true
    params:
        # center all values to their mean for each feature
        with_mean: true
```

Other available scalers: StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

**filter-based feature selector**

A selector select a subset of features, which is the core of feature selection.

There is a global configuration variable `n_features_to_select` that defines a fixed number feature to select.
If the variable is not set and no internal criteria are defined for a selector, all features will be selected.

For a simple filter-based selector, the configuration section is:
```yaml
DiffExp_TTest:
    name: DiffExpFilter
    type: selector
    params:
        # use negative log adjusted p-values for ranking
        score_type: neglogp
        # differential expression method
        method: ttest
```

The `DiffExp_TTest` selector performs differential expression on the data matrix using independent t-test for each feature.
Actually, the script invokes `differential_expression.R` to generate p-values.
The `score_type` params transforms the differential expression results of each feature to a score. In this example, 
the negative log adjusted p-values are used for ranking features. If `n_features_to_select` is defined, 
at most this number of features will be selected.

**selector based on differential expression**
The `DiffExpFilter` executes `differential_expression.R` to select features.
The differential expression script is standalone script that can perform feature selection with various options.
Similar to `machine_learning.py`, `differential_expression.R` accepts an input expression matrix and a sample classes file as input,
with positive class and negative class specified with the `--positive-class` and `--negative-class` option. After 
finishing differential expression, the script outputs a table with at least three columns: feature_name, log2FoldChange and padj (adjusted p-value).

Currently, the script implemented the following DE methods: deseq2, edger_exact, edger_glmqlf, edger_glmlrt,
 wilcox \(Wilcoxon rank-sum test, or Mann–Whitney U test\), limma and ttest \(unpaired t-test\).

For DESeq2, edgeR and wilcox, the input expression matrix is assumed to be a read counts matrix.
For edgeR and limma, a normalization can be specified: RLE, CPM, TMM or upperquartile. DESeq2 uses its own normalization methods.
For limma and ttest, the input expression matrix is assumed to normalized and log2-transformed.
For wilcox, the input count matrix is normalized and then log2-transformed. To prevent zeros in log function, a pseudo-count
is added to every counts, which can be specified by the `--pseudo-count` option.

For DESeq2, edgeR and limma, a file containing batch information can be provided by the `--batch/-b` option.
In DESeq2 and edgeR, batch effects are removed by using batch variables as covariates in negative binomial regression.
In limma, batch effects are first removed by linear regression using the `removeBatchEffect` function.

Command-line usage of `differential_expression.R`:
```
usage: bin/differential_expression.R [-h] -i MATRIX -c CLASSES [-s SAMPLES] -p
                                     POSITIVE_CLASS -n NEGATIVE_CLASS
                                     [-b BATCH] [--batch-index BATCH_INDEX]
                                     [-m {deseq2,edger_exact,edger_glmqlf,edger_glmlrt,wilcox,limma,ttest}]
                                     [--norm-method {RLE,CPM,TMM,upperquartile}]
                                     [--pseudo-count PSEUDO_COUNT] -o
                                     OUTPUT_FILE

Differential expression

optional arguments:
  -h, --help            show this help message and exit
  -i MATRIX, --matrix MATRIX
                        input count matrix. Rows are genes. Columns are
                        samples.
  -c CLASSES, --classes CLASSES
                        input sample class information. Column 1: sample_id,
                        Column 2: class
  -s SAMPLES, --samples SAMPLES
                        input file containing sample ids for differential
                        expression
  -p POSITIVE_CLASS, --positive-class POSITIVE_CLASS
                        comma-separated class names to use as positive class
  -n NEGATIVE_CLASS, --negative-class NEGATIVE_CLASS
                        comma-separated class names to use as negative class
  -b BATCH, --batch BATCH
                        batch information to remove
  --batch-index BATCH_INDEX
                        column number of the batch to remove
  -m {deseq2,edger_exact,edger_glmqlf,edger_glmlrt,wilcox,limma,ttest}, --method {deseq2,edger_exact,edger_glmqlf,edger_glmlrt,wilcox,limma,ttest}
                        differential expression method to use
  --norm-method {RLE,CPM,TMM,upperquartile}
                        normalization method for count-based methods
  --pseudo-count PSEUDO_COUNT
                        pseudo-count added to log2 transform in ttest
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        output file
```

Example output of `differential_expression.R` (using DESeq2):
```
baseMean        log2FoldChange  lfcSE   stat    pvalue  padj
hsa-let-7a-3p|miRNA|hsa-let-7a-3p|hsa-let-7a-3p|hsa-let-7a-3p|0|21      612.040526671729        3.10679875843708
1.03430534034172        3.00375395665136        0.00266670886908437     0.00553987261835591
hsa-let-7a-5p|miRNA|hsa-let-7a-5p|hsa-let-7a-5p|hsa-let-7a-5p|0|22      31411.1827219365        3.66665458899661
0.726406098332838       5.04766493207021        4.4724262801185e-07     1.77064089614528e-06
hsa-let-7b-3p|miRNA|hsa-let-7b-3p|hsa-let-7b-3p|hsa-let-7b-3p|0|22      115.821202185891        3.93146114221439
1.08488185382557        3.62386109450633        0.000290237520297458    0.000717057403087837
hsa-let-7b-5p|miRNA|hsa-let-7b-5p|hsa-let-7b-5p|hsa-let-7b-5p|0|22      29872.1083869283        2.58958074077096
1.61681717531621        1.60165340912122        0.109232273268584       0.154723343393663
hsa-let-7c-3p|miRNA|hsa-let-7c-3p|hsa-let-7c-3p|hsa-let-7c-3p|0|22      2.60866963317644        1.38121239691098
0.478335824437209       2.88753701133729        0.00388270923167969     0.00779770710561867
```

The `score_type` parameter in `DiffExpFilter` specifies how to rank features. Three values are possible:
`neglogp`, `pi_value`, `log2fc`. `pi_value` is a combined score of `log2fc` and `neglogp`.

**wrapper-based feature selector**
If the feature selector is wrapper-based, it requires a classifier to perform feature selection.
The configuration section of an example wrapper-based feature selection is:
```yaml
MaxFeatures_ElasticNet:
    name: MaxFeatures
    type: selector
    params:
      n_features_to_select: 10
      classifier: SGDClassifier
      classifier_params:
        penalty: elasticnet
        max_iter: 100
        tol: 0.001
      grid_search: true
      grid_search_params:
        param_grid:
          alpha: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
          l1_ratio: [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
        cv:
            # name of the train-test splitter
            splitter: StratifiedShuffleSplit
            # number of train-test splits and runs for each parameter combination
            n_splits: 5
            # fraction/number of samples to set as test samples during each cross-validation run
            test_size: 0.1
        iid: false
        scoring: roc_auc
```

This selector is called *MaxFeatures*, which select features based on feature importance or feature coefficients in the internal
classifier. The *MaxFeatures* selector accepts a parameter `n_features_to_select` that limit maximum number of features to select.
Other wrapper-based feature selector include: *RobustSelector*, *RFE*, *RFECV*.

A wrapper selector requires `classifier` to be specified under the `params` key.
`classifier` is the name of a predefined classifier. 
`machine_learning.py` wraps many classifiers in the *scikit-learn* Python package with the same name.
You can refer to documentation of *scikit-learn* for parameters of each classifier.

Optionally, additional parameters can be specified in `classifier_params` along with `classifier`.
In this example, because the selector uses SGDClassifier to implement ElasticNet feature selection,
the `penalty` parameter in `classifier_params` is set to `elasticnet`.

The `splitter` parameter is the name of the splitter that splits samples into training samples and test samples.
`machine_learning.py` wraps most splitters in the *scikit-learn* package. You can refer to 
[documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) of *scikit-learn* for more information.

**Train the classifier**

After preprocessing, a classifier is trained on the preprocessed matrix.
The configuration section is similar to a wrapper-based feature selector:

```yaml
classifier: LogisticRegression
# parameters for classifier
classifier_params:
    penalty: l2
    solver: liblinear

# grid search for hyper-parameters of the classifier
grid_search: false
grid_search_params:
    # parameter grid that defines possible values for each parameter
    param_grid:
        C: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    # cross-validation parameters for grid search
    cv:
        splitter: StratifiedShuffleSplit
        n_splits: 5
        test_size: 0.1
    iid: false
    scoring: roc_auc
sample_weight: balanced
```

Note that there are 5 varibles for a classifier: `classifier`, `classifier_params`, `grid_search`, `grid_search_params`, `sample_weight`.
Only `classifier` is required.

sample_weight: when set to 'balanced', it computes sample weight using the `sklearn.utils.class_weight.compute_sample_weight` function.
Balanced sample weights are inversely proportional to `n_samples / (n_classes * np.bincount(y))`. This is a simple way to handle 
imbalanced classes.

**Grid search for hyper-parameter optimization**

If you need to optimize hyper-parameters of the internal classifier, you can set `grid_search` to `true`
and add additional parameters in `grid_search_params`.
`param_grid` defines a the parameter space to search. 
Each item in `param_grid` specifies a list of possible values for each hyper-parameter.
All combinations of hyper-parameters are evaluated and the combination with best metric is selected.
The hyper-parameters are optimized on the training samples before being passed to the selector.

For the following example, two hyper-parameters `alpha` and `l1_ratio` needs to be optimized:
```yaml
grid_search: true
grid_search_params:
    param_grid:
        alpha: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        l1_ratio: [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    cv:
        # name of the train-test splitter
        splitter: StratifiedShuffleSplit
        # number of train-test splits and runs for each parameter combination
        n_splits: 5
        # fraction/number of samples to set as test samples during each cross-validation run
        test_size: 0.1
    iid: false
    scoring: roc_auc
```

Grid search sets hyper-parameters `alpha` and `l1_ratio` to `[(0.00001, 0.15), (0.001, 0.15), (0.001, 0.15), ..., (0.00001, 0.30)...]`
in each iteration. 

The `scoring` parameter defines the metric for optimizing hyper-paremeter for the classifier.
By default, grid search evaluated each combination of hyper-parameters by cross-validation, which can be specified
under the `cv` key. The performance metrics on test datasets are averaged across cross-validation runs for each
hyper-parameter combination. The hyper-parameter combination with the highest performance metric is selected.

**Performance evaluation by cross-validation**

Cross-validation is the outer loop of the pipeline. During each cross-validation run
all samples are splitted into training samples and test samples using a splitter.
Feature preprocessing, classifier training and hyper-parameter optimization is only done
on training samples. The test samples are used to evaluated the combined classifier.


Configuration parameters for cross-validation is in `cv_params`:
```yaml
# cross-validation parameters for performance evaluation
cv_params:
    splitter: StratifiedShuffleSplit
    # number of train-test splits for cross-validation
    n_splits: 10
    # number or proportion of samples to use as test set
    test_size: 0.1
    # scoring metric for performance evaluation
    scoring: roc_auc
```

`cv_params` is similar to `cv` variable in `grid_search_params`.

**An example feature selection run**

A complete example of feature selection command:

```bash
bin/machine_learning.py run_pipeline \
    -i example/data/matrix.txt \
    --sample-classes example/data/sample_classes.txt \
    --positive-class "stage_A,stage_B,stage_C" \
    --negative-class "Normal" \
    -c example/data/config.yaml \
    -o example/output
```

The output directory is `example/output`.

**Output files of feature selection**

Many files are generated in the output directory after the above command:

| File name pattern | Descrpition |
| :--- | :--- |
| `features.txt` | Selected features. Plain text with one column: feature names |
| `feature_importances.txt` | Plain text with two columns: feature name, feature importance |
| `samples.txt` | Sample IDs in input matrix selected for feature selection |
| `classes.txt` | Sample class labels selected for feature selection |
| `final_model.pkl` | Final model fitted on all samples in Python pickle format |
| `metrics.train.txt` | Evaluation metrics on training data. First row is metric names. First column is index of each train-test split |
| `metrics.test.txt` | Same format with `metrics.train.txt` on test data. |
| `cross_validation.h5` | Cross-validation details in HDF5 format. |

**Cross validation details \(cross\_validation.h5\)**

| Dataset name | Dimension | Description |
| :--- | :--- | :--- |
| feature\_selection | \(n\_splits, n\_features\) | Binary matrix indicating features selected in each cross-validation split |
| labels | \(n\_samples,\) | True class labels |
| predicted\_labels | \(n\_splits, n\_samples\) | Predicted class labels on all samples |
| predictions | \(n\_splits, n\_samples\) | Predicted probabilities of the positive class \(or decision function for SVM\) |
| train\_index | \(n\_splits, n\_samples\) | Binary matrix indicating training samples in each cross-validation split |

## Run combinations of selectors and classifiers feature selection in batch

Assume that we have already run `normalization` module.
The filenames of the expression matrix files are in the following format:

`output/$dataset/matrix_processing/${filter}.${normalization}.${batch_removal}.${count_method}.txt`


Then we can run feature selection module using the following command:

```bash
exseek.py feature_selection -d ${dataset}
```

The *feature_selection* command tries all combinations of these variables:

* n_features_to_select: maximum number of features to select
* classifier: classifier to use
* selector: feature selection algorithm to use
* fold_change_direction: direction for the fold change filter. Possible values: 'any', 'up', 'down'.

You can set possible values for each variable in `config/machine_learning.yaml`.

**n_features_to_select**

Global variable that affects overrides all selectors. If you want to try multiple values, you can set:

```yaml
n_features_to_select: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**selectors**

Each item in the `selectors` variable is identical to a selector configuration for `machine_learning.py`.
```yaml
selectors:
  DiffExp_TTest:
    name: DiffExpFilter
    type: selector
    params:
      score_type: neglogp
      method: ttest
  MaxFeatures_RandomForest:
    name: MaxFeatures
    type: selector
    params:
      classifier: RandomForestClassifier
      grid_search: true
      grid_search_params:
        param_grid:
          n_estimators: [25, 50, 75]
          max_depth: [3, 4, 5]
```

**classifiers**

Each item in `classifiers` variable is identical to a classifier configuration for `machine_learning.py`.
```yaml
classifiers:
  LogRegL2:
    classifier: LogisticRegression
    # parameters for the classifier used for feature selection
    classifier_params:
      penalty: l2
      solver: liblinear
    # grid search for hyper-parameters for the classifier
    grid_search: true
    grid_search_params:
      param_grid:
        C: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
  RandomForest:
    classifier: RandomForestClassifier
    grid_search: true
    grid_search_params:
      param_grid:
        n_estimators: [25, 50, 75]
        max_depth: [3, 4, 5]
```

The complete configuration file can be found in (https://github.com/lulab/exSEEK/blob/master/config/machine_learning.yaml).
## Output directories of exSEEK feature selection

Feature selection results using one combination of parameters are saved in a separate directory:

```
${output_dir}/cross_validation/${preprocess_method}.${count_method}/${compare_group}/${classifier}.${n_select}.${selector}.${fold_change_filter_direction}
```

**Variables in file patterns**

| Variable | Descrpition |
| :--- | :--- |
| `output_dir` | Output directory for the dataset, e.g. `output/dataset` |
| `preprocess_method` | Combination of matrix processing methods |
| `count_method` | Type of feature counts, e.g. `domains_combined`, `domains_long`, `transcript`, `featurecounts` |
| `compare_group` | Name of the negative-positive class pair defined in `compare_groups.yaml` |
| `classifier` | Classifier defined in the configuration file |
| `n_select` | Maximum number of features to select |
| `selector` | Feature selection method, e.g. `robust`, `rfe` |
| `fold_change_filter_direction` | Direction of fold change for filtering features. Three possible values: `up`, `down` and `any` |


## Summary table of exSEEK feature selection

After finishing `exseek.py feature_selection`, a summary directory is created: `output/$dataset/summary/cross_validation`

Two files are generated: `metrics.train.txt` and `metrics.test.txt`. `metrics.test.txt` contains performance metrics computed on test data.

Example output file contents of `metrics.test.txt`:

```
classifier      n_features      selector        fold_change_direction   compare_group   filter_method   imputation
normalization   batch_removal   count_method    preprocess_method       split   accuracy        average_precision
f1_score        precision       recall  roc_auc
LogRegL2        10      DiffExp_TTest   any     Normal-HCC      filter  null    Norm_RLE        Batch_limma_1   mirna_and_domains_rna   filter.null.Norm_RLE.Batch_limma_1      0       0.7142857142857143      0.1504884004884005      0.0
0.0     0.0     0.0
LogRegL2        10      DiffExp_TTest   any     Normal-HCC      filter  null    Norm_RLE        Batch_limma_1   mirna_and_domains_rna   filter.null.Norm_RLE.Batch_limma_1      1       0.9285714285714286      0.75    0.8     1.0     0.6666666666666666      0.7272727272727272
```

Summarize metrics as a matrix using Python (Jupyter is recommended):

```python
import pandas as pd
summary = pd.read_table('example/output/summary/cross_validation/metrics.test.txt', sep='\t')
df = summary.query('(compare_group == "Normal-HCC") and (normalization == "Norm_RLE") and (batch_removal == "Batch_limma_1") and (n_features == 10) and (count_method == "mirna_and_domains_rna")')\
    .groupby(['classifier', 'selector'])['roc_auc'].mean().unstack()
df
```
The above code first select a subset of matrix with the following filters:

* `compare_group == "Normal-HCC"`: only compares Normal with HCC
* `normalization == "Norm_RLE"`: use RLE as normalization method
* `batch_removal == "Batch_limma_1"`: use limma as batch effect removal method (remove batch 1)
* `n_features == 10`: only select 10 features at maximum
* `count_method == "mirna_and_domains_rna"`: use miRNA + ncRNA domains as features

Then AUROC \(`roc_auc`\) across cross-validation runs are averaged and create a matrix with classifiers as rows and selectors as columns:

| selector | DiffExp_TTest | MaxFeatures_ElasticNet | MaxFeatures_LogRegL1 | MaxFeatures_LogRegL2 | MaxFeatures_RandomForest | MultiSURF | ReliefF | SIS | SURF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DecisionTree | 0.860606 | 0.904848 | 0.856061 | 0.886364 | 0.862727 | 0.931818 | 0.904848 | 0.880303 | 0.927879 |
| LogRegL2 | 0.800000 | 0.660606 | 0.794545 | 0.753333 | 0.741212 | 0.627273 | 0.744242 | 0.727879 | 0.693939 |
| MLP | 0.847273 | 0.756970 | 0.790303 | 0.790909 | 0.804848 | 0.756970 | 0.701212 | 0.752121 | 0.766061 |
| RBFSVM | 0.843030 | 0.903030 | 0.947273 | 0.928485 | 0.952121 | 0.961818 | 0.965455 | 0.916364 | 0.956970 |
| RandomForest | 0.988485 | 0.978788 | 0.981212 | 0.980606 | 0.979394 | 0.984545 | 0.981212 | 0.985455 | 0.986061 |

Similarly, you can summarize AUROC using only miRNA features:
```python
df = summary.query('(compare_group == "Normal-HCC") and (normalization == "Norm_RLE") and (batch_removal == "Batch_limma_1") and (n_features == 10) and (count_method == "mirna_only")')\
    .groupby(['classifier', 'selector'])['roc_auc'].mean().unstack()
```

The output table:

| selector | DiffExp_TTest | MaxFeatures_ElasticNet | MaxFeatures_LogRegL1 | MaxFeatures_LogRegL2 | MaxFeatures_RandomForest | MultiSURF | ReliefF | SIS | SURF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DecisionTree | 0.735152 | 0.929394 | 0.864848 | 0.933939 | 0.897879 | 0.904545 | 0.900303 | 0.892121 | 0.928485 |
| LogRegL2 | 0.573636 | 0.743030 | 0.828485 | 0.818182 | 0.716364 | 0.807879 | 0.775152 | 0.828485 | 0.807879 |
| MLP | 0.765455 | 0.813333 | 0.823030 | 0.825455 | 0.903030 | 0.881818 | 0.946667 | 0.828485 | 0.859394 |
| RBFSVM | 0.816970 | 0.938788 | 0.964242 | 0.979394 | 0.981818 | 0.976364 | 0.983030 | 0.979394 | 0.980606 |
| RandomForest | 0.954545 | 0.983636 | 0.990303 | 0.983030 | 0.985758 | 0.987273 | 0.992121 | 0.988788 | 0.989091 |



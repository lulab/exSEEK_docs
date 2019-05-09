# Feature Selection

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

**feature selection**

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

If the feature selector is wrapper-based, it requires a classifier to perform feature selection.
The configuration section of an example wrapper-based feature selection is:
```yaml
MaxFeatures_ElasticNet:
    name: MaxFeatures
    type: selector
    params:
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

A wrapper selector requires `classifier` to be specified under the `params` key.
`classifier` is the name of a predefined classifier. 
`machine_learning.py` wraps many classifiers in the *scikit-learn* Python package with the same name.
You can refer to documentation of *scikit-learn* for parameters of each classifier.

Optionally, additional parameters can be specified in `classifier_params` along with `classifier`.
In this example, because the selector uses SGDClassifier to implement ElasticNet feature selection,
the `penalty` parameter in `classifier_params` is set to `elasticnet`.

If you need to optimize hyper-parameters of the internal classifier, you can set `grid_search` to `true`
and add additional parameters in `grid_search_params`.
`param_grid` defines a the parameter space to search. 
Each item in `param_grid` specifies a list of possible values for each hyper-parameter.
All combinations of hyper-parameters are evaluated and the combination with best metric is selected.
The hyper-parameters are optimized on the training samples before being passed to the selector.

The `scoring` parameter defines the metric for optimizing hyper-paremeter for the classifier.
By default, grid search evaluated each combination of hyper-parameters by cross-validation, which can be specified
under the `cv` key. 

The `splitter` parameter is the name of the splitter that splits samples into training samples and test samples.
`machine_learning.py` wraps most splitters in the *scikit-learn* package. You can refer to 
[documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) of *scikit-learn* for more information.






## Run feature selection module

Assume that we have already run `normalization` module and selected best matrix processing method based on the UCA score, we can run feature selection module using the following command:

```bash
exseek.py feature_selection -d ${dataset}
```

## Output files

### Outuput directory

Feature selection results using one combination of parameters are saved in a separate directory:

```text
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

### Files in output directory

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


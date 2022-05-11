# Model Agnostic Local Explanations of Reject

This repository contains the implementation of the methods proposed in the paper [Model Agnostic Local Explanations of Reject](paper.pdf) by André Artelt, Roel Visser and Barbara Hammer.

The experiments as described in the paper are implemented in the folder [Implementation](Implementation/).

## Abstract

The application of machine learning based decision making systems in safety critical areas requires reliable high certainty predictions.

Reject options are a common way of ensuring a sufficiently high certainty of predictions made by the system. While being able to reject uncertain samples is important, it is also of importance to be able to explain why a particular sample was rejected. However, explaining general reject options is still an open problem.

We propose a model agnostic method for locally explaining arbitrary reject options by means of interpretable models and counterfactual explanations.

## Details
### Implementation of experiments
The shell script `run_experiments.sh` runs all experiments.

### Other (important) stuff
#### `explanation.py`
Implementation of our proposed *model agnostic local explanation of reject*.

#### `conformalprediction.py`
Implementation of conformal prediction.

#### `reject_option.py`
Implementation of the reject options discussed in the paper.

## Data

Note that we did not publish all data sets due to unclear copyrights. Please contact us if you are interested in the medical data sets.

## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE](LICENSE)

## How to cite

You can cite the version on [arXiv](TODO)

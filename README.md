# Code Repository for "Mitigating Exposure Bias in Recommender Systems – A Comparative Analysis of Discrete Choice Models"

This is a code repository for the unpublished research paper "Mitigating Exposure Bias in Recommender Systems – A Comparative Analysis of Discrete Choice Models" by Thorsten Krause, Alina Deriyeva, Jan H. Beinke, Gerrit Y. Bartels, and Oliver Thomas. If you use any part of this code, please cite it using the following BibTex:

<BibTex>

## Repository Under Construction
Please note that this repository is currently under construction and may not be fully functional. We apologize for any inconvenience this may cause. We are working to complete the repository as soon as possible.

### Baseline Code
We plan to include the code for the baselines used in this project as soon as the respective authors include the necessary licenses. We apologize for any delay this may cause.

Thank you for your patience and understanding.

## Data set

## Installing the Necessary Components

To install the necessary components for this project, please run the following command:

```
pip install -r requirements.txt
```

## Running the Code

To run the entire experiment, please follow the instructions below:

<Instructions>

To run individual models, please follow the instructions below:

<Instructions>

## Hyperparameters

All hyperparameters used in this project are contained in the `src/config.py` file.

## Expected Results

The expected results of this project include the following:
- Exposure bias benchmarks for all models in the first experiment
- Respective nDCG scores for the first experiment
- Exposure bias benchmarks for all models in the second experiment
- Respective nDCG scores for the second experiment

These results are visualized in the following .svg images, which can be found in the `data/output/plots` directory:
- `bias_overexposure_side_by_side.svg`: Exposure bias benchmarks for all models in the first experiment
- `performance_overexposure_side_by_side.svg`: Respective nDCG scores for the first experiment
- `bias_competition.svg`: Exposure bias benchmarks for all models in the second experiment
- `performance_competition_side_by_side.svg`: Respective nDCG scores for the second experiment

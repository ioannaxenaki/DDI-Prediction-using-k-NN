# DDI-Prediction-using-k-NN
Drug-Drug Interactions (DDIs) occur when two or more drugs are taken together and one affects the activity, efficacy, or safety of another. This project implements a k-nearest neighbor (k-NN) approach to predict the specific type of pharmacological effect (out of 86 different relation types) between drug pairs based on their molecular structures.

# Project Overview
The core idea of this implementation is that chemically similar drug pairs are likely to exhibit similar interaction types. For each test drug pair, the script finds the most similar training drug pair and assigns its interaction type (k=1).   
The algorithm computes similarities using two precomputed molecular features:
+ Morgan Features: Binary fingerprint vectors representing molecular substructures, compared using Tanimoto (Jaccard) distance.   
+ RDKit2D Features: Continuous numerical vectors capturing properties like molecular weight and lipophilicity, compared using Cosine distance.

These feature distances are calculated for both drugs in a pair, averaged, and then combined using a weighted sum (defaulting to 0.5 for each feature type).

# Setup and Data
This project relies on data from the [DDI-Bench](https://github.com/LARS-research/DDI-Bench) repository. To run the code, download the following files from DDI-Bench:   
+ ```train.txt```: The training reference dataset mapping ```drug_id_a```, ```drug_id_b```, and ```relation_id```.
+ ```test_S0.txt```: The test set used for evaluation.
+ ```relation2id.json```: A dictionary mapping the integer relation IDs (0-85) to their text descriptions.
+ ```DB_molecular_feats.pkl```: A pickle file containing the SMILES strings, Node IDs, and precomputed Morgan and RDKit2D features for 1,710 drugs.

# Usage
The main script, ```Project_1.py```, runs via the command line and supports two modes:   

## 1. Train Mode (Evaluation)

Evaluates the model's accuracy on the test data. Because comparing the full test set against the full training set requires over 1.2 billion distance calculations, use the --cutoff parameter to limit the number of test samples during initial testing.

```
python Project_1.py \
    --molecular_feats path/to/DB_molecular_feats.pkl \
    --train path/to/train.txt \
    --test path/to/test_S0.txt \
    --cutoff 20 \
    --mode train
```

## 2. Inference Mode (Prediction)
Predicts the specific text description of the interaction between any two specific DrugBank IDs.   
```
python Project_1.py \
    --molecular_feats path/to/DB_molecular_feats.pkl \
    --relation2id path/to/relation2id.json \
    --train path/to/train.txt \
    --mode inference \
    --drugbank_1 DB13231 \
    --drugbank_2 DB00244
```

# Acknowledgements
The dataset and molecular features accompany the following publication:
Zhenqian Shen, Mingyang Zhou, Yongqi Zhang, Quanming Yao, Benchmarking drug–drug interaction prediction methods: a perspective of distribution changes, Bioinformatics, Volume 41, Issue 11, November 2025, btaf569, https://doi.org/10.1093/bioinformatics/btaf569.

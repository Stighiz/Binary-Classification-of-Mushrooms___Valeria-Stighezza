# Binary-Classification-of-Mushrooms

This repository contains a machine learning project for classifying mushrooms as edible or poisonous using the Secondary Mushroom dataset. It implements custom decision trees and random forests, executed through two main scripts: `main__run1.py` and `main__run2.py`, each applying the full pipeline to distinct train-test splits to ensure robustness and assess model generalization.

The project includes data preprocessing (splitting data, removing features, imputing missing values with medians for numerical and modes for categorical features, and eliminating duplicates); model construction of decision trees and random forests using different impurity measures (Gini, entropy, or squared error) for optimal node splitting; and hyperparameter tuning via grid search and 5-fold cross-validation with parallel processing.
Performance of the implemented models is evaluated with metrics including accuracy, precision, recall, F1-score and 0-1 loss, visualized via confusion matrices and performance plots. 

The tuned random forest achieves perfect test accuracy (100%) in both runs, demonstrating the power of ensemble methods and precise hyperparameter tuning. 

The results, including tree diagrams and evaluation charts, are organized into two folders: `imgs__run1` and `imgs__run2` (one folder for each run).

A detailed explanation of the entire process, from preprocessing to evaluation, is provided in the report file: `Binary_Classification_of_Mushrooms___Valeria_Stighezza.pdf`.

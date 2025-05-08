# Binary-Classification-of-Mushrooms

This repository contains a machine learning project for classifying mushrooms as edible or poisonous using the Secondary Mushroom dataset, implementing decision trees and random forests from scratch. 

Due to the long execution time, two separate runs of the same pipeline are provided in `src\main__run1.py` and `src\main__run2.py`, each applying the full pipeline to distinct train-test splits. This allows for better evaluation of model robustness and generalization.

The project includes data preprocessing (splitting data, removing features, imputing missing values with medians for numerical and modes for categorical features, and eliminating duplicates); model construction of decision trees and random forests using different impurity measures (Gini, entropy, or squared error) for optimal node splitting; and hyperparameter tuning via grid search and 5-fold cross-validation with parallel processing.
Performance of the implemented models is evaluated with metrics including accuracy, precision, recall, F1-score and 0-1 loss, visualized via confusion matrices and performance plots. 

The tuned random forest achieves perfect test accuracy (100%) in both runs, demonstrating the power of ensemble methods and precise hyperparameter tuning. 

The results, including tree diagrams and evaluation charts, are organized into two folders: `src\imgs__run1` and `src\imgs__run2` (one folder for each run).

A detailed explanation of the entire process, from preprocessing to evaluation, is provided in the report file: `report\Binary_Classification_of_Mushrooms___Valeria_Stighezza.pdf`.

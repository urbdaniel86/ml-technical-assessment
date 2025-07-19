import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import shap
from shap import Explanation
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnModel:
    def __init__(self, config):          # load hyperparams, paths
        ...
    def load_data(self):                # read data
        ...
    def preprocess(self, df):           # feature engineering
        ...
    def train(self):                    # fits XGBoost
        ...
    def evaluate(self, df_test):        # compute metrics
        ...
    def save(self, model_path):         # save model artifact
        ...

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ...
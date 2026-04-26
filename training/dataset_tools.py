"""Run this file to analyze the current training data CSVs.

Edit FILES below to point at whatever you want to inspect, then:

    python training/dataset_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, Dataset

FILES = [
    os.path.join(REPO_ROOT, "training", "data", "hce_5kn.csv"),
    os.path.join(REPO_ROOT, "training", "data", "hce_10kn.csv"),
]

if __name__ == '__main__':
    Dataset.from_files(FILES, outcome_scale=1.0).analyze("Raw")

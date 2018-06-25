#!/bin/bash
python scripts/fix_csv_files.py
python scripts/preprocess.py
python scripts/prepare_features.py
python scripts/bayesian.py

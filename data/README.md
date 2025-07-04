# Data Directory

This directory contains the datasets used for credit risk modeling and analysis.

## Required Datasets

### Default Credit Card Dataset
Download from UCI Machine Learning Repository:
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
```

### German Credit Dataset
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc
```

### Lending Club Dataset
Download from Kaggle:
```bash
# Requires kaggle API setup
kaggle datasets download -d wordsforthewise/lending-club
```

## Data Format

After downloading, your data directory should look like:
```
data/
├── raw/
│   ├── credit_card_default.xls
│   ├── german.data
│   ├── lending_club.csv
│   └── ...
├── processed/
│   ├── train.csv
│   ├── test.csv
│   └── validation.csv
└── feature_definitions.csv
```

## Data Preprocessing

The raw data will be automatically preprocessed:

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
# This will clean and prepare the data
X_train, X_test, y_train, y_test = preprocessor.load_and_split_data('data/raw/credit_card_default.xls')
```

## Expected Data Schema

### Required Columns
Your dataset should include the following types of features:

**Demographic Features:**
- `age`: Age of the applicant
- `gender`: Gender (M/F or 1/0)
- `education`: Education level
- `marriage`: Marital status

**Financial Features:**
- `limit_balance`: Credit limit
- `pay_*`: Payment history (last 6 months)
- `bill_amt_*`: Bill amounts (last 6 months)
- `pay_amt_*`: Payment amounts (last 6 months)

**Target Variable:**
- `default`: 1 if defaulted, 0 otherwise

### Example CSV Format
```csv
age,gender,education,marriage,limit_balance,pay_1,pay_2,bill_amt_1,pay_amt_1,default
25,2,2,1,20000,2,2,3913,0,1
26,2,2,2,120000,0,0,2682,1725,0
29,2,2,2,90000,0,0,29239,1518,0
```

## Data Quality Requirements

- **Missing Values**: < 5% per column
- **Outliers**: Will be handled automatically
- **Data Types**: Mixed (numerical and categorical)
- **Minimum Sample Size**: 1,000 records recommended

## Custom Dataset

To use your own dataset:

1. Format your data according to the schema above
2. Place it in `data/raw/your_dataset.csv`
3. Update the configuration in `config.py`:

```python
# Dataset configuration
DATASET_PATH = 'data/raw/your_dataset.csv'
TARGET_COLUMN = 'default'
FEATURE_COLUMNS = ['age', 'gender', 'education', ...]
```

## Data Statistics

### Default Credit Card Dataset
- Records: 30,000
- Features: 24
- Default Rate: 22.1%
- Missing Values: 0%

### German Credit Dataset
- Records: 1,000
- Features: 21
- Default Rate: 30%
- Missing Values: 0%

## Privacy Notice

This project handles sensitive financial data. Please ensure:
- Data is properly anonymized
- Compliance with local privacy regulations (GDPR, CCPA, etc.)
- Secure storage and transmission
- Regular data audits

# ANNCustomerChurn

This project implements an Artificial Neural Network (ANN) to predict customer churn. The model is trained to classify whether a customer will leave the service based on demographic and account-related features. A Streamlit interface is included for real-time predictions.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository provides a full pipeline for customer churn prediction using deep learning. It covers data preprocessing, model training, evaluation, and deployment through a simple user interface.

## Motivation

Customer churn is a critical metric for subscription-based businesses. Accurately predicting customer churn enables targeted retention strategies, reducing revenue loss and increasing customer lifetime value.

## Dataset

The dataset includes the following types of features:

- Demographic: Age, Gender, Geography
- Account Features: Credit Score, Balance, Tenure, Number of Products, Has Credit Card, Is Active Member
- Target: Exited (binary churn indicator)

*Note: Replace with actual dataset description or link if using a public dataset.*

## Technologies

- Python 3.8+
- Pandas and NumPy for data manipulation
- Scikit-learn for preprocessing and evaluation metrics
- TensorFlow and Keras for model building
- Streamlit for the web-based application
- Pickle for saving encoders and scalers

## Project Structure

```
ANNCustomerChurn/
├── data/
│   └── raw_data.csv
├── models/
│   ├── churn_model.h5
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```
git clone https://github.com/shishiradk/ANNCustomerChurn.git
cd ANNCustomerChurn
```

Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate      # Windows
```

Install required packages:

```
pip install -r requirements.txt
```

## Usage

**1. Preprocess the data**

```
python src/preprocess.py --input data/raw_data.csv --output data/processed_data.csv
```

**2. Train the model**

```
python src/train_model.py --data data/processed_data.csv --model-output models/churn_model.h5
```

**3. Evaluate the model**

```
python src/evaluate.py --model models/churn_model.h5 --data data/processed_data.csv
```

**4. Run the Streamlit app**

```
streamlit run src/app.py
```

## Model Training and Evaluation

- Model architecture: Dense layers with dropout regularization
- Activation functions: ReLU for hidden layers, Sigmoid for output
- Loss function: Binary Cross-Entropy
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

## Results

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.85  |
| Precision | 0.82  |
| Recall    | 0.78  |
| F1-score  | 0.80  |

*These values are sample outputs. Replace with actual results after evaluation.*

## Future Work

- Include hyperparameter tuning
- Add support for cross-validation
- Integrate with deployment pipelines (Docker, CI/CD)
- Expand feature engineering
- Compare with traditional models (Logistic Regression, Random Forest)

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request with a detailed description

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Author: Shishir Adk  
Email: shishir@example.com  
GitHub: https://github.com/shishiradk

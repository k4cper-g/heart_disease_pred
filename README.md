# Heart Disease Prediction using Naive Bayes

This project aims to predict the presence and stage of heart disease in patients using a Naive Bayes classifier. Two models are developed: one for binary classification (healthy/sick) and another for multi-class classification of disease stages.

**Author:** Kacper Gadomski
**Version:** 0.4 (as per report)
**Last Modification (Report):** January 23, 2024

## Table of Contents
1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    * [Algorithm](#algorithm)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Evaluation](#model-evaluation)
4.  [Models & Results](#models--results)
    * [Model 1: Binary Classification (Healthy/Sick)](#model-1-binary-classification-healthysick)
    * [Model 2: Disease Stage Classification](#model-2-disease-stage-classification)
5.  [Repository Structure](#repository-structure)
6.  [Installation](#installation)
7.  [Usage](#usage)
8.  [Detailed Report](#detailed-report)

## Overview

Cardiovascular diseases are a leading cause of death globally. Early prediction can significantly improve patient outcomes. This project explores the use of a Naive Bayes probabilistic algorithm to develop models for detecting heart disease based on clinical data. The project provides two distinct approaches:
* A high-accuracy binary model to determine if a patient is healthy or exhibits signs of heart disease.
* A multi-class model to classify the specific stage of heart disease.

## Dataset

The data used is from the Cleveland Clinic Foundation, processed and made available in the `processedcleveland.csv` file. 
* **Instances:** Originally over 300 patients. After handling missing values, the exact number used by the scripts might vary slightly.
* **Attributes:** The models use 13 clinical features and 1 target attribute.

The 14 attributes used in the analysis are: 
1.  `age`: Age of the patient (in years)
2.  `sex`: Sex (1 = male; 0 = female)
3.  `cp`: Chest pain type
    * Value 1: typical angina
    * Value 2: atypical angina
    * Value 3: non-anginal pain
    * Value 4: asymptomatic
4.  `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
5.  `chol`: Serum cholesterol in mg/dl
6.  `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7.  `restecg`: Resting electrocardiographic results
    * Value 0: normal
    * Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8.  `thalach`: Maximum heart rate achieved
9.  `exang`: Exercise induced angina (1 = yes; 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: The slope of the peak exercise ST segment
    * Value 1: upsloping
    * Value 2: flat
    * Value 3: downsloping
12. `ca`: Number of major vessels (0-3) colored by fluoroscopy
13. `thal`: Thalassemia
    * Value 3: normal
    * Value 6: fixed defect
    * Value 7: reversible defect
14. `num` (Target Variable): Diagnosis of heart disease (angiographic disease status)
    * Value 0: < 50% diameter narrowing (healthy)
    * Value 1-4: > 50% diameter narrowing (various stages of sickness) 

## Methodology

### Algorithm
The core classification algorithm used is **Naive Bayes**. This choice was based on its simplicity, transparency, efficiency, and often strong performance despite its "naive" assumption of feature independence. 

### Data Preprocessing
Several preprocessing steps were applied to the data: 
1.  **Handling Missing Values:** Missing values, represented by '?', were replaced with `pd.NA` and then rows containing NA values were dropped. 
2.  **Feature Binning (Discretization):** Continuous numerical features were discretized into categorical bins to improve compatibility with Naive Bayes and potentially enhance performance. The following attributes were binned: 
    * `age`: Binned into 9 categories.
    * `thalach`: Binned into 7 categories.
    * `chol`: Binned into 9 categories.
    * `trestbps`: Binned into 8 categories.
    * `oldpeak`: Binned into 8 categories.
3.  **Correlation Analysis:** Pearson correlation was calculated. Attributes with an absolute correlation greater than 0.8 with other features (excluding the target 'num') were dropped to reduce multicollinearity. 
4.  **Dichotomization (for Model 1):** For the binary classification model, the target attribute `num` was dichotomized. Patients with `num = 0` are considered healthy, while patients with `num` values 1, 2, 3, or 4 are grouped together as sick (represented by 1). 

### Model Evaluation
The models were evaluated using the following metrics:
* **Accuracy:** The proportion of correctly classified instances. 
* **Confusion Matrix:** To analyze the types of correct and incorrect predictions for each class. 
* **Standard Deviation of Accuracy:** Calculated over multiple cycles (5 cycles in the experiments) to assess the stability and consistency of the model's performance. 

**Experimental Setup:**
* **Model 1:** The dataset was split into a training set (75%) and a testing set (25%). 
* **Model 2:** The script `model2.py` uses the entire dataset for testing in each cycle after shuffling, calculating probabilities based on the current state of the (shuffled) global `df`. This approach is used to evaluate performance across different data orderings.

## Models & Results

### Model 1: Binary Classification (Healthy/Sick)
* **Objective:** To accurately determine if a patient is healthy or sick.
* **Key Preprocessing:** Dichotomization of the target variable `num`.
* **Performance:** 
    * Average Accuracy: Approximately **85%**.
    * Standard Deviation of Accuracy: Approximately **1.90**.
    * The model demonstrates high stability and accuracy, making it suitable for initial screening.

### Model 2: Disease Stage Classification
* **Objective:** To classify the specific stage of heart disease (values 0, 1, 2, 3, 4).
* **Performance:** 
    * Average Accuracy: Approximately **64%**. (The PDF shows one run with 64%, the script runs 5 cycles and the average might differ slightly).
    * This model provides more detailed diagnostic information but at the cost of lower overall accuracy compared to the binary model. The performance is noted to be sensitive to dataset size.

**Conclusion:** The choice between models depends on the clinical context. Model 1 offers high accuracy for binary presence/absence of disease, while Model 2 provides a more granular staging but with reduced accuracy. 

## Repository Structure

The project is organized as follows:

heart_disease_pred-master/
│
├── files/
│   └── processedcleveland.csv       # The primary dataset used for model training and evaluation.
│
├── reports/
│   └── Raport - “Predykcja stadium choroby serca”.pdf  # Comprehensive project report detailing methodology,
│                                                       # analysis, and results (in Polish).
│
├── model1.py                        # Python script implementing Model 1:
│                                    # Binary classification (Healthy/Sick).
│
├── model2.py                        # Python script implementing Model 2:
│                                    # Disease stage classification (Stages 0-4).
│
└── README.md                        # This file, providing an overview of the project.

## Installation

To set up the project environment and run the models, you'll need Python 3.x and the following Python libraries:

* **pandas:** For data manipulation and CSV file handling.
* **numpy:** For numerical operations.
* **scikit-learn:** For machine learning utilities, specifically `train_test_split` used in `model1.py`.

You can install these dependencies using `pip`. It's recommended to use a virtual environment for managing project dependencies.

1.  **Ensure Python is installed.**
    You can download it from [python.org](https://www.python.org/).

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn
    ```

## Usage

After installing the necessary dependencies, you can run the prediction models as follows:

1.  **Clone or download the repository:**
    If you haven't already, get a local copy of the project.

2.  **Navigate to the project directory:**
    Open your terminal or command prompt and change to the root directory of the project (e.g., `heart_disease_pred-master`).
    ```bash
    cd path/to/heart_disease_pred-master
    ```

3.  **Run the models:**
    The dataset `processedcleveland.csv` is expected to be in the `files/` subdirectory relative to the scripts.

    * **To run Model 1 (Binary Classification - Healthy/Sick):**
        ```bash
        python model1.py
        ```
        This script will perform 5 cycles of training and testing. For each cycle, it will print the accuracy and the confusion matrix. Finally, it will display the average accuracy and standard deviation across all cycles.

    * **To run Model 2 (Disease Stage Classification - Stages 0-4):**
        ```bash
        python model2.py
        ```
        Similar to Model 1, this script will run for 5 cycles, printing the accuracy and confusion matrix for each. It will conclude with the average accuracy and standard deviation.

**Expected Output:**
For both models, the console output will show:
* Processing messages.
* For each of the 5 cycles:
    * Cycle number.
    * Calculated accuracy for that cycle (e.g., "Accuracy: 85%").
    * The confusion matrix.
* A summary at the end:
    * Average accuracy across all cycles.
    * Standard deviation of the accuracy.

## Detailed Report

For a comprehensive understanding of this project, including an in-depth look at the research objectives, data preprocessing steps, model implementation details, experimental setup, results analysis, and overall conclusions, please refer to the full project report.

The report provides a thorough exploration of the methodologies employed and offers a deeper discussion of the findings summarized in this README.

📄 **[View the Full Project Report (PDF)](reports/Raport%20-%20%E2%80%9CPredykcja%20stadium%20choroby%20serca%E2%80%9D.pdf)** [cite: 2]

**Note:** The detailed report is written in Polish. [cite: 2]

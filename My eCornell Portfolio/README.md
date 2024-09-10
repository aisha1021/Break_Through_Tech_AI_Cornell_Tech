# Machine Learning Project: Glassdoor Job Reviews Dataset

## Overview

This project focuses on implementing a machine learning model following the machine learning life cycle. The primary goal is to predict job ratings based on employee reviews from the Glassdoor dataset. We will use text-based natural language processing (NLP) and numerical feature-based techniques to build a predictive model.

## Project Structure

### 1. **Dataset**
The dataset used is the *Glassdoor Job Reviews Dataset*, which contains reviews from employees about various companies. The dataset includes the following features:
- **Firm**: Name of the company.
- **Date Review**: Date of the review.
- **Job Title**: Position of the reviewer.
- **Current**: Whether the employee is currently working for the company.
- **Location**: Location of the job.
- **Overall Rating**: Rating of the company (Label).
- **Work-life balance, Culture values, Diversity inclusion, Career opportunities, Compensation & benefits, Senior management**: Numerical ratings from the reviews.
- **Headline, Pros, Cons**: Text-based reviews.

### 2. **Problem Definition**
The goal is to predict the **overall job rating** based on text reviews and other features. This is a **supervised classification problem** where the rating ranges from 1 to 5.

### 3. **Data Preparation**
- Load and clean the dataset using **Pandas**.
- Perform feature engineering by creating new features such as `review_text` by combining the headline, pros, and cons.
- Handle missing data, converting numerical columns into appropriate data types.
- Perform **Natural Language Processing (NLP)** using **TF-IDF vectorization** on text reviews.

### 4. **Machine Learning Model**
The project uses a **Neural Network** model implemented using **TensorFlow/Keras**:
- The model consists of an input layer, three hidden layers, and an output layer using the softmax activation function for classification.
- Other models considered include **Logistic Regression** and **Random Forest Classifier**.

### 5. **Model Training & Evaluation**
- Train the model using 75% of the data, with 25% of the training data reserved for validation.
- Evaluate the model's performance on the test set using accuracy, precision, recall, F1-score, and confusion matrix.
- Perform hyperparameter tuning and model selection to improve performance.

### 6. **Results**
- The model achieved an accuracy of around **50.7%** on the test set, with further improvements possible through additional feature engineering and hyperparameter tuning.
- The neural network model performed well in predicting higher job ratings but had challenges with lower ratings.

### 7. **Conclusion**
This project demonstrates the full machine learning life cycle, from data preparation and feature engineering to model training and evaluation. The model provides insights into how companies can improve employee satisfaction by understanding the factors contributing to overall job ratings.

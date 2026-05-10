# pymlFinal – Dawson Wright

## Project Overview

Handwriting recognition is a challenging problem, especially when attempting to identify exact characters. Many real-world handwriting recognition systems rely on advanced deep learning techniques such as Convolutional Neural Networks (CNNs).

For this project, I simplified the problem by focusing on **binary classification** rather than full character recognition. Instead of predicting the exact handwritten character, this system determines whether the input is:

- **Alphabetic**
- **Numeric**

This narrower scope allowed me to focus on core machine learning concepts such as:

- Feature engineering
- Dataset balancing
- Model comparison
- Model deployment

---

## Project Goals

The primary objectives of this project were to:

- Build a binary classifier for handwritten input
- Use traditional machine learning models learned in class
- Work with real-world handwriting datasets
- Deploy the final model in an interactive web application

---

## Datasets Used

This project uses two handwriting datasets sourced from Kaggle:

- Handwritten English letters and digits dataset
- Additional handwritten digit dataset for balancing numeric samples

### Labeling Strategy

Labels were assigned automatically based on folder names:

- Folders named `0–9` → **Numeric**
- All other folders (e.g. `A`, `b`, `A_caps`) → **Alphabetic**

### Examples:

| Folder Name | Classification |
|------------|----------------|
| `7`        | Numeric        |
| `A`        | Alphabetic     |
| `A_caps`   | Alphabetic     |

This automated labeling system eliminated the need for manual annotation.

---

## Feature Engineering

Because this project does not use deep learning, images were converted into numerical feature vectors manually.

### Features extracted include:

- Flattened pixel values
- Mean brightness
- Standard deviation
- Horizontal projection values
- Vertical projection values

These features help represent:

- Shape
- Stroke distribution
- Character structure

---

## Models Tested

Several machine learning models were compared:

- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

### Results Summary

- **Random Forest:** Strong overall performance
- **KNN:** Best numeric recall
- **Logistic Regression:** Lower performance
- **Decision Tree:** Less stable overall

Although Random Forest performed well, **KNN provided the best balance between alphabetic and numeric classification**, so it was selected for deployment.

---

## Key Improvements

### Initial Challenge:
The original model struggled significantly with identifying numeric samples.

### Solution:
- Added a second numeric dataset
- Balanced training data
- Improved class distribution

### Result:
This significantly improved numeric recall and overall system balance.

---

## Deployment

The final model was deployed using:

- **Flask** (web application framework)
- **Render** (cloud deployment platform)

### Web App Features:
- Users draw a character using their mouse
- Model predicts whether the input is alphabetic or numeric
- Confidence score is displayed using prediction probabilities

---

## Final Outcome

The completed system:

- Classifies handwritten user input
- Performs reasonably well across both classes
- Provides interactive predictions through a deployed web application

---

## Lessons Learned

This project reinforced several important machine learning principles:

- Feature engineering is critical
- Balanced datasets significantly impact performance
- Different models have different strengths and weaknesses
- Deployment requires consistency between training and inference pipelines

---

## Future Improvements

Potential next steps include:

- Predicting exact characters instead of binary categories
- Implementing CNN-based deep learning approaches
- Improving canvas preprocessing for better user-drawn predictions
- Enhancing UI/UX for deployment

---

## Technologies Used

- Python
- Flask
- Render
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- Joblib

---

# Self-Grading Rubric – pymlFinal Project

---

# 1. Problem Definition & Motivation

## a. Clearly Defined Problem Statement
**Score (1–5):** 5

**Reasoning:**  
I made sure that I explained exactly what my problem was, do make a binary classifier instead of full character recognition.

---

## b. Understanding of Domain and Context
**Score (1–5):** 4

**Reasoning:**  
Explained that I wanted to lean towards core concepts rather than CNNs

---

# 2. Data Acquisition & Preprocessing

## a. Data Cleaning and Handling Missing Values
**Score (1–5):** 5

**Reasoning:**  
I made sure that the datasets I used used pictures that looked similar in terms of format, and I was able to divide the two datasets folders in an effective way

---

## b. Feature Engineering and Selection
**Score (1–5):** 5

**Reasoning:**  
Was able to successfully manually extract features without the datasets providing them for me

---

## c. Data Normalization and Scaling
**Score (1–5):** 4

**Reasoning:**  
Made sure I mad the pictures on all samples be the same size so that photo size did not effect the result

---

# 3. Model Selection, Evaluation & Justification

## a. Selection of Appropriate ML Algorithms
**Score (1–5):** 5

**Reasoning:**  
Tested a variety of models and gave good reasoning for the one I chose

---

## b. Model Training and Tuning
**Score (1–5):** 4

**Reasoning:**  
Worked a lot on balancing the datasets and tuning the features

---

## c. Evaluation Metrics and Performance Analysis
**Score (1–5):** 5

**Reasoning:**  
Explained the preformance differences between models and what my initial struggles were before fixing them

---

# 4. Creativity and Innovation

## a. Novelty and Originality of Approach
**Score (1–5):** 3

**Reasoning:**  
Did use a more simple version of the problem, even if it was to focus on core ideas

---

## b. Exploration of Advanced Techniques (Deep Learning, etc.)
**Score (1–5):** 3

**Reasoning:**  
intentionally did not use deep learning to focus on core ideas

---

# 5. Presentation

## a. Quality of Visualizations and Insights
**Score (1–5):** 4

**Reasoning:**  
Had few visual plots to represent the data, but I feel I did a good job explaining the data in a simple way.

---

## b. Ability to Communicate Results Effectively
**Score (1–5):** 5

**Reasoning:**  
Explained my final results and demo in a concice and simple way.

---

# Final Self-Assessment

## Total Score:
**52/60**

---

## Author

**Dawson Wright**

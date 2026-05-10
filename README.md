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

## Author

**Dawson Wright**

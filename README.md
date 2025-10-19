# Explainable Glaucoma Detection from Retinal Images and Morphological features

## Project Summary
### Team
- Leonardo Canuto Junior
### Language
Python
### Libraries
Numpy, Pandas, Scikit Learn, Matplotlib

## Problem Context
**Introduction**

Glaucoma is a progressive eye disease that can lead to irreversible vision loss if not detected early. It is one of the leading causes of blindness worldwide. Detecting glaucoma at an early stage is challenging because its symptoms are not easily noticeable, and the differences between healthy and glaucomatous eyes can be subtle in retinal images.

**The Problem**

We have to train a machine learning model capable of classifying retinal images to distinguish between healthy and glaucomatous eyes. Some of the main challenges of this task include:
- High variability between patients' eyes.
- Low-quality or noisy retinal images.
- Visual similarities between healthy and glaucomatous optic nerves, especially when glaucoma is in an early stage. Manual diagnosis by experts is time-consuming and prone to subjectivity, which makes automated classification methods highly valuable.

Figure 1 shows glaucomatous and healthy eye fundus images centered on the optic nerve head.

<img width="656" height="303" alt="image" src="https://github.com/user-attachments/assets/d9086d28-ed94-4d83-aab3-a2630e6af289" />

Figure 1: Glaucomatous (left) and healthy (right) eye fundus images.

**Proposed Solution**

Convolutional Neural Networks (CNNs) offer a powerful solution for image classification tasks. By training a CNN on a large set of labeled retinal images, the network can automatically learn to extract relevant features and differentiate between healthy and glaucomatous cases.

Key advantages of using CNNs:
- Automatic feature extraction.
- High potential accuracy.
- Fast image classification once trained.

**Challenges and Limitations**

Despite the potential of neural networks, there are important limitations:

- Risk of overfitting, especially with small datasets.
- Limited interpretability of the models (“black box” problem).
- Need for extensive clinical validation to ensure real-world reliability.

**Classification based on morphological features**

To overcome the black box problem and enhance explainability, interpretable models such as decision trees will be trained using the extracted morphological features. Decision trees can help doctors understand the reasoning behind each classification.

Morphological features such as:

- Cup and disc areas
- Vertical cup-to-disc ratio
- Vessels curvature

These features are clinically relevant and can provide understandable indicators of glaucoma risk. Figure 2 shows morphological features related with the optic cup and disc, which are two relevant structures found in the optic nerve head of the retina.

# Solution Report

## 1. Problem description 
### 1.1. Dataset characteristics

### 1.2. Classification details

## 2. Methodology 
### 2.1. Data preprocessing 

### 2.2. Justification for the opted models

### 2.3. Selection and evaluation metrics

### 2.4. Evaluation methodology

## 3. Results and Discussion
### 3.1. Model selection and evaluation

### 3.2. Description and interpretation of the results

## 4. Conclusion




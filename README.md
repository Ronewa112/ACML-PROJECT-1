# 🧠 CNN-Based Diagnosis of Lung Opacity and Viral Pneumonia from X-ray Images

This repository contains the implementation of a deep learning project aimed at diagnosing **Lung Opacity** and **Viral Pneumonia** from chest **X-ray images** using **Convolutional Neural Networks (CNNs)**.

---

## 🩻 Project Overview

Lung diseases such as pneumonia and opacity-related conditions are among the leading causes of death globally. This project applies CNN architectures to automate the classification of chest X-ray images into:

- **Normal**
- **Lung Opacity**
- **Viral Pneumonia**

Three custom CNN models were built and evaluated using a dataset of 3,475 X-ray images to determine the most accurate and generalizable architecture.

---

## 🎯 Objectives

- Design and implement CNN models for multi-class classification of lung conditions.
- Compare models using performance metrics: Accuracy, Precision, Recall, and F1-Score.
- Determine the impact of architecture complexity and hyperparameters on classification performance.
- Provide a diagnostic aid to support medical professionals in faster, more accurate diagnoses.

---

## 🛠️ Methodology

- **Dataset Source**: [Kaggle – Lung Disease X-rays](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data)
- **Preprocessing**:
  - Resized images to 64×64 pixels
  - Normalization and label encoding
  - Data augmentation (rotation, flipping, zooming)
- **Tools & Libraries**: Python, PyTorch, TensorFlow, Scikit-learn, Google Colab

### 📓 Run It on Google Colab

You can run the full project using the interactive notebook here:  
👉 [Google Colab Notebook](https://colab.research.google.com/drive/1uu_NuQfXvkLH71h2QpWVvnxSu9lQO_21?usp=sharing)

---

## 🧪 CNN Architectures

| Model | Layers | Accuracy | Precision | Recall | F1-Score |
|-------|--------|----------|-----------|--------|----------|
| Model 1 | 3 Conv Layers | 78.20% | 79.03% | 78.20% | 77.85% |
| Model 2 | 2 Conv Layers | 80.50% | 80.51% | 80.50% | 80.45% |
| Model 3 | 4 Conv Layers | **81.26%** | **81.32%** | **81.26%** | **81.23%** |

> **Model 3** achieved the best performance and generalization across all evaluation metrics.

---

## 📊 Results

- **Best Model**: Model 3 (4 convolutional layers)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Epochs**: 5 (to reduce overfitting)
- **Evaluation**: Accuracy/loss curves, confusion matrices, ROC curves

---

## 🔍 Future Work

- Expand dataset and improve class balance
- Apply **transfer learning** (e.g., ResNet, EfficientNet)
- Integrate explainability techniques such as **Grad-CAM**
- Deploy model in real-time diagnostic systems

---

## 👨‍💻 Authors

This project was submitted as part of the **Adaptative Computation and Machine Learning (COMS7047A)** module under the **School of Computer Science and Applied Mathematics**, **Faculty of Science**.

- **Siphosethu Lucas Mathonsi** – *3004983*
- **Ntanganedzeni Mandiwana** – *2356380*
- **Mueletshedzi Mukhaninga** – *3019980*
- **Ronewa Nephiphidi** – *3000595*

**Date Submitted**: May 30, 2025

---

## 📁 Repository Structure


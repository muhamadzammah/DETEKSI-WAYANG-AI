# Wayang AI Detection System

**Wayang AI Detection** is an artificial intelligence system based on Computer Vision and Deep Learning, designed to automatically classify wayang (traditional Indonesian puppet) characters from digital images. The system combines Convolutional Neural Networks (CNN) with Transfer Learning using MobileNetV2, and is equipped with digital image processing techniques and an interactive web interface.

The main goal of this system is to support the preservation of Indonesian wayang culture through AI technology, while also serving as an informative and user-friendly educational tool.

## 🎯 System Objectives

This system aims to:

* Classify wayang characters based on digital images
* Identify wayang characters quickly and accurately
* Display character descriptions as educational information
* Visualize image processing stages for model analysis
* Provide a real-world implementation of CNN and Transfer Learning on local cultural data

## ⚙️ How the System Works

The general workflow of the system is as follows:

### 1. Image Input

Users upload wayang images through the web interface.

### 2. Image Preprocessing

The system performs several image processing steps:

* Resize image to 224×224 pixels
* Convert to grayscale
* Thresholding (Otsu)
* Edge detection (Canny, Sobel, Prewitt)
* Morphological operations (opening and closing)

### 3. CNN Classification

The preprocessed image is processed by a CNN model based on MobileNetV2 to determine the wayang character class.

### 4. Output Results

The system displays:

* Wayang character name
* Confidence score
* Character description
* Visualization of image processing results

## 🏗️ System Architecture

The system consists of three main components:

### Backend (FastAPI)

* Handles request and response
* Manages image upload
* Performs image preprocessing
* Runs AI model prediction
* Returns results in JSON format

### AI Model (TensorFlow & Keras)

* Uses MobileNetV2 pretrained on ImageNet
* Fine-tuned on final layers to adapt to wayang characteristics
* Uses class weighting for imbalanced datasets

### Frontend (HTML, Bootstrap, JavaScript)

* Responsive and modern web interface
* Image upload and preview
* Displays prediction results and image visualization

## 📊 Methods Used

* Convolutional Neural Network (CNN)
* Transfer Learning (MobileNetV2)
* Model Fine-Tuning
* Data Augmentation
* Digital Image Processing
* Class Weighting
* Early Stopping & Learning Rate Scheduler

## 🖼️ Dataset

The dataset consists of images of wayang characters from various classes such as Arjuna, Bima, Semar, Gatotkaca, and others. The dataset is organized into folders based on class labels and used for training the model.

**Note:** The dataset is not included in this repository due to size limitations.

## 🚀 System Advantages

* Uses a lightweight and efficient model (MobileNetV2)
* Performs well on limited datasets
* Provides image processing visualization
* Interactive web interface
* Easy to develop and deploy

## 📚 Future Improvements

* Adding more wayang character classes
* Integrating Grad-CAM for model interpretability
* Cloud deployment (Render, Railway, Docker)
* Mobile version or Progressive Web App (PWA)

## 🇮🇩 Cultural Contribution

This system is expected to contribute to the digital preservation and promotion of wayang as an important Indonesian cultural heritage through the application of modern artificial intelligence technology.

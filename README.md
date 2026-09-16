# 🧠 Brain MRI Tumor Analyzer

An AI-powered **Brain MRI Tumor Classification and Analysis** web application that analyzes MRI brain images using a trained **Deep Residual CNN exported to ONNX**, combined with computer vision techniques for brain extraction, tissue segmentation, suspicious-region analysis, and visual interpretation.

🌐 **Live Demo:** https://brain-mri-classifier-rl1s.onrender.com

---

## 📌 Project Overview

**Brain MRI Tumor Analyzer** is a web-based computer vision and deep learning application designed to analyze brain MRI images and classify them as:

* 🧠 **Normal**
* ⚠️ **Tumor**

The system combines **Deep Learning** with traditional **Computer Vision** techniques to provide not only a classification result but also additional image-analysis information such as suspicious regions, tumor-area estimation, asymmetry, edema-like regions, necrotic regions, and ring-enhancement characteristics.

> **Important:** This project is intended for educational and research purposes. It is **not a medical diagnostic system** and should not be used as a substitute for evaluation by a qualified medical professional.

---

## ✨ Key Features

### 🤖 AI-Based MRI Classification

* Uses a trained Deep Residual CNN.
* Model is exported to **ONNX** for lightweight CPU inference.
* Produces normal/tumor probabilities.
* Calculates a classification confidence score.

### 🖼️ MRI Image Processing

The application performs multiple image-processing stages:

1. Grayscale conversion
2. Image resizing
3. CLAHE contrast enhancement
4. Bilateral filtering
5. Gaussian smoothing
6. Brain extraction / skull stripping
7. K-means tissue segmentation
8. Suspicious-region extraction
9. Tumor-region analysis
10. Visualization and overlay generation

### 🔍 Tumor Region Analysis

For images classified as tumor-positive, the system analyzes several image characteristics, including:

* Tumor-area estimation
* Suspicious regions
* Brain-area measurement
* Area ratio
* Intensity characteristics
* Contrast ratio
* Brain asymmetry
* Edema-like regions
* Necrotic-region estimation
* Ring-enhancement characteristics
* Border irregularity

### 🔥 Class Activation Map (CAM)

The application generates a **Class Activation Map** to highlight image regions that contribute to the tumor classification.

This provides an additional visual interpretation of the model's prediction.

### 📤 Multiple Image Upload

The API supports:

* JPG
* JPEG
* PNG
* Up to **4 images per request**
* Maximum individual upload size of **10 MB**

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     User / Doctor    │
                    │   Upload MRI Image   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Flask Web App     │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Input Validation  │
                    │  JPG / PNG Checking  │
                    └──────────┬───────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │       Preprocessing         │
                │                             │
                │ • Grayscale                 │
                │ • Resize 256×256            │
                │ • CLAHE                     │
                │ • Bilateral Filter          │
                │ • Gaussian Blur             │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │       Brain Extraction      │
                │       / Skull Stripping     │
                │                             │
                │ • Otsu Thresholding         │
                │ • Morphological Closing     │
                │ • Connected Components      │
                │ • Hole Filling              │
                │ • Erosion                   │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │     Tissue Segmentation     │
                │        K-Means (K=5)        │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │      ONNX Deep Learning     │
                │      Residual CNN Model     │
                │                             │
                │       128 × 128 Input       │
                └──────────────┬──────────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
             ┌─────────────┐     ┌─────────────┐
             │ Prediction  │     │     CAM     │
             │             │     │ Heatmap     │
             │ Normal /    │     └──────┬──────┘
             │ Tumor       │            │
             └──────┬──────┘            │
                    └──────────┬────────┘
                               ▼
                ┌─────────────────────────────┐
                │   Region & Biomarker        │
                │        Analysis             │
                │                             │
                │ • Tumor Region              │
                │ • Edema                     │
                │ • Necrosis                  │
                │ • Ring Enhancement          │
                │ • Asymmetry                 │
                │ • Area / Contrast           │
                └──────────────┬──────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Visualization & JSON │
                    │       Results        │
                    └──────────────────────┘
```

---

## 🧠 Deep Learning Model

The project uses a trained **Deep Residual CNN** that has been exported to **ONNX format**.

ONNX Runtime is used for inference instead of loading the original PyTorch model.

### Why ONNX?

Using ONNX Runtime provides several deployment advantages:

* Lower memory requirements
* CPU-based inference
* Faster and lightweight deployment
* Easier deployment on cloud platforms
* No requirement for a GPU during inference

The classifier uses a `128 × 128` normalized grayscale input and produces two class probabilities:

```text
Class 0 → Normal
Class 1 → Tumor
```

The implementation applies softmax to the model logits to obtain the probabilities.

---

## 🔬 Computer Vision Pipeline

### 1. Preprocessing

The input MRI is converted to grayscale and resized to:

```text
256 × 256
```

Contrast enhancement is performed using:

```text
CLAHE
```

Noise reduction is then performed using:

```text
Bilateral Filter
        ↓
Gaussian Blur
```

---

### 2. Brain Extraction

The system attempts to remove non-brain regions using image-processing operations.

The process includes:

* Otsu thresholding
* Morphological closing
* Connected-component analysis
* Largest-component selection
* Flood filling
* Morphological erosion

The resulting mask is used to isolate the brain region.

---

### 3. Tissue Segmentation

K-means clustering is applied to the extracted brain region.

Default:

```text
K = 5
```

This separates different intensity groups within the brain image and provides a tissue segmentation map.

---

### 4. Tumor Region Extraction

For tumor-positive predictions, suspicious regions are determined using a combination of:

* Model activation information
* Image intensity
* Brain mask
* Morphological operations

The system attempts to identify high-intensity regions that may correspond to suspicious areas.

---

### 5. Biomarker / Region Analysis

The application calculates image-derived measurements such as:

```text
Brain Area
Tumor Area
Suspicious Regions
Area Ratio
Asymmetry
Intensity Ratio
Contrast Ratio
Ring Score
Edema Ratio
Necrotic Ratio
Border Irregularity
```

These measurements are intended to provide additional computational information about the MRI image.

---

## 🔥 Class Activation Map

The deep-learning model also provides intermediate feature maps.

The application uses the classifier's fully connected weights together with the feature maps to generate a **Class Activation Map (CAM)**.

Conceptually:

```text
CNN Feature Maps
       │
       ▼
Class-specific weights
       │
       ▼
Weighted Feature Combination
       │
       ▼
ReLU
       │
       ▼
Normalization
       │
       ▼
CAM Heatmap
```

The heatmap is resized and used during suspicious-region analysis and visualization.

---

## 🌐 API Endpoints

### Home

```http
GET /
```

Loads the web application interface.

---

### Health Check

```http
GET /health
```

Returns the server and model status.

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "server": "MRI Brain Tumor Analyzer"
}
```

---

### Pipeline Self-Test

```http
GET /test
```

Runs a synthetic image through the analysis pipeline to verify that the server and model are functioning.

---

### MRI Analysis

```http
POST /analyze
```

Upload MRI images using the `files` field.

Supported formats:

```text
.jpg
.jpeg
.png
```

Maximum:

```text
4 images
```

The endpoint returns analysis results for the submitted images.

---

## 📂 Project Structure

```text
Brain-MRI-Classifier/
│
├── app.py
│
├── model/
│   ├── tumor_classifier.py
│   ├── brain_tumor_detector.onnx
│   └── fc_weights.npy
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── Procfile
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

The repository currently contains the Flask application, model directory, static assets, templates, deployment configuration, and Python dependency files.

---

## 🛠️ Technologies Used

| Technology          | Purpose                              |
| ------------------- | ------------------------------------ |
| Python              | Core programming language            |
| Flask               | Web application framework            |
| ONNX Runtime        | Deep-learning model inference        |
| OpenCV              | Computer vision and image processing |
| NumPy               | Numerical computation                |
| Pillow              | Image handling                       |
| Gunicorn            | Production WSGI server               |
| HTML/CSS/JavaScript | Frontend interface                   |
| Render              | Cloud deployment                     |

The project's dependency configuration includes Flask, ONNX Runtime, Gunicorn, OpenCV, NumPy and Pillow.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sonam-1986/Brain-MRI-Classifier.git
```

```bash
cd Brain-MRI-Classifier
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

The application runs using the `PORT` environment variable when supplied and binds to `0.0.0.0`, making it suitable for deployment environments.

Open:

```text
http://127.0.0.1:5001
```

---

## ☁️ Deployment

The application is configured for deployment using **Gunicorn**.

The included deployment command is:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1
```

This configuration is defined in the project's `Procfile`.

### Render Deployment

The project can be deployed as a Python web service on Render.

Typical configuration:

```text
Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1
```

### Live Application

**Brain MRI Classifier:**

https://brain-mri-classifier-rl1s.onrender.com

---

## 📊 Processing Flow

```text
MRI Image
    │
    ▼
Image Validation
    │
    ▼
Preprocessing
    │
    ├── Grayscale
    ├── Resize
    ├── CLAHE
    ├── Bilateral Filtering
    └── Gaussian Filtering
    │
    ▼
Brain Extraction
    │
    ▼
K-Means Tissue Segmentation
    │
    ▼
Deep Residual CNN
    │
    ▼
Normal / Tumor
    │
    ├───────────────┐
    ▼               ▼
Confidence        CAM
                    │
                    ▼
              Region Analysis
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Edema      Necrosis    Ring Pattern
        │           │           │
        └───────────┼───────────┘
                    ▼
              Final Results
```

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Develop an AI-assisted brain MRI classification system.
2. Apply deep learning to MRI image classification.
3. Combine deep learning with traditional computer vision.
4. Extract and analyze suspicious image regions.
5. Generate visual explanations using CAM.
6. Build a lightweight web application for model inference.
7. Deploy the application as an accessible cloud-based service.

---

## 🔐 Input Validation & Security

The application includes basic upload validation:

* Only JPG/JPEG/PNG images are accepted.
* File uploads are limited to 10 MB.
* Maximum four files can be submitted in one request.
* Uploaded filenames are sanitized using `secure_filename`.

These restrictions help prevent unsupported files from entering the image-processing pipeline.

---

## ⚠️ Medical Disclaimer

This project is an **educational/research prototype**.

The predictions, confidence values, segmentation results, and calculated image features should **not be interpreted as a medical diagnosis**.

MRI interpretation and tumor diagnosis must be performed by qualified healthcare professionals using appropriate clinical information, imaging protocols, and validated diagnostic systems.

---

## 🔮 Future Enhancements

Potential future improvements include:

* Multi-class tumor classification
* Tumor-type classification
* MRI sequence recognition
* Improved segmentation using U-Net/SegFormer
* 3D MRI volume analysis
* Improved explainability using Grad-CAM
* Model performance dashboard
* Patient-independent evaluation
* Larger and more diverse datasets
* External validation on independent datasets
* Authentication and secure patient-data handling
* GPU-based inference
* Model monitoring and performance tracking

---

## 📚 Applications

The project can be used as a foundation for:

* Computer Vision projects
* Deep Learning projects
* Medical Image Analysis research
* AI/ML academic projects
* MRI image classification experiments
* Explainable AI demonstrations
* College final-year projects

---

## 👩‍💻 Author

**Sonam-1986**

GitHub:

https://github.com/Sonam-1986

Project:

https://github.com/Sonam-1986/Brain-MRI-Classifier

---

## ⭐ Support

If you find this project useful for learning or research, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

Add an appropriate open-source license to the repository before distributing the project publicly.

Recommended options include:

* MIT License
* Apache License 2.0
* GPL-3.0

Choose the license according to how you want others to use, modify, and redistribute the project.

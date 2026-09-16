# 🧠 Brain MRI Tumor Classifier

An AI-powered web application for analyzing **Brain MRI images** using Deep Learning and Computer Vision techniques.

## 🚀 Live Demo

[Brain MRI Tumor Classifier](https://brain-mri-classifier-rl1s.onrender.com?utm_source=chatgpt.com)

## 📌 Features

* 🧠 Brain MRI image analysis
* 🤖 Deep Learning-based tumor classification
* 🔬 Image preprocessing and brain extraction
* 📊 Suspicious region analysis
* 🔥 Class Activation Map (CAM) visualization
* 🌐 Simple Flask web interface
* 📤 Supports multiple MRI images

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **ONNX Runtime**
* **OpenCV**
* **NumPy**
* **Pillow**
* **Deep Residual CNN**
* **Gunicorn**

## 🔄 Workflow

```text
MRI Image
   ↓
Preprocessing
   ↓
Brain Extraction
   ↓
Deep Learning Model
   ↓
Tumor / Suspicious Region Analysis
   ↓
Prediction + Visualization
```

## 📂 Project Structure

```text
Brain-MRI-Classifier/
├── app.py
├── model/
│   ├── tumor_classifier.py
│   ├── brain_tumor_detector.onnx
│   └── fc_weights.npy
├── static/
├── templates/
├── requirements.txt
├── Procfile
└── runtime.txt
```

## ▶️ Run Locally

```bash
git clone https://github.com/Sonam-1986/Brain-MRI-Classifier.git
cd Brain-MRI-Classifier
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5001
```

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis or clinical evaluation.

## 👩‍💻 Author

**Sonam-1986**

[GitHub Repository](https://github.com/Sonam-1986/Brain-MRI-Classifier?utm_source=chatgpt.com)

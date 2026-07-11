# 🩺 AI Pneumonia Detection from Chest X-ray

A deep learning application that detects **Pneumonia** from Chest X-ray images using **MobileNetV2 Transfer Learning** and provides predictions through a **Streamlit** web application.

---

## Features

- Chest X-ray image upload
- Real-time AI prediction
- Confidence score
- AI-generated medical report
- Streamlit web interface
- MobileNetV2 Transfer Learning

---

## Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow

---

## Project Workflow

Chest X-ray Image

↓

Image Preprocessing

↓

MobileNetV2 Model

↓

Prediction

↓

Medical Report

---

## Folder Structure

```text
app.py
train.py
predict.py
evaluate.py
graphs.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

## Model

Transfer Learning using **MobileNetV2**

Input Size:

224 × 224

Output:

- NORMAL
- PNEUMONIA

---

## Future Improvements

- Grad-CAM
- PDF Report
- Explainable AI
- Cloud Deployment

---

## Developer

**Phani Kumar**

B.Tech Artificial Intelligence & Machine Learning
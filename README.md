# 🩺 Breast Cancer Detection & Classification using Ultrasound with HVIT-AAF

![AI](https://img.shields.io/badge/DeepLearning-TensorFlow-brightgreen)
![Ultrasound](https://img.shields.io/badge/Ultrasound-Images-blue)
![Frontend](https://img.shields.io/badge/Flask-Frontend-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An AI-powered deep learning application that detects and classifies **breast cancer** into **benign**, **malignant**, or **normal** from ultrasound images. This project integrates an innovative hybrid model (HVIT-AAF), a custom MobileViT-Transformer-enhanced CNN architecture, and a lightweight Flask frontend interface.

---

## 📌 Project Highlights

- ✅ **Architecture**: HVIT-AAF (Hybrid Vision Transformer with Adaptive Attention Fusion)
- 🧠 **Backbone**: EfficientNetB0 + Custom MobileViT Blocks
- 🧪 **Test Accuracy**: **99.11%**
- 📈 **AUC Score**: **99.84%**
- 🎯 **Training Framework**: TensorFlow 2.x
- 🖼️ **Grad-CAM & LIME** based visualization for interpretability
- 💻 **Frontend**: Flask-based web interface
- 🚀 Trained on **Google Colab** (P100 GPU) and tested on **Kaggle**

---

## 🧬 Dataset

- **Original Dataset**:  
  Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). _Dataset of breast ultrasound images_. Data in Brief, 28, 104863.  
  📚 [DOI Link](https://doi.org/10.1016/j.dib.2019.104863)

- **Augmented Dataset (Balanced)**  
  🔗 [Kaggle Link](https://www.kaggle.com/datasets/salmanbnr/breast-cancer-ultrasound-images/data)

---

## 🛠️ My Workflow (Custom Dataset Preparation)

1. **Removed Mask Images** from BUSI dataset for classification-only task
2. Uploaded the clean dataset to **Google Drive**
3. Applied **data augmentation** using custom `ImageDataGenerator`:
   - Rotation, Zoom, Shift, Flip, Shear, Noise
4. Made dataset **balanced** and expanded it to ~5600 images
5. Saved and uploaded the **augmented dataset to Kaggle**
6. Trained the model on Kaggle using **TensorFlow + P100 GPU**
7. Downloaded the final **.h5 model** and integrated with Flask **frontend**

---

## 🧠 Architecture: HVIT-AAF

> A fusion of EfficientNetB0 and Transformer-inspired MobileViT block with Adaptive Attention Fusion (AAF).

```
Input Image (300x300x3)
     ↓
EfficientNetB0 (Feature Extraction)
     ↓
Conv2D (128 filters) → Output: (10x10x128)
     ↓
MobileViT-Inspired Block (Transformer: MHSA x 4)
     ↓
Adaptive Attention Fusion (CNN + Transformer)
     ↓
Global Average Pooling
     ↓
Dense(64, activation='swish') → Dropout(0.5)
     ↓
Softmax Output (3 classes: Benign, Malignant, Normal)
```

📊 **Performance:**

| Metric    | Best Run |
| --------- | -------- |
| Accuracy  | 99.11%   |
| AUC       | 99.84%   |
| Precision | 99.11%   |
| Recall    | 99.05%   |

---

## 📷 Frontend Interface

The web interface is built using **Flask**. Users can:

- Upload an ultrasound image
- Get predictions instantly (in ~0.0094s)
- View results
- Visualize important regions using Grad-CAM and LIME

**Frontend Screenshots**  
📤 Upload Interface  
🔍 Prediction Output  
🧠 Grad-CAM Heatmap  
🧪 LIME Explanation

---

## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/Salmanbnr/BreastCancerDetection_FinalYearProject.git
   cd BreastCancerDetection_FinalYearProject
   ```

2. **Create and Activate Virtual Enviroment**

   ```bash
   python -m venv breast_venv
   # For Windows
   breast_venv\Scripts\activate
   # For macOS/Linux
   source breast_venv/bin/activate
   ```

3. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000`

---

## 💡 Features

- 📊 Balanced dataset via intelligent augmentation
- 📷 High-performance hybrid model
- 🌐 Easy-to-use frontend
- 📈 Detailed visual analysis via interpretability tools

---

## 👤 Author

**Muhammad Salman**  
📧 salmanbnr5@gmail.com  
🎓 Final Year BSCS Student — Govt Post Graduate Jahanzeb College, Swat  
🧠 Passionate about AI for HealthTech

---

## 📚 Thesis

Thesis of this Project is present inside thesis directory

---

## 📄 License

This project is licensed under the MIT License.

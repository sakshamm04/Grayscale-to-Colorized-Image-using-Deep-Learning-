---
title: Grayscale To Colorized Image Converter
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
license: mit
---

<img width="1280" height="640" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/49745482-3e17-4d29-8ab8-9e1c6ab41b68" />

# **AI-Powered Grayscale-to-Color Image Converter**

This project delivers an end-to-end solution for transforming grayscale images into vibrant colorized versions with deep learning. It is tailored for restoring historical black-and-white photographs and combines advanced AI colorization, rigorous quality assessment, and a modern web deployment pipeline.

---

## **Project Overview**

The application leverages a **pretrained Caffe** deep neural network—trained on diverse image datasets—to perform automatic colorization while preserving structural integrity. A custom regression model provides quantitative quality assessment through **SSIM** and **PSNR** metrics, ensuring reliable evaluation.

---

## **Key Features**

### **Core AI Functionality**
- **Automatic Image Colorization** – Pretrained Caffe model produces vivid, realistic colorizations.  
- **Quality Assessment** – Regression model predicts **SSIM** and **PSNR** scores.  
- **High Structural Accuracy** – Achieves **99 % SSIM**, keeping image details intact.  
- **PSNR Performance** – Current range **17–28 dB**; target **30 dB** for further improvement.

### **Technical Innovation**
- **Efficient Model Deployment** – Git LFS handles large 129 MB model files.  
- **Serialized Model Storage** – PKL files reload models instantly without retraining.  
- **Optimized Pipeline** – Streamlined workflow from upload → colorization → quality analysis.

### **Modern Web Interface**
- **Professional UI Design** with responsive grid layout.  
- **Mobile-Responsive** for desktop, tablet, and phone.  
- **Interactive File Selection** shows chosen file name & size.  
- **Real-Time Loading Indicators** via animated spinner.  
- **Comprehensive Downloads** for original, colorized images, and RGB histograms.

### **Advanced Analytics**
- **RGB Histogram Visualization** – Interactive color-distribution chart.  
- **Quality Metrics Dashboard** – Live SSIM/PSNR display.  
- **Visual Comparison** – Side-by-side original versus colorized.

### **Cloud Integration**
- **Appwrite Cloud Storage** for image persistence and sharing.  
- **HuggingFace Spaces Deployment** for production hosting.  
- **Secure File Handling** using base64 encoding in transit.

---

## **Technical Architecture**

### **Deep Learning Stack**
- **Framework**: Caffe with specialized colorization architecture.  
- **Model Size**: 129 MB pretrained weights.  
- **Input Processing**: LAB color-space conversion for accuracy.  
- **Network Layers**: Custom `class8_ab` and `conv8_313_rh`.

### **Backend Technologies**
- **Flask** + Waitress WSGI  
- **OpenCV** for image ops  
- **scikit-learn** regression model  
- **Matplotlib** for histograms

### **Frontend Technologies**
- Modern CSS & responsive design  
- JavaScript for file/loader interactivity  
- Base64 inline image rendering (no temp files)

### **Deployment Infrastructure**
- **Git LFS**, **Docker**, **HuggingFace Spaces**, **Appwrite**

---

## **Performance Metrics**

| Metric | Result |
|--------|--------|
| **SSIM** | ~ 99 % |
| **PSNR** | 17–28 dB |
| **Avg. Colorization Time** | < 10 s |
| **Model Load Time** | < 5 s |

**Evaluation Dataset**: 10 curated grayscale–color pairs covering portraits, landscapes, and historic scenes.

---

## **Enhanced User Experience**

- Smart file selection with size display  
- Loading spinner (“AI is colorizing…”)  
- One-click downloads for every output  
- “Process Another Image” restart flow  
- Clearly separated sections: Original, Colorized, Metrics, Histogram

---

## **Future Enhancements**

1. **PSNR Optimization** to surpass 30 dB.  
2. **Larger & Diverse Datasets** for robustness.  
3. **Batch Processing** of multiple images.  
4. **4K+ Resolution** support.  

Planned roadmap: user accounts, REST API, style-transfer modes, native mobile apps.

---

## **Project Impact**

This repository showcases:

- **Research-grade quality** (SSIM/PSNR evaluation)  
- **Production-ready deployment** on scalable cloud infra  
- **Professional UX** with modern design and analytics  
- **Comprehensive documentation** for quick adoption

---

## **Technologies Used**

| Category | Stack |
|----------|-------|
| Deep Learning | Caffe |
| Regression | scikit-learn (Pickle) |
| Backend | Flask, Waitress |
| Frontend | HTML / CSS / JS |
| Imaging | OpenCV, Matplotlib |
| Cloud | Appwrite, HuggingFace |
| Versioning | Git LFS |

---

## **Conclusion**

This project demonstrates the synergy of cutting-edge AI, robust engineering, and thoughtful UX in digital image restoration. By uniting a powerful Caffe-based colorization model with real-time quality metrics and a polished interface, it delivers a practical platform for historians, designers, and developers alike.

**Key Achievement**: A research-grade model packaged as an accessible web application with professional user experience and cloud-native deployment.
"""

# overwrite README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

print("README.md updated successfully.")

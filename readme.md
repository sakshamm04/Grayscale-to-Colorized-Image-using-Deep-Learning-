---
title: Grayscale To Colorized Image Converter
emoji: 🌍
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

<img width="1280" height="640" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/49745482-3e17-4d29-8ab8-9e1c6ab41b68" />

AI-Powered Grayscale to Colorized Image Converter
This project presents a comprehensive solution for converting grayscale images to vibrant colorized versions using state-of-the-art deep learning techniques. Designed for restoring historical black-and-white photographs, the system combines advanced AI colorization with rigorous quality assessment and modern web deployment.

Project Overview
The application leverages a pretrained Caffe deep neural network model, trained on diverse image datasets, to perform automatic colorization while maintaining structural integrity. A custom regression model provides quantitative quality assessment through SSIM and PSNR metrics, ensuring reliable performance evaluation.

Key Features
Core AI Functionality
Automatic Image Colorization: Utilizes a pretrained Caffe deep learning model trained on diverse datasets to produce vivid and realistic colorizations

Quality Assessment: Custom regression model predicts SSIM (Structural Similarity Index Measure) and PSNR (Peak Signal-to-Noise Ratio) scores

High Structural Accuracy: Achieves 99% structural preservation (SSIM), ensuring image details remain intact

PSNR Performance: Current range of 17-28 dB with optimization potential toward the ideal 30 dB target

Technical Innovation
Efficient Model Deployment: Git LFS integration for seamless large model file management (129MB+ model files)

Serialized Model Storage: PKL (Pickle) files enable rapid model loading without retraining, reducing computation time significantly

Optimized Pipeline: Streamlined processing workflow from upload to colorization to quality analysis

Modern Web Interface
Professional UI Design: Modern gradient-based design with responsive grid layout

Mobile-Responsive: Optimized for desktop, tablet, and mobile viewing

Interactive File Selection: Dynamic file name display with size information

Real-time Loading Indicators: Professional loading states with spinning animations

Comprehensive Downloads: One-click download for original images, colorized results, and RGB histograms

Advanced Analytics
RGB Histogram Visualization: Interactive color distribution analysis showing frequency across pixel ranges

Quality Metrics Dashboard: Real-time SSIM and PSNR score display in styled metric cards

Visual Comparison: Side-by-side original vs. colorized image comparison

Cloud Integration
Appwrite Cloud Storage: Seamless cloud storage integration for image persistence and sharing

HuggingFace Spaces Deployment: Production-ready deployment on HuggingFace's ML platform

Secure File Handling: Base64 encoding for secure image transmission and display

Technical Architecture
Deep Learning Stack
Framework: Caffe neural network with specialized colorization architecture

Model Size: 129MB pretrained model optimized for diverse image types

Input Processing: LAB color space conversion for enhanced colorization accuracy

Network Layers: Custom class8_ab and conv8_313_rh layer configurations

Backend Technologies
Flask: Lightweight web server with Waitress WSGI for production deployment

OpenCV: Advanced image processing and computer vision operations

scikit-learn: Custom regression model for quality metric prediction

Matplotlib: RGB histogram generation and visualization

Frontend Technologies
Modern CSS: Gradient backgrounds, card layouts, hover effects, and smooth animations

Responsive Design: Grid-based layout with mobile-first approach

JavaScript: Interactive file selection, loading states, and form validation

Base64 Image Handling: Efficient image display without temporary file storage

Deployment Infrastructure
Git LFS: Large file storage for model files with automatic tracking

Docker: Containerized deployment with optimized Debian base image

HuggingFace Spaces: Scalable cloud hosting with automatic builds

Appwrite: Cloud storage backend with secure API integration

Performance Metrics
Quality Assessment Results
SSIM Score: ~99% structural similarity preservation

PSNR Range: 17-28 dB (targeting 30 dB optimization)

Processing Speed: <10 seconds average colorization time

Model Loading: <5 seconds with Git LFS optimization

Evaluation Dataset
Training Data: 10 curated grayscale-colorized image pairs

Validation: Cross-validation on diverse image types and subjects

Performance: Consistent quality across portraits, landscapes, and historical photos

Enhanced User Experience
Interactive Features
Smart File Selection: Visual feedback with file name and size display

Loading States: Professional spinner with "AI is colorizing..." messaging

Universal Downloads: Download buttons for all generated content

Process Management: "Process Another Image" workflow optimization

Visual Design Elements
Modern Aesthetics: Professional gradient backgrounds and card-based layouts

Responsive Grid: Automatic adaptation to screen sizes

Hover Effects: Interactive elements with smooth transitions

Clear Sections: Dedicated spaces for each component (original, colorized, metrics, histogram)

Future Enhancements
Technical Improvements
PSNR Optimization: Advanced training techniques to achieve 30+ dB targets

Expanded Datasets: Broader evaluation with diverse image categories

Batch Processing: Multiple image processing capabilities

Higher Resolution: Support for 4K+ image colorization

Feature Roadmap
User Management: Account systems with image history

API Integration: RESTful API for third-party applications

Style Transfer: Multiple colorization styles and artistic filters

Mobile App: Native iOS/Android applications

Project Impact
This application demonstrates a complete end-to-end solution for AI-powered image restoration, combining:

Research-Grade Quality: Academic-level SSIM/PSNR evaluation metrics

Professional UI/UX: Industry-standard interface design and user experience

Production Deployment: Scalable cloud infrastructure with Git LFS optimization

Comprehensive Analytics: Visual and quantitative quality assessment tools

Innovation Highlights
Git LFS Integration: First-class large model file management for ML applications

Modern Web Design: Contemporary UI patterns with mobile-responsive layouts

Real-time Analytics: Instant quality metrics with visual histogram analysis

Cloud-Native Architecture: Seamless integration of storage, compute, and deployment services

Technologies Used
Deep Learning Framework: Caffe for neural network-based colorization

Regression Model: Custom-built using scikit-learn; stored and loaded via Pickle for efficiency

Web Technologies: Flask for backend server, HTML/CSS/JavaScript for interactive UI

Image Processing: OpenCV and Matplotlib for image manipulation and visualization

Cloud Services: Appwrite for storage, HuggingFace Spaces for deployment

Version Control: Git LFS for large model file management

Conclusion
This project exemplifies the convergence of advanced AI research, modern web development, and cloud deployment practices. By combining a powerful Caffe-based colorization model with comprehensive quality assessment and an intuitive user interface, it provides a robust platform for digital image restoration.

The efficient use of Git LFS for model storage, professional UI design, and cloud integration makes this solution suitable for both academic research and practical applications in digital restoration, creative workflows, and historical preservation projects.

Key Achievement: Successfully transformed a complex AI research project into a production-ready web application with professional-grade user experience and deployment infrastructure.

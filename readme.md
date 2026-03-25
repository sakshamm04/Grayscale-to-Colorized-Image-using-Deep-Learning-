<img width="1366" height="768" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/bfd832a8-2630-43ca-8b84-713b45c915ee" />

** If the .caffemodel file is not the full size, you can download the complete version from this link: https://drive.google.com/file/d/1s91J6rYWJLhp2Y3V64wT4klzSbR_ScG_/view?usp=sharing.

# Grayscale Image Colorizer

A web application that colorizes grayscale images using an open-source pretrained Caffe deep learning model. The focus of this project is the full application layer — integrating the model into a production-ready Flask backend, building a regression model for quality assessment, and deploying it with Docker.

---

## Overview

The application wraps a **pretrained Caffe** colorization network with a complete web pipeline. A custom regression model evaluates output quality using **SSIM** and **PSNR** metrics, and results are presented through a clean, responsive interface.

---

## Features

- **Image Colorization** — Integrates a pretrained Caffe model to produce colorized outputs from grayscale inputs
- **Quality Assessment** — Custom regression model predicts SSIM and PSNR scores per output
- **RGB Histogram** — Color distribution visualization of the colorized image
- **Side-by-Side Comparison** — Original vs colorized view with one-click downloads
- **Responsive UI** — Works across desktop, tablet, and mobile
- **Docker Deployment** — Containerized for consistent, portable deployment

---

## Technical Architecture

| Layer | Stack |
|-------|-------|
| Deep Learning | Pretrained Caffe (colorization_release_v2) |
| Quality Model | scikit-learn Linear Regression (PKL) |
| Backend | Flask + Waitress WSGI |
| Image Processing | OpenCV, Matplotlib |
| Frontend | HTML / CSS / JavaScript |
| Model Storage | Git LFS |
| Deployment | Docker |

---

## Performance

| Metric | Result |
|--------|--------|
| SSIM | ~99% |
| PSNR | 17–28 dB |
| Avg. Colorization Time | < 10s |
| Model Load Time | < 5s |

Evaluated on 10 curated grayscale–color pairs covering portraits, landscapes, and historical scenes.

---

## What I Built

The pretrained Caffe model handles colorization. Everything else was built from scratch:

- Flask application and REST endpoints
- Regression model trained on SSIM/PSNR metrics
- Full frontend with file handling, loading states, and downloads
- Git LFS integration for large model files
- Docker containerization for deployment

---

## Future Improvements

- Push PSNR above 30 dB with better training data
- Add batch image processing
- Support higher resolution inputs

# Aerial Threat Surveillance Pipeline

An enterprise-grade computer vision pipeline designed to detect, track, and log small-object threats (people and vehicles) from high-altitude aerial drone footage. 

This system utilizes Slicing Aided Hyper Inference (SAHI) to overcome standard downscaling bottlenecks in deep learning models, allowing for high-accuracy detection on 4K pixel-dense targets.

## System Architecture

The project is built using a modular, Object-Oriented architecture to separate the inference engine, spatial analytics, and data persistence layers:

*   **Inference Engine (`main.py`):** Handles the OpenCV video ingestion, CLI argument parsing, and hardware profiling (tracking Latency, FPS, and CUDA VRAM).
*   **Spatial Analytics (`analytics.py`):** An optimized temporal accumulation matrix using NumPy that generates real-time thermal threat density heatmaps without dropping pipeline FPS.
*   **Data Persistence (`logger.py`):** Implements an asynchronous debouncing logic to prevent Disk I/O bottlenecks. Metadata (timestamps, confidence scores, bounding boxes) is routed to a local SQLite database, while cropped threat images are saved to isolated blob storage.

## Tech Stack
*   **Core:** Python
*   **Computer Vision:** OpenCV, YOLOv8, SAHI (Slicing Aided Hyper Inference)
*   **Data Processing:** NumPy, PyTorch (Tensor processing)
*   **Database:** SQLite

## Installation & Execution

1. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
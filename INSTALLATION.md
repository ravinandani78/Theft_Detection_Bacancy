# Installation Guide - E2E Object Detection Pipeline with Compression

## 🎯 Overview

This guide will help you set up the complete E2E Object Detection Pipeline that includes:
- **Parallel Video Processing**: Process 5 videos simultaneously
- **Integrated Compression**: Frame-level JPEG compression with 100% preservation
- **YOLOv11 Detection**: Real-time object detection with visualization
- **MLflow Tracking**: Comprehensive experiment and performance tracking
- **Single File Solution**: Everything in `main.py` for simplicity

## 🚀 Quick Setup

### 1. System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.10+ (tested on Python 3.10.12)
- **RAM**: 16GB minimum (32GB recommended for 5 parallel streams)
- **Storage**: 10GB free space (for videos, models, and outputs)
- **GPU**: CUDA-capable GPU (optional, CPU mode supported)

### 2. System Dependencies

#### Ubuntu/Debian (Recommended)
```bash
# Update system packages
sudo apt-get update

# Install Python and development tools
sudo apt-get install -y python3 python3-pip python3-dev python3-venv

# Install OpenCV system dependencies
sudo apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1
sudo apt-get install -y libglib2.0-0 libxrender1 libgomp1

# Install additional libraries for video processing
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

#### macOS
```bash
# Install Python (if not already installed)
brew install python@3.10

# Install system dependencies
brew install pkg-config
```

#### Windows
- Install Python 3.10+ from [python.org](https://python.org)
- Install Microsoft Visual C++ Build Tools
- Ensure Python is added to PATH

### 3. Python Environment Setup

#### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

#### Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Project Structure Setup

```bash
# Clone or download the project
# Ensure you have the following structure:

Theft_Detection_Bacancy/
├── main.py                 # Complete pipeline (single file solution)
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── videos/                # Input videos directory
│   ├── sample_video_1.mp4
│   ├── sample_video_2.mp4
│   ├── sample_video_3.mp4
│   ├── sample_video_4.mp4
│   └── sample_video_5.mp4
├── output/                # Output directory (auto-created)
│   ├── compressed/        # Compressed videos
│   ├── detections/        # Detection videos
│   └── frames/           # Sample frames
├── logs/                  # Logs directory (auto-created)
└── mlruns/               # MLflow tracking (auto-created)
```

### 5. Configuration

#### Edit config.yaml
```yaml
# Update video paths in config.yaml
input_videos:
- path: videos/your_video_1.mp4
  stream_id: stream_1
- path: videos/your_video_2.mp4
  stream_id: stream_2
# ... add up to 5 videos

# Configure model device
model:
  device: cpu  # or 'cuda' for GPU
  name: yolo11s.pt
  confidence_threshold: 0.5

# Configure output paths
output:
  save_detection_videos: true
  save_compressed_videos: true
  detection_videos_path: output/detections
  compressed_videos_path: output/compressed
```

## 🧪 Testing Installation

### 1. Basic Test
```bash
# Test pipeline help
python3 main.py --help
```

### 2. Configuration Validation
```bash
# Validate configuration
python3 main.py --validate
```

### 3. Run Pipeline
```bash
# Run with CPU (default)
python3 main.py --parallel-mode threading --gpu cpu

# Run with GPU (if available)
python3 main.py --parallel-mode threading --gpu cuda
```

## 🔧 Advanced Configuration

### GPU Setup (Optional)

#### CUDA Installation
```bash
# Check if CUDA is available
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If CUDA not available, install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance Optimization

#### For High-Performance Systems
```yaml
# In config.yaml
performance:
  max_workers: 5          # Process all 5 videos in parallel
  frame_buffer_size: 200  # Larger buffer for smooth processing
  batch_size: 1          # Process frame by frame

model:
  device: cuda           # Use GPU if available
```

#### For Limited Resources
```yaml
# In config.yaml
performance:
  max_workers: 2         # Process 2 videos at a time
  frame_buffer_size: 50  # Smaller buffer
  
model:
  device: cpu           # Use CPU
```

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenCV Import Error
```bash
# Error: ImportError: libGL.so.1: cannot open shared object file
sudo apt-get install -y libgl1-mesa-glx

# Error: cv2 module not found
pip uninstall opencv-python
pip install opencv-python==4.6.0.66
```

#### 2. NumPy Compatibility Issues
```bash
# Error: numpy.dtype size changed
pip uninstall numpy
pip install numpy==1.23.5
```

#### 3. YOLO Model Download Issues
```bash
# Manual model download
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
```

#### 4. MLflow Issues
```bash
# Reset MLflow tracking
rm -rf mlruns/
python3 main.py --no-mlflow  # Run without MLflow
```

#### 5. Memory Issues
```bash
# Reduce parallel processing
# Edit config.yaml:
performance:
  max_workers: 1  # Process videos sequentially
```

### Performance Issues

#### Slow Processing
1. **Use GPU**: Set `device: cuda` in config.yaml
2. **Reduce Workers**: Lower `max_workers` if system is overloaded
3. **Check System Resources**: Monitor CPU/RAM usage

#### Frame Dropping
- The current implementation ensures 100% frame preservation
- If issues occur, check disk space and memory availability

## 📊 Verification

### Check Installation Success
```bash
# Run verification script
python3 -c "
import cv2
import torch
from ultralytics import YOLO
import mlflow
import yaml
import numpy as np

print('✅ OpenCV:', cv2.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())
print('✅ NumPy:', np.__version__)
print('✅ MLflow:', mlflow.__version__)
print('✅ YOLO: Available')
print('✅ All dependencies installed successfully!')
"
```

### Expected Output Structure
After successful run, you should see:
```
output/
├── compressed/
│   ├── stream_1_compressed.mp4
│   ├── stream_2_compressed.mp4
│   └── ...
├── detections/
│   ├── stream_1_detections.mp4
│   ├── stream_2_detections.mp4
│   └── ...
└── frames/
    └── sample_frames/

logs/
└── pipeline.log

mlruns/
└── experiment_tracking_data/
```

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate setup
python3 main.py --validate

# 4. Run pipeline
python3 main.py --parallel-mode threading --gpu cpu

# 5. Check results
ls -la output/compressed/ output/detections/
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure your system meets the minimum requirements
4. Check the logs in `logs/pipeline.log` for detailed error messages

## 🎯 Next Steps

After successful installation:
1. **Customize Configuration**: Edit `config.yaml` for your specific needs
2. **Add Your Videos**: Place your video files in the `videos/` directory
3. **Monitor Performance**: Use the detailed performance summary for optimization
4. **Experiment Tracking**: Explore MLflow UI with `mlflow ui` command
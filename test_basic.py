#!/usr/bin/env python3
"""
Basic test script for the E2E Object Detection Pipeline
Tests core functionality without heavy dependencies
"""

import sys
import os
import yaml
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Check required sections
        required_sections = ['input_videos', 'model', 'output', 'compression']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - {len(config['input_videos'])} video streams configured")
        print(f"   - Model: {config['model']['name']}")
        print(f"   - Device: {config['model']['device']}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_video_files():
    """Test video file accessibility"""
    print("\nTesting video files...")
    
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        video_configs = config.get('input_videos', [])
        valid_videos = 0
        
        for video_config in video_configs:
            video_path = video_config['path']
            stream_id = video_config['stream_id']
            
            if Path(video_path).exists():
                # Try to open with OpenCV
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"   ✅ {stream_id}: {width}x{height}, {fps:.1f}fps, {frame_count} frames")
                    valid_videos += 1
                    cap.release()
                else:
                    print(f"   ❌ {stream_id}: Cannot open video file")
            else:
                print(f"   ❌ {stream_id}: File not found - {video_path}")
        
        print(f"✅ {valid_videos}/{len(video_configs)} videos are valid")
        return valid_videos == len(video_configs)
        
    except Exception as e:
        print(f"❌ Video file testing failed: {e}")
        return False

def test_compression_module():
    """Test compression module import and basic functionality"""
    print("\nTesting compression module...")
    
    try:
        from compression_module import VideoCompressor, StreamingVideoCompressor
        
        # Test basic initialization
        config = {
            'compression': {
                'codec': 'libx265',
                'crf': 28,
                'preset': 'ultrafast'
            },
            'output': {
                'compressed_videos_path': 'output/compressed'
            }
        }
        
        compressor = VideoCompressor(config)
        print("   ✅ VideoCompressor initialized")
        
        streaming_compressor = StreamingVideoCompressor(config, "test_stream")
        print("   ✅ StreamingVideoCompressor initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Compression module test failed: {e}")
        return False

def test_output_directories():
    """Test output directory creation"""
    print("\nTesting output directories...")
    
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        output_config = config.get('output', {})
        directories = [
            output_config.get('base_path', 'output'),
            output_config.get('compressed_videos_path', 'output/compressed'),
            output_config.get('detection_videos_path', 'output/detections'),
            output_config.get('frames_path', 'output/frames')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            if Path(directory).exists():
                print(f"   ✅ {directory}")
            else:
                print(f"   ❌ Failed to create {directory}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Output directory test failed: {e}")
        return False

def test_basic_video_processing():
    """Test basic video frame processing"""
    print("\nTesting basic video processing...")
    
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Get first video
        video_config = config['input_videos'][0]
        video_path = video_config['path']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   ❌ Cannot open video: {video_path}")
            return False
        
        # Read a few frames
        frames_read = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                frames_read += 1
                
                # Basic frame processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Simulate compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, compressed = cv2.imencode('.jpg', frame, encode_param)
                decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            else:
                break
        
        cap.release()
        
        if frames_read > 0:
            print(f"   ✅ Processed {frames_read} frames successfully")
            return True
        else:
            print(f"   ❌ No frames could be processed")
            return False
        
    except Exception as e:
        print(f"   ❌ Basic video processing test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🧪 Running Basic Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Video Files", test_video_files),
        ("Compression Module", test_compression_module),
        ("Output Directories", test_output_directories),
        ("Basic Video Processing", test_basic_video_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The pipeline is ready for testing.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

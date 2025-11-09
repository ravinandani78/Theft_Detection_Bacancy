"""
Utility functions for the E2E Object Detection Pipeline
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def create_sample_videos(output_dir: str = "videos", num_videos: int = 5, 
                        duration: int = 30, fps: int = 30, resolution: Tuple[int, int] = (640, 480)):
    """
    Create sample videos for testing the pipeline.
    
    Args:
        output_dir: Directory to save sample videos
        num_videos: Number of sample videos to create
        duration: Duration of each video in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_videos} sample videos in {output_dir}")
    
    for i in range(num_videos):
        video_path = output_path / f"sample_video_{i+1}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, resolution)
        
        total_frames = duration * fps
        
        for frame_idx in range(total_frames):
            # Create a synthetic frame with moving objects
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Add background gradient
            for y in range(resolution[1]):
                frame[y, :] = [50 + y // 4, 100, 150 - y // 6]
            
            # Add moving rectangles (simulate objects)
            time_factor = frame_idx / total_frames
            
            # Moving rectangle 1
            x1 = int((time_factor * resolution[0]) % resolution[0])
            y1 = int(resolution[1] * 0.3)
            cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 30), (0, 255, 0), -1)
            
            # Moving rectangle 2
            x2 = int(((1 - time_factor) * resolution[0]) % resolution[0])
            y2 = int(resolution[1] * 0.7)
            cv2.rectangle(frame, (x2, y2), (x2 + 40, y2 + 25), (255, 0, 0), -1)
            
            # Add some noise
            noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            # Add frame number text
            cv2.putText(frame, f"Video {i+1} Frame {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()
        logger.info(f"Created sample video: {video_path}")
    
    logger.info("Sample video creation completed")


def update_config_for_samples(config_path: str = "config.yaml", video_dir: str = "videos"):
    """
    Update configuration file to use sample videos.
    
    Args:
        config_path: Path to configuration file
        video_dir: Directory containing sample videos
    """
    # Load existing config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Find sample videos
    video_path = Path(video_dir)
    sample_videos = list(video_path.glob("sample_video_*.mp4"))
    
    if not sample_videos:
        logger.warning(f"No sample videos found in {video_dir}")
        return
    
    # Update input videos configuration
    input_videos = []
    for i, video_file in enumerate(sorted(sample_videos)[:5]):  # Max 5 videos
        input_videos.append({
            'path': str(video_file),
            'stream_id': f"sample_stream_{i+1}"
        })
    
    config['input_videos'] = input_videos
    
    # Save updated config
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    logger.info(f"Updated config with {len(input_videos)} sample videos")


def check_system_requirements():
    """
    Check system requirements and dependencies.
    
    Returns:
        Dictionary with system information and requirements status
    """
    import platform
    import psutil
    
    system_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'requirements_met': True,
        'warnings': [],
        'errors': []
    }
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')))
    if python_version < (3, 8):
        system_info['errors'].append("Python 3.8+ required")
        system_info['requirements_met'] = False
    
    # Check memory
    if system_info['memory_gb'] < 8:
        system_info['warnings'].append("Less than 8GB RAM detected, performance may be limited")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            system_info['gpu'] = {
                'available': True,
                'count': gpu_count,
                'name': gpu_name,
                'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            }
        else:
            system_info['gpu'] = {'available': False}
            system_info['warnings'].append("GPU not available, using CPU mode")
    except ImportError:
        system_info['errors'].append("PyTorch not installed")
        system_info['requirements_met'] = False
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            system_info['ffmpeg'] = True
        else:
            system_info['ffmpeg'] = False
            system_info['errors'].append("FFmpeg not found")
            system_info['requirements_met'] = False
    except FileNotFoundError:
        system_info['ffmpeg'] = False
        system_info['errors'].append("FFmpeg not installed")
        system_info['requirements_met'] = False
    
    return system_info


def benchmark_system(duration: int = 10):
    """
    Run a simple benchmark to test system performance.
    
    Args:
        duration: Benchmark duration in seconds
        
    Returns:
        Benchmark results dictionary
    """
    import torch
    import time
    
    logger.info(f"Running system benchmark for {duration} seconds...")
    
    results = {
        'duration': duration,
        'cpu_performance': 0,
        'gpu_performance': 0,
        'memory_usage': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # CPU benchmark
    start_time = time.time()
    cpu_ops = 0
    
    while time.time() - start_time < duration / 2:
        # Simple CPU operations
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        c = np.dot(a, b)
        cpu_ops += 1
    
    results['cpu_performance'] = cpu_ops
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        start_time = time.time()
        gpu_ops = 0
        
        while time.time() - start_time < duration / 2:
            # Simple GPU operations
            a = torch.rand(1000, 1000, device=device)
            b = torch.rand(1000, 1000, device=device)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            gpu_ops += 1
        
        results['gpu_performance'] = gpu_ops
    
    # Memory usage
    import psutil
    memory = psutil.virtual_memory()
    results['memory_usage'] = {
        'total_gb': round(memory.total / (1024**3), 2),
        'available_gb': round(memory.available / (1024**3), 2),
        'percent_used': memory.percent
    }
    
    logger.info("Benchmark completed")
    return results


def validate_video_files(video_paths: List[str]) -> Dict[str, Any]:
    """
    Validate video files for the pipeline.
    
    Args:
        video_paths: List of video file paths
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid_files': [],
        'invalid_files': [],
        'total_duration': 0,
        'total_frames': 0,
        'resolutions': {},
        'fps_values': {}
    }
    
    for video_path in video_paths:
        try:
            if not Path(video_path).exists():
                results['invalid_files'].append({
                    'path': video_path,
                    'error': 'File not found'
                })
                continue
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                results['invalid_files'].append({
                    'path': video_path,
                    'error': 'Cannot open video file'
                })
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            video_info = {
                'path': video_path,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size_mb': round(Path(video_path).stat().st_size / (1024*1024), 2)
            }
            
            results['valid_files'].append(video_info)
            results['total_duration'] += duration
            results['total_frames'] += frame_count
            
            # Track resolutions and FPS
            resolution = f"{width}x{height}"
            results['resolutions'][resolution] = results['resolutions'].get(resolution, 0) + 1
            results['fps_values'][str(int(fps))] = results['fps_values'].get(str(int(fps)), 0) + 1
            
            cap.release()
            
        except Exception as e:
            results['invalid_files'].append({
                'path': video_path,
                'error': str(e)
            })
    
    return results


def generate_performance_report(mlflow_run_id: str, output_path: str = "performance_report.json"):
    """
    Generate a performance report from MLflow run data.
    
    Args:
        mlflow_run_id: MLflow run ID
        output_path: Output file path for the report
    """
    try:
        import mlflow
        
        # Get run data
        run = mlflow.get_run(mlflow_run_id)
        
        report = {
            'run_id': mlflow_run_id,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'status': run.info.status,
            'parameters': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'artifacts': run.info.artifact_uri,
            'generated_at': datetime.now().isoformat()
        }
        
        # Calculate derived metrics
        if 'total_runtime' in report['metrics']:
            runtime = report['metrics']['total_runtime']
            total_frames = report['metrics'].get('final_total_frames_processed', 0)
            
            if runtime > 0:
                report['derived_metrics'] = {
                    'average_fps': total_frames / runtime,
                    'processing_efficiency': report['metrics'].get('final_total_fps', 0) / (total_frames / runtime) if total_frames > 0 else 0
                }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report generated: {output_path}")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        return None


def cleanup_outputs(base_path: str = "output", keep_latest: int = 5):
    """
    Clean up old output files, keeping only the latest runs.
    
    Args:
        base_path: Base output directory
        keep_latest: Number of latest runs to keep
    """
    output_path = Path(base_path)
    
    if not output_path.exists():
        return
    
    # Clean up video files
    for subdir in ['compressed', 'detections']:
        subdir_path = output_path / subdir
        if subdir_path.exists():
            video_files = list(subdir_path.glob("*.mp4"))
            video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in video_files[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.info(f"Removed old file: {old_file}")
                except Exception as e:
                    logger.warning(f"Could not remove {old_file}: {e}")
    
    # Clean up frame samples
    frames_path = output_path / "frames"
    if frames_path.exists():
        frame_files = list(frames_path.glob("*.jpg"))
        frame_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only recent frames
        for old_frame in frame_files[keep_latest * 10:]:  # Keep more frames
            try:
                old_frame.unlink()
            except Exception as e:
                logger.warning(f"Could not remove {old_frame}: {e}")
    
    logger.info(f"Cleanup completed, kept {keep_latest} latest runs")


if __name__ == "__main__":
    # Utility script can be run directly for setup tasks
    import argparse
    
    parser = argparse.ArgumentParser(description="E2E Pipeline Utilities")
    parser.add_argument("--create-samples", action="store_true", help="Create sample videos")
    parser.add_argument("--check-system", action="store_true", help="Check system requirements")
    parser.add_argument("--benchmark", type=int, default=10, help="Run benchmark (duration in seconds)")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old output files")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_videos()
        update_config_for_samples()
    
    if args.check_system:
        info = check_system_requirements()
        print(json.dumps(info, indent=2))
    
    if args.benchmark:
        results = benchmark_system(args.benchmark)
        print(json.dumps(results, indent=2))
    
    if args.cleanup:
        cleanup_outputs()

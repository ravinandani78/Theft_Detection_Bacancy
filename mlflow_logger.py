"""
MLflow Integration Module for E2E Object Detection Pipeline

This module provides comprehensive MLflow logging capabilities for tracking
model performance, metrics, and artifacts in the object detection pipeline.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import cv2
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import tempfile
import threading
from queue import Queue, Empty
import pandas as pd

logger = logging.getLogger(__name__)


class MLflowLogger:
    """
    Comprehensive MLflow logger for object detection pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLflow logger.
        
        Args:
            config: Configuration dictionary containing MLflow settings
        """
        self.config = config
        self.mlflow_config = config.get('mlflow', {})
        
        # MLflow settings
        self.enabled = self.mlflow_config.get('enabled', True)
        self.experiment_name = self.mlflow_config.get('experiment_name', 'object_detection_pipeline')
        self.tracking_uri = self.mlflow_config.get('tracking_uri', 'mlruns')
        self.log_model = self.mlflow_config.get('log_model', True)
        self.log_metrics_interval = self.mlflow_config.get('log_metrics_interval', 10)
        self.log_sample_frames = self.mlflow_config.get('log_sample_frames', True)
        self.sample_frame_interval = self.mlflow_config.get('sample_frame_interval', 100)
        
        # Initialize MLflow
        self.run_id = None
        self.experiment_id = None
        self.metrics_buffer = {}
        self.frame_counter = 0
        
        # Async logging
        self.logging_queue = Queue()
        self.logging_thread = None
        self.is_logging = False
        
        if self.enabled:
            self._setup_mlflow()
            self._start_async_logging()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment and tracking."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(self.experiment_name)
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Could not create/get experiment: {e}")
                self.experiment_id = mlflow.create_experiment(
                    f"{self.experiment_name}_{int(time.time())}"
                )
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow setup complete. Experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            self.enabled = False
    
    def _start_async_logging(self):
        """Start asynchronous logging thread."""
        if not self.enabled:
            return
        
        self.is_logging = True
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
        logger.info("Started async MLflow logging thread")
    
    def _logging_worker(self):
        """Worker thread for asynchronous logging."""
        while self.is_logging:
            try:
                # Get logging task from queue
                task = self.logging_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                task_type = task.get('type')
                
                if task_type == 'metric':
                    self._log_metric_sync(task['key'], task['value'], task.get('step'))
                elif task_type == 'param':
                    self._log_param_sync(task['key'], task['value'])
                elif task_type == 'artifact':
                    self._log_artifact_sync(task['path'], task.get('artifact_path'))
                elif task_type == 'image':
                    self._log_image_sync(task['image'], task['name'], task.get('step'))
                elif task_type == 'dict':
                    self._log_dict_sync(task['dictionary'], task['artifact_file'])
                
                self.logging_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in MLflow logging worker: {e}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
        """
        if not self.enabled:
            return
        
        try:
            # Generate run name if not provided
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"detection_pipeline_{timestamp}"
            
            # Start MLflow run
            mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = mlflow.active_run().info.run_id
            
            # Log initial parameters
            self._log_config_params()
            
            logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self.enabled = False
    
    def _log_config_params(self):
        """Log configuration parameters."""
        try:
            # Model parameters
            model_config = self.config.get('model', {})
            for key, value in model_config.items():
                self.log_param(f"model_{key}", value)
            
            # Performance parameters
            perf_config = self.config.get('performance', {})
            for key, value in perf_config.items():
                self.log_param(f"performance_{key}", value)
            
            # Compression parameters
            comp_config = self.config.get('compression', {})
            for key, value in comp_config.items():
                self.log_param(f"compression_{key}", value)
            
            # General parameters
            self.log_param("num_video_streams", len(self.config.get('input_videos', [])))
            self.log_param("pipeline_version", "1.0.0")
            
        except Exception as e:
            logger.error(f"Failed to log config parameters: {e}")
    
    def log_param(self, key: str, value: Any):
        """
        Log a parameter asynchronously.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.enabled:
            return
        
        try:
            task = {
                'type': 'param',
                'key': key,
                'value': str(value)
            }
            self.logging_queue.put_nowait(task)
        except Exception as e:
            logger.error(f"Failed to queue parameter logging: {e}")
    
    def _log_param_sync(self, key: str, value: str):
        """Synchronously log parameter."""
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric asynchronously.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.enabled:
            return
        
        try:
            task = {
                'type': 'metric',
                'key': key,
                'value': float(value),
                'step': step
            }
            self.logging_queue.put_nowait(task)
        except Exception as e:
            logger.error(f"Failed to queue metric logging: {e}")
    
    def _log_metric_sync(self, key: str, value: float, step: Optional[int] = None):
        """Synchronously log metric."""
        try:
            mlflow.log_metric(key, value, step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")
    
    def log_detection_metrics(self, stream_stats: Dict[str, Any], detector_stats: Dict[str, Any]):
        """
        Log detection performance metrics.
        
        Args:
            stream_stats: Statistics from stream processing
            detector_stats: Statistics from object detector
        """
        if not self.enabled:
            return
        
        try:
            # Log detector metrics
            self.log_metric("detector_avg_inference_time", detector_stats.get('avg_inference_time', 0))
            self.log_metric("detector_current_fps", detector_stats.get('current_fps', 0))
            self.log_metric("detector_total_inferences", detector_stats.get('total_inferences', 0))
            
            # Log per-stream metrics
            total_fps = 0
            total_detections = 0
            active_streams = 0
            
            for stream_id, stats in stream_stats.items():
                stream_fps = stats.get('average_fps', 0)
                stream_detections = stats.get('detections_count', 0)
                
                self.log_metric(f"stream_{stream_id}_fps", stream_fps)
                self.log_metric(f"stream_{stream_id}_detections", stream_detections)
                self.log_metric(f"stream_{stream_id}_frames_processed", stats.get('frames_processed', 0))
                
                if stats.get('is_running', False):
                    total_fps += stream_fps
                    total_detections += stream_detections
                    active_streams += 1
            
            # Log aggregate metrics
            self.log_metric("total_fps", total_fps)
            self.log_metric("total_detections", total_detections)
            self.log_metric("active_streams", active_streams)
            self.log_metric("avg_fps_per_stream", total_fps / max(active_streams, 1))
            
        except Exception as e:
            logger.error(f"Failed to log detection metrics: {e}")
    
    def log_frame_sample(self, frame: np.ndarray, detections: List[Dict], 
                        stream_id: str, frame_number: int):
        """
        Log a sample frame with detections.
        
        Args:
            frame: Frame image as numpy array
            detections: List of detection results
            stream_id: Stream identifier
            frame_number: Frame number
        """
        if not self.enabled or not self.log_sample_frames:
            return
        
        # Check if we should log this frame
        if frame_number % self.sample_frame_interval != 0:
            return
        
        try:
            # Create annotated frame
            annotated_frame = self._create_annotated_frame(frame, detections)
            
            # Generate image name
            timestamp = int(time.time())
            image_name = f"detection_sample_{stream_id}_{frame_number}_{timestamp}"
            
            task = {
                'type': 'image',
                'image': annotated_frame,
                'name': image_name,
                'step': frame_number
            }
            self.logging_queue.put_nowait(task)
            
            # Also log detection metadata
            detection_data = {
                'stream_id': stream_id,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'detection_count': len(detections),
                'detections': detections
            }
            
            self.log_dict(detection_data, f"detection_metadata_{image_name}.json")
            
        except Exception as e:
            logger.error(f"Failed to log frame sample: {e}")
    
    def _create_annotated_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Create annotated frame with detection visualizations.
        
        Args:
            frame: Original frame
            detections: Detection results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated
    
    def _log_image_sync(self, image: np.ndarray, name: str, step: Optional[int] = None):
        """Synchronously log image."""
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                mlflow.log_artifact(tmp_file.name, f"sample_frames/{name}.jpg")
                os.unlink(tmp_file.name)  # Clean up temp file
                
        except Exception as e:
            logger.error(f"Failed to log image {name}: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact file name
        """
        if not self.enabled:
            return
        
        try:
            task = {
                'type': 'dict',
                'dictionary': dictionary,
                'artifact_file': artifact_file
            }
            self.logging_queue.put_nowait(task)
        except Exception as e:
            logger.error(f"Failed to queue dictionary logging: {e}")
    
    def _log_dict_sync(self, dictionary: Dict[str, Any], artifact_file: str):
        """Synchronously log dictionary."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(dictionary, tmp_file, indent=2, default=str)
                tmp_file.flush()
                mlflow.log_artifact(tmp_file.name, artifact_file)
                os.unlink(tmp_file.name)  # Clean up temp file
                
        except Exception as e:
            logger.error(f"Failed to log dictionary {artifact_file}: {e}")
    
    def log_model_info(self, model, model_name: str = "yolo_detector"):
        """
        Log model information and artifacts.
        
        Args:
            model: Model object to log
            model_name: Name for the logged model
        """
        if not self.enabled or not self.log_model:
            return
        
        try:
            # Log model as artifact
            mlflow.pytorch.log_model(
                model,
                model_name,
                registered_model_name=f"{self.experiment_name}_{model_name}"
            )
            
            # Log model metadata
            model_info = {
                'model_type': 'YOLOv11',
                'model_name': model_name,
                'framework': 'ultralytics',
                'logged_at': datetime.now().isoformat()
            }
            
            self.log_dict(model_info, f"{model_name}_info.json")
            
            logger.info(f"Logged model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_system_metrics(self):
        """Log system performance metrics."""
        if not self.enabled:
            return
        
        try:
            import psutil
            import GPUtil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.log_metric("system_cpu_percent", cpu_percent)
            self.log_metric("system_memory_percent", memory.percent)
            self.log_metric("system_memory_available_gb", memory.available / (1024**3))
            
            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.log_metric("gpu_utilization", gpu.load * 100)
                    self.log_metric("gpu_memory_percent", gpu.memoryUtil * 100)
                    self.log_metric("gpu_temperature", gpu.temperature)
            except:
                pass  # GPU monitoring not available
                
        except ImportError:
            logger.warning("psutil or GPUtil not available for system metrics")
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")
    
    def create_performance_summary(self, final_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive performance summary.
        
        Args:
            final_stats: Final statistics from the pipeline
            
        Returns:
            Performance summary dictionary
        """
        summary = {
            'pipeline_summary': {
                'total_runtime': final_stats.get('total_runtime', 0),
                'total_frames_processed': sum(
                    stats.get('frames_processed', 0) 
                    for stats in final_stats.get('stream_stats', {}).values()
                ),
                'total_detections': sum(
                    stats.get('detections_count', 0) 
                    for stats in final_stats.get('stream_stats', {}).values()
                ),
                'average_fps': final_stats.get('detector_stats', {}).get('current_fps', 0),
                'streams_processed': len(final_stats.get('stream_stats', {}))
            },
            'detector_performance': final_stats.get('detector_stats', {}),
            'stream_performance': final_stats.get('stream_stats', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def finalize_run(self, final_stats: Optional[Dict[str, Any]] = None):
        """
        Finalize the MLflow run with summary metrics.
        
        Args:
            final_stats: Optional final statistics to log
        """
        if not self.enabled:
            return
        
        try:
            if final_stats:
                # Log final performance summary
                summary = self.create_performance_summary(final_stats)
                self.log_dict(summary, "performance_summary.json")
                
                # Log final metrics
                pipeline_summary = summary['pipeline_summary']
                for key, value in pipeline_summary.items():
                    if isinstance(value, (int, float)):
                        self.log_metric(f"final_{key}", value)
            
            # Wait for async logging to complete
            self._stop_async_logging()
            
            # End MLflow run
            mlflow.end_run()
            logger.info(f"Finalized MLflow run: {self.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to finalize MLflow run: {e}")
    
    def _stop_async_logging(self):
        """Stop asynchronous logging thread."""
        if self.logging_thread and self.logging_thread.is_alive():
            self.is_logging = False
            
            # Send shutdown signal
            try:
                self.logging_queue.put_nowait(None)
            except:
                pass
            
            # Wait for thread to finish
            self.logging_thread.join(timeout=10.0)
            logger.info("Stopped async MLflow logging thread")


class MetricsCollector:
    """
    Collects and aggregates metrics for MLflow logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_history = {}
        self.collection_interval = config.get('mlflow', {}).get('log_metrics_interval', 10)
        self.last_collection_time = time.time()
    
    def collect_metrics(self, stream_stats: Dict[str, Any], detector_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and aggregate metrics from all sources.
        
        Args:
            stream_stats: Stream processing statistics
            detector_stats: Detector performance statistics
            
        Returns:
            Aggregated metrics dictionary
        """
        current_time = time.time()
        
        # Collect current metrics
        metrics = {
            'timestamp': current_time,
            'detector': detector_stats,
            'streams': stream_stats
        }
        
        # Store in history
        self.metrics_history[current_time] = metrics
        
        # Keep only recent history (last 1000 entries)
        if len(self.metrics_history) > 1000:
            oldest_keys = sorted(self.metrics_history.keys())[:-1000]
            for key in oldest_keys:
                del self.metrics_history[key]
        
        return metrics
    
    def get_aggregated_metrics(self, time_window: float = 60.0) -> Dict[str, Any]:
        """
        Get aggregated metrics over a time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Aggregated metrics
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter metrics within time window
        recent_metrics = {
            timestamp: metrics 
            for timestamp, metrics in self.metrics_history.items() 
            if timestamp >= cutoff_time
        }
        
        if not recent_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {
            'time_window': time_window,
            'sample_count': len(recent_metrics),
            'detector_avg_fps': np.mean([
                m['detector'].get('current_fps', 0) 
                for m in recent_metrics.values()
            ]),
            'total_avg_fps': np.mean([
                sum(s.get('average_fps', 0) for s in m['streams'].values())
                for m in recent_metrics.values()
            ])
        }
        
        return aggregated

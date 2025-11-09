"""
Object Detection Module for E2E Pipeline using YOLOv11

This module provides real-time object detection capabilities using YOLOv11 model
with support for multiple concurrent video streams and GPU acceleration.
"""

import cv2
import numpy as np
import torch
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from queue import Queue, Empty
from ultralytics import YOLO
import asyncio

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    YOLOv11-based object detector with real-time inference capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the object detector.
        
        Args:
            config: Configuration dictionary containing model and detection settings
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.output_config = config.get('output', {})
        self.performance_config = config.get('performance', {})
        
        # Model parameters
        self.model_name = self.model_config.get('name', 'yolo11s.pt')
        self.device = self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = self.model_config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.model_config.get('iou_threshold', 0.45)
        self.max_detections = self.model_config.get('max_detections', 100)
        
        # Initialize model
        self.model = None
        self.class_names = None
        self.colors = None
        
        # Performance tracking
        self.inference_times = []
        self.fps_history = []
        
        self._load_model()
        self._setup_visualization()
        
    def _load_model(self):
        """Load YOLOv11 model."""
        try:
            logger.info(f"Loading YOLOv11 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Move model to specified device
            if self.device != 'cpu':
                self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _setup_visualization(self):
        """Setup colors for bounding box visualization."""
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (detections list, inference time)
        """
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times for FPS calculation
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First (and only) result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        detection = {
                            'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                            'class_name': self.class_names[class_ids[i]],
                            'color': self.colors[class_ids[i]].tolist()
                        }
                        detections.append(detection)
            
            return detections, inference_time
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return [], time.time() - start_time
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], 
                           stream_id: str = "", show_fps: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            stream_id: Stream identifier for display
            show_fps: Whether to show FPS overlay
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = tuple(map(int, detection['color']))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add FPS and stream info overlay
        if show_fps:
            current_fps = self.get_current_fps()
            info_text = f"Stream: {stream_id} | FPS: {current_fps:.1f} | Detections: {len(detections)}"
            
            # Draw info background
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (10, 10), (20 + text_size[0], 40), (0, 0, 0), -1)
            
            # Draw info text
            cv2.putText(vis_frame, info_text, (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_frame
    
    def get_current_fps(self) -> float:
        """
        Calculate current FPS based on recent inference times.
        
        Returns:
            Current FPS
        """
        if len(self.inference_times) < 2:
            return 0.0
        
        avg_inference_time = np.mean(self.inference_times[-10:])  # Use last 10 frames
        return 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get detection performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'current_fps': 0.0,
                'total_inferences': 0
            }
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'current_fps': self.get_current_fps(),
            'total_inferences': len(self.inference_times)
        }


class StreamDetectionWorker:
    """
    Worker class for processing detection on a single video stream.
    """
    
    def __init__(self, config: Dict[str, Any], stream_id: str, detector: ObjectDetector):
        """
        Initialize stream detection worker.
        
        Args:
            config: Configuration dictionary
            stream_id: Unique stream identifier
            detector: Shared ObjectDetector instance
        """
        self.config = config
        self.stream_id = stream_id
        self.detector = detector
        self.output_config = config.get('output', {})
        
        # Processing queues
        self.input_queue = Queue(maxsize=config.get('performance', {}).get('frame_buffer_size', 30))
        self.output_queue = Queue(maxsize=30)
        
        # Worker thread
        self.worker_thread = None
        self.is_running = False
        
        # Statistics
        self.frames_processed = 0
        self.detections_count = 0
        self.start_time = None
        
        # Output video writer
        self.video_writer = None
        self.original_fps = None
        self.setup_output_writer()
    
    def setup_output_writer(self):
        """Setup video writer for saving detection results."""
        if self.output_config.get('save_detection_videos', True):
            output_path = Path(self.output_config.get('detection_videos_path', 'output/detections'))
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Video writer will be initialized when first frame is received
            self.output_file = output_path / f"{self.stream_id}_detections.mp4"
    
    def start_processing(self):
        """Start the detection processing worker."""
        if self.is_running:
            logger.warning(f"Detection worker already running for {self.stream_id}")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.worker_thread = threading.Thread(
            target=self._detection_worker,
            daemon=True
        )
        self.worker_thread.start()
        
        logger.info(f"Started detection worker for {self.stream_id}")
    
    def _detection_worker(self):
        """Main detection worker loop."""
        try:
            while self.is_running:
                try:
                    # Get frame from input queue
                    frame_data = self.input_queue.get(timeout=1.0)
                    
                    if frame_data is None:  # Shutdown signal
                        break
                    
                    frame = frame_data['frame']
                    timestamp = frame_data['timestamp']
                    frame_number = frame_data['frame_number']
                    
                    # Perform detection
                    detections, inference_time = self.detector.detect_objects(frame)
                    
                    # Visualize detections
                    vis_frame = self.detector.visualize_detections(
                        frame, detections, self.stream_id, 
                        show_fps=self.output_config.get('fps_overlay', True)
                    )
                    
                    # Initialize video writer if needed
                    if self.video_writer is None and self.output_config.get('save_detection_videos', True):
                        height, width = vis_frame.shape[:2]
                        
                        # Get original FPS from frame data or use default
                        if 'original_fps' in frame_data:
                            self.original_fps = frame_data['original_fps']
                        elif self.original_fps is None:
                            self.original_fps = 30.0  # fallback
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(
                            str(self.output_file), fourcc, self.original_fps, (width, height)
                        )
                        
                        logger.info(f"Video writer initialized for {self.stream_id} with FPS: {self.original_fps}")
                        
                        # Verify video writer opened successfully
                        if not self.video_writer.isOpened():
                            logger.error(f"Failed to open video writer for {self.stream_id}")
                            self.video_writer = None
                    
                    # Save frame to video
                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.write(vis_frame)
                        
                        # Flush writer periodically for better reliability
                        if self.frames_processed % 30 == 0:
                            # Force flush (some codecs need this)
                            pass
                    
                    # Prepare output data
                    output_data = {
                        'frame': vis_frame,
                        'original_frame': frame,
                        'detections': detections,
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'stream_id': self.stream_id,
                        'inference_time': inference_time,
                        'detection_count': len(detections)
                    }
                    
                    # Add to output queue
                    if not self.output_queue.full():
                        self.output_queue.put(output_data)
                    
                    # Update statistics
                    self.frames_processed += 1
                    self.detections_count += len(detections)
                    
                    # Mark task as done
                    self.input_queue.task_done()
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in detection worker for {self.stream_id}: {e}")
        
        except Exception as e:
            logger.error(f"Fatal error in detection worker for {self.stream_id}: {e}")
        
        finally:
            # Cleanup
            if self.video_writer:
                self.video_writer.release()
            
            logger.info(f"Detection worker stopped for {self.stream_id}. Processed {self.frames_processed} frames")
    
    def add_frame(self, frame_data: Dict) -> bool:
        """
        Add a frame to the processing queue with optimized parallel handling.
        
        Args:
            frame_data: Dictionary containing frame and metadata
            
        Returns:
            True if frame was added successfully, False if queue is full
        """
        try:
            # Try to add frame without blocking
            self.input_queue.put_nowait(frame_data)
            return True
        except:
            # For parallel processing, be more aggressive about queue management
            # Remove up to 3 oldest frames if queue is full
            frames_removed = 0
            max_remove = 3
            
            while frames_removed < max_remove:
                try:
                    self.input_queue.get_nowait()  # Remove oldest frame
                    frames_removed += 1
                except:
                    break
            
            # Now try to add the new frame
            try:
                self.input_queue.put_nowait(frame_data)
                if frames_removed > 0:
                    logger.debug(f"Removed {frames_removed} old frames for {self.stream_id} to make space")
                return True
            except:
                return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get detection result from output queue.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            Detection result dictionary or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stop_processing(self):
        """Stop the detection processing."""
        self.is_running = False
        
        # Send shutdown signal
        try:
            self.input_queue.put_nowait(None)
        except:
            pass
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        logger.info(f"Stopped detection worker for {self.stream_id}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        elapsed_time = time.time() - (self.start_time or time.time())
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'stream_id': self.stream_id,
            'frames_processed': self.frames_processed,
            'detections_count': self.detections_count,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'is_running': self.is_running
        }


class MultiStreamDetectionManager:
    """
    Manages object detection across multiple video streams concurrently.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-stream detection manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detector = ObjectDetector(config)
        self.workers = {}
        self.display_windows = {}
        
        # Create output directories
        output_path = Path(config.get('output', {}).get('base_path', 'output'))
        output_path.mkdir(parents=True, exist_ok=True)
    
    def add_stream(self, stream_id: str) -> StreamDetectionWorker:
        """
        Add a new stream for detection processing.
        
        Args:
            stream_id: Unique stream identifier
            
        Returns:
            StreamDetectionWorker instance
        """
        if stream_id in self.workers:
            logger.warning(f"Stream {stream_id} already exists")
            return self.workers[stream_id]
        
        worker = StreamDetectionWorker(self.config, stream_id, self.detector)
        self.workers[stream_id] = worker
        worker.start_processing()
        
        logger.info(f"Added detection stream: {stream_id}")
        return worker
    
    def remove_stream(self, stream_id: str):
        """
        Remove a stream from detection processing.
        
        Args:
            stream_id: Stream identifier to remove
        """
        if stream_id in self.workers:
            self.workers[stream_id].stop_processing()
            del self.workers[stream_id]
            
            # Close display window if exists
            if stream_id in self.display_windows:
                cv2.destroyWindow(f"Detection - {stream_id}")
                del self.display_windows[stream_id]
            
            logger.info(f"Removed detection stream: {stream_id}")
    
    def process_frame(self, stream_id: str, frame_data: Dict) -> bool:
        """
        Process a frame for the specified stream.
        
        Args:
            stream_id: Stream identifier
            frame_data: Frame data dictionary
            
        Returns:
            True if frame was processed successfully
        """
        if stream_id not in self.workers:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        return self.workers[stream_id].add_frame(frame_data)
    
    def get_results(self, timeout: float = 0.1) -> Dict[str, Any]:
        """
        Get detection results from all streams.
        
        Args:
            timeout: Maximum time to wait for results per stream
            
        Returns:
            Dictionary mapping stream_id to detection results
        """
        results = {}
        for stream_id, worker in self.workers.items():
            result = worker.get_result(timeout=timeout)
            if result:
                results[stream_id] = result
        
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """
        Display detection results in separate windows.
        
        Args:
            results: Dictionary of detection results per stream
        """
        if not self.config.get('output', {}).get('display_realtime', True):
            return
        
        for stream_id, result in results.items():
            window_name = f"Detection - {stream_id}"
            
            # Create window if it doesn't exist
            if stream_id not in self.display_windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)
                self.display_windows[stream_id] = True
            
            # Display frame
            cv2.imshow(window_name, result['frame'])
    
    def stop_all_streams(self):
        """Stop all detection streams."""
        for stream_id in list(self.workers.keys()):
            self.remove_stream(stream_id)
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("Stopped all detection streams")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all streams.
        
        Returns:
            Dictionary containing statistics for all streams
        """
        stats = {
            'detector_stats': self.detector.get_detection_stats(),
            'stream_stats': {}
        }
        
        for stream_id, worker in self.workers.items():
            stats['stream_stats'][stream_id] = worker.get_processing_stats()
        
        return stats

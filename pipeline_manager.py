"""
Pipeline Manager for E2E Object Detection System

This module orchestrates the entire end-to-end pipeline, managing concurrent
video compression and object detection across multiple video streams.
"""

import asyncio
import logging
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import cv2
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from compression_module import StreamingVideoCompressor, create_compressor
from detection_module import MultiStreamDetectionManager
from mlflow_logger import MLflowLogger, MetricsCollector

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Main pipeline manager that orchestrates compression and detection processes.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Core components
        self.compression_workers = {}
        self.detection_manager = None
        self.mlflow_logger = None
        self.metrics_collector = None
        
        # Pipeline state
        self.is_running = False
        self.start_time = None
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.pipeline_stats = {
            'streams_processed': 0,
            'total_frames': 0,
            'total_detections': 0,
            'errors': 0
        }
        
        # Setup components
        self._setup_logging()
        self._setup_output_directories()
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory
        log_file = log_config.get('log_file', 'logs/pipeline.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        
        logger.info("Logging configured successfully")
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        output_config = self.config.get('output', {})
        
        directories = [
            output_config.get('base_path', 'output'),
            output_config.get('compressed_videos_path', 'output/compressed'),
            output_config.get('detection_videos_path', 'output/detections'),
            output_config.get('frames_path', 'output/frames')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created")
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize detection manager
            self.detection_manager = MultiStreamDetectionManager(self.config)
            logger.info("Detection manager initialized")
            
            # Initialize MLflow logger
            self.mlflow_logger = MLflowLogger(self.config)
            logger.info("MLflow logger initialized")
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(self.config)
            logger.info("Metrics collector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def start_pipeline(self):
        """
        Start the complete E2E pipeline.
        """
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting E2E Object Detection Pipeline")
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Start MLflow run
            run_tags = {
                'pipeline_version': '1.0.0',
                'num_streams': str(len(self.config.get('input_videos', []))),
                'model': self.config.get('model', {}).get('name', 'yolo11s.pt')
            }
            self.mlflow_logger.start_run(tags=run_tags)
            
            # Log model information
            if hasattr(self.detection_manager.detector, 'model'):
                self.mlflow_logger.log_model_info(
                    self.detection_manager.detector.model,
                    "yolo_detector"
                )
            
            # Start processing streams
            await self._process_all_streams()
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.pipeline_stats['errors'] += 1
            raise
        
        finally:
            await self._cleanup_pipeline()
    
    async def _process_all_streams(self):
        """
        Process all video streams concurrently.
        """
        input_videos = self.config.get('input_videos', [])
        
        if not input_videos:
            logger.error("No input videos configured")
            return
        
        logger.info(f"Processing {len(input_videos)} video streams in PARALLEL")
        
        # Create tasks for each video stream - ALL RUNNING SIMULTANEOUSLY
        tasks = []
        for i, video_config in enumerate(input_videos):
            task = asyncio.create_task(
                self._process_single_stream(video_config)
            )
            tasks.append(task)
            logger.info(f"Started parallel task {i+1}/{len(input_videos)}: {video_config['stream_id']}")
        
        # Start metrics monitoring
        metrics_task = asyncio.create_task(self._monitor_metrics())
        tasks.append(metrics_task)
        
        logger.info(f"All {len(input_videos)} video streams now processing in parallel!")
        
        # Wait for all tasks to complete or shutdown signal
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for i, result in enumerate(results[:-1]):  # Exclude metrics task
                if isinstance(result, Exception):
                    logger.error(f"Stream {i+1} failed: {result}")
                else:
                    logger.info(f"Stream {i+1} completed successfully")
                    
        except Exception as e:
            logger.error(f"Error in parallel stream processing: {e}")
    
    async def _process_single_stream(self, video_config: Dict[str, Any]):
        """
        Process a single video stream with compression and detection.
        
        Args:
            video_config: Configuration for the video stream
        """
        stream_id = video_config['stream_id']
        video_path = video_config['path']
        
        logger.info(f"Starting processing for stream {stream_id}: {video_path}")
        
        try:
            # Validate input file
            if not Path(video_path).exists():
                logger.error(f"Video file not found: {video_path}")
                return
            
            # Initialize compression worker
            compressor = StreamingVideoCompressor(self.config, stream_id)
            self.compression_workers[stream_id] = compressor
            
            # Add stream to detection manager
            detection_worker = self.detection_manager.add_stream(stream_id)
            
            # Start compression in streaming mode
            compressor.start_streaming_compression(video_path)
            
            # Process frames in real-time
            await self._stream_processing_loop(stream_id, compressor, detection_worker)
            
        except Exception as e:
            logger.error(f"Error processing stream {stream_id}: {e}")
            self.pipeline_stats['errors'] += 1
        
        finally:
            # Cleanup stream
            if stream_id in self.compression_workers:
                self.compression_workers[stream_id].stop_compression()
                del self.compression_workers[stream_id]
            
            self.detection_manager.remove_stream(stream_id)
            logger.info(f"Finished processing stream {stream_id}")
    
    async def _stream_processing_loop(self, stream_id: str, 
                                    compressor: StreamingVideoCompressor,
                                    detection_worker):
        """
        Main processing loop for a single stream.
        
        Args:
            stream_id: Stream identifier
            compressor: Video compressor instance
            detection_worker: Detection worker instance
        """
        frame_count = 0
        last_frame_time = time.time()
        
        # Process all frames from compression
        all_frames_processed = False
        
        while self.is_running and not self.shutdown_event.is_set() and not all_frames_processed:
            try:
                # Get compressed frame
                frame_data = compressor.get_compressed_frame(timeout=2.0)
                
                if frame_data is None:
                    # Check if compression is still running
                    if not compressor.is_running:
                        logger.info(f"Compression finished for stream {stream_id}")
                        # Continue processing remaining frames in detection queue
                        break
                    continue
                
                # Send frame to detection - optimized for parallel processing
                retry_count = 0
                max_retries = 200  # Allow more retries for parallel processing
                frame_added = False
                
                while retry_count < max_retries and self.is_running:
                    if self.detection_manager.process_frame(stream_id, frame_data):
                        frame_added = True
                        break
                    else:
                        # Shorter wait for better parallel performance
                        await asyncio.sleep(0.002)  # 2ms wait
                        retry_count += 1
                
                if not frame_added:
                    logger.warning(f"Frame {frame_count} dropped for {stream_id} after {max_retries} retries")
                    # Continue processing - don't stop the entire stream for one frame
                
                frame_count += 1
                self.pipeline_stats['total_frames'] += 1
                
                # Log sample frames to MLflow
                if frame_count % self.mlflow_logger.sample_frame_interval == 0:
                    # Get detection result for logging
                    result = detection_worker.get_result(timeout=0.1)
                    if result:
                        self.mlflow_logger.log_frame_sample(
                            result['original_frame'],
                            result['detections'],
                            stream_id,
                            frame_count
                        )
                        self.pipeline_stats['total_detections'] += result['detection_count']
                
                # No throttling - process frames as fast as possible to preserve all frames
                
            except Exception as e:
                logger.error(f"Error in processing loop for {stream_id}: {e}")
                await asyncio.sleep(0.1)
        
        # Wait for all remaining frames to be processed by detection
        logger.info(f"Waiting for detection to complete remaining frames for {stream_id}")
        wait_count = 0
        max_wait = 100  # Maximum wait cycles
        
        while wait_count < max_wait:
            # Check if detection queue is empty and worker is idle
            worker = self.detection_manager.workers.get(stream_id)
            if worker and (worker.input_queue.qsize() > 0 or worker.frames_processed < frame_count):
                await asyncio.sleep(0.1)
                wait_count += 1
            else:
                break
        
        logger.info(f"Processed {frame_count} frames for stream {stream_id}")
        self.pipeline_stats['streams_processed'] += 1
    
    async def _monitor_metrics(self):
        """
        Monitor and log performance metrics.
        """
        logger.info("Started metrics monitoring")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get current statistics
                all_stats = self.detection_manager.get_all_stats()
                
                # Collect metrics
                metrics = self.metrics_collector.collect_metrics(
                    all_stats.get('stream_stats', {}),
                    all_stats.get('detector_stats', {})
                )
                
                # Log to MLflow
                self.mlflow_logger.log_detection_metrics(
                    all_stats.get('stream_stats', {}),
                    all_stats.get('detector_stats', {})
                )
                
                # Log system metrics
                self.mlflow_logger.log_system_metrics()
                
                # Display real-time results
                results = self.detection_manager.get_results(timeout=0.1)
                if results:
                    self.detection_manager.display_results(results)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit key pressed, shutting down...")
                    self.shutdown_event.set()
                    break
                
                # Wait before next metrics collection
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in metrics monitoring: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("Stopped metrics monitoring")
    
    async def _cleanup_pipeline(self):
        """
        Cleanup pipeline resources.
        """
        logger.info("Cleaning up pipeline resources...")
        
        try:
            # Stop all compression workers
            for stream_id, compressor in self.compression_workers.items():
                compressor.stop_compression()
            
            # Stop detection manager
            if self.detection_manager:
                self.detection_manager.stop_all_streams()
            
            # Calculate final statistics
            end_time = time.time()
            total_runtime = end_time - (self.start_time or end_time)
            
            final_stats = {
                'total_runtime': total_runtime,
                'pipeline_stats': self.pipeline_stats,
                **self.detection_manager.get_all_stats()
            }
            
            # Finalize MLflow run
            if self.mlflow_logger:
                self.mlflow_logger.finalize_run(final_stats)
            
            # Log final summary
            logger.info("Pipeline Summary:")
            logger.info(f"  Total Runtime: {total_runtime:.2f} seconds")
            logger.info(f"  Streams Processed: {self.pipeline_stats['streams_processed']}")
            logger.info(f"  Total Frames: {self.pipeline_stats['total_frames']}")
            logger.info(f"  Total Detections: {self.pipeline_stats['total_detections']}")
            logger.info(f"  Errors: {self.pipeline_stats['errors']}")
            
            if total_runtime > 0:
                avg_fps = self.pipeline_stats['total_frames'] / total_runtime
                logger.info(f"  Average FPS: {avg_fps:.2f}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        finally:
            self.is_running = False
            logger.info("Pipeline cleanup completed")
    
    def stop_pipeline(self):
        """
        Stop the pipeline gracefully.
        """
        logger.info("Stopping pipeline...")
        self.shutdown_event.set()
        self.is_running = False


class PipelineOrchestrator:
    """
    High-level orchestrator for managing multiple pipeline instances.
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.active_pipelines = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_pipeline(self, config_path: str, pipeline_id: Optional[str] = None) -> str:
        """
        Run a pipeline instance.
        
        Args:
            config_path: Path to configuration file
            pipeline_id: Optional pipeline identifier
            
        Returns:
            Pipeline ID
        """
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time())}"
        
        if pipeline_id in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} is already running")
        
        try:
            # Create and start pipeline
            pipeline = PipelineManager(config_path)
            self.active_pipelines[pipeline_id] = pipeline
            
            logger.info(f"Starting pipeline {pipeline_id}")
            await pipeline.start_pipeline()
            
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to run pipeline {pipeline_id}: {e}")
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
            raise
        
        finally:
            # Cleanup
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
    
    def stop_pipeline(self, pipeline_id: str):
        """
        Stop a specific pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
        """
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Pipeline {pipeline_id} not found")
            return
        
        pipeline = self.active_pipelines[pipeline_id]
        pipeline.stop_pipeline()
        logger.info(f"Stopped pipeline {pipeline_id}")
    
    def stop_all_pipelines(self):
        """Stop all active pipelines."""
        for pipeline_id in list(self.active_pipelines.keys()):
            self.stop_pipeline(pipeline_id)
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get status of a specific pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Pipeline status dictionary
        """
        if pipeline_id not in self.active_pipelines:
            return {'status': 'not_found'}
        
        pipeline = self.active_pipelines[pipeline_id]
        
        return {
            'status': 'running' if pipeline.is_running else 'stopped',
            'start_time': pipeline.start_time,
            'stats': pipeline.pipeline_stats,
            'streams': len(pipeline.compression_workers)
        }
    
    def list_active_pipelines(self) -> List[str]:
        """
        List all active pipeline IDs.
        
        Returns:
            List of active pipeline IDs
        """
        return list(self.active_pipelines.keys())


# Utility functions for pipeline management

def validate_config(config_path: str) -> Dict[str, Any]:
    """
    Validate pipeline configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")
    
    # Validate required sections
    required_sections = ['input_videos', 'model', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate input videos
    input_videos = config.get('input_videos', [])
    if not input_videos:
        raise ValueError("No input videos configured")
    
    for i, video in enumerate(input_videos):
        if 'path' not in video or 'stream_id' not in video:
            raise ValueError(f"Invalid video config at index {i}")
        
        if not Path(video['path']).exists():
            logger.warning(f"Video file not found: {video['path']}")
    
    # Validate model config
    model_config = config.get('model', {})
    if 'name' not in model_config:
        raise ValueError("Model name not specified")
    
    logger.info("Configuration validation passed")
    return config


async def run_single_pipeline(config_path: str):
    """
    Convenience function to run a single pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Validate configuration
    validate_config(config_path)
    
    # Create and run pipeline
    pipeline = PipelineManager(config_path)
    await pipeline.start_pipeline()

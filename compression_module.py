"""
Video Compression Module for E2E Object Detection Pipeline

This module provides video compression functionality with streaming capabilities
to enable concurrent compression and detection processing.
"""

import subprocess
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import cv2
import numpy as np
from queue import Queue
import threading
import time

logger = logging.getLogger(__name__)


class VideoCompressor:
    """
    Handles video compression with streaming output capabilities.
    Supports both file-to-file compression and frame-by-frame streaming.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video compressor with configuration.
        
        Args:
            config: Configuration dictionary containing compression settings
        """
        self.config = config
        self.compression_config = config.get('compression', {})
        self.output_config = config.get('output', {})
        
        # Create output directories
        self.compressed_path = Path(self.output_config.get('compressed_videos_path', 'output/compressed'))
        self.compressed_path.mkdir(parents=True, exist_ok=True)
        
        # Compression parameters
        self.codec = self.compression_config.get('codec', 'libx265')
        self.crf = self.compression_config.get('crf', 28)
        self.preset = self.compression_config.get('preset', 'ultrafast')
        self.audio_codec = self.compression_config.get('audio_codec', 'aac')
        self.audio_bitrate = self.compression_config.get('audio_bitrate', '128k')
        self.downscale = self.compression_config.get('downscale', False)
        self.target_resolution = self.compression_config.get('target_resolution', '1920:1080')
        
    def compress_video_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Compress a video file using FFmpeg.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output compressed video (optional)
            
        Returns:
            Path to compressed video file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            subprocess.CalledProcessError: If compression fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            input_name = Path(input_path).stem
            output_path = self.compressed_path / f"{input_name}_compressed.mp4"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', input_path,
            '-vcodec', self.codec,
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-acodec', self.audio_codec,
            '-b:a', self.audio_bitrate
        ]
        
        if self.downscale:
            command += ['-vf', f'scale={self.target_resolution}']
        
        command.append(str(output_path))
        
        logger.info(f"Compressing video: {input_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Successfully compressed video: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Video compression failed: {e.stderr}")
            raise
    
    async def compress_video_async(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Asynchronously compress a video file.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output compressed video (optional)
            
        Returns:
            Path to compressed video file
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compress_video_file, input_path, output_path)


class StreamingVideoCompressor:
    """
    Handles real-time video compression with frame streaming capabilities.
    Enables concurrent compression and detection processing.
    """
    
    def __init__(self, config: Dict[str, Any], stream_id: str):
        """
        Initialize streaming video compressor.
        
        Args:
            config: Configuration dictionary
            stream_id: Unique identifier for this video stream
        """
        self.config = config
        self.stream_id = stream_id
        self.compression_config = config.get('compression', {})
        self.output_config = config.get('output', {})
        
        # Frame queues for streaming
        self.compressed_frame_queue = Queue(maxsize=config.get('performance', {}).get('frame_buffer_size', 30))
        self.is_running = False
        self.compression_thread = None
        
        # Video properties (will be set when video is opened)
        self.fps = None
        self.width = None
        self.height = None
        self.fourcc = None
        
        # Statistics
        self.frames_processed = 0
        self.compression_start_time = None
        
    def start_streaming_compression(self, input_path: str, frame_callback: Optional[Callable] = None):
        """
        Start streaming compression of a video file.
        
        Args:
            input_path: Path to input video file
            frame_callback: Optional callback function to receive compressed frames
        """
        if self.is_running:
            logger.warning(f"Streaming compression already running for {self.stream_id}")
            return
        
        self.is_running = True
        self.compression_start_time = time.time()
        
        # Start compression in a separate thread
        self.compression_thread = threading.Thread(
            target=self._compression_worker,
            args=(input_path, frame_callback),
            daemon=True
        )
        self.compression_thread.start()
        
        logger.info(f"Started streaming compression for {self.stream_id}: {input_path}")
    
    def _compression_worker(self, input_path: str, frame_callback: Optional[Callable] = None):
        """
        Worker function for streaming compression.
        
        Args:
            input_path: Path to input video file
            frame_callback: Optional callback function to receive frames
        """
        cap = None
        writer = None
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            # Get video properties
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust resolution if downscaling is enabled
            if self.compression_config.get('downscale', False):
                target_res = self.compression_config.get('target_resolution', '1920:1080').split(':')
                self.width, self.height = int(target_res[0]), int(target_res[1])
            
            # Setup output video writer if saving compressed video
            if self.output_config.get('save_detection_videos', True):
                output_path = Path(self.output_config.get('compressed_videos_path', 'output/compressed'))
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{self.stream_id}_compressed.mp4"
                
                self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_file), self.fourcc, self.fps, (self.width, self.height))
            
            logger.info(f"Compression worker started for {self.stream_id} - Resolution: {self.width}x{self.height}, FPS: {self.fps}")
            
            # Process frames
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"End of video reached for {self.stream_id}")
                    break
                
                # Resize frame if downscaling
                if self.compression_config.get('downscale', False):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Apply basic compression (quality reduction)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Adjust quality as needed
                _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)
                decompressed_frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)
                
                # Add frame to queue for detection pipeline
                if not self.compressed_frame_queue.full():
                    self.compressed_frame_queue.put({
                        'frame': decompressed_frame,
                        'timestamp': time.time(),
                        'frame_number': self.frames_processed,
                        'stream_id': self.stream_id,
                        'original_fps': self.fps,
                        'original_width': self.width,
                        'original_height': self.height
                    })
                else:
                    logger.warning(f"Frame buffer full for {self.stream_id}, dropping frame")
                
                # Write to output video if enabled
                if writer is not None:
                    writer.write(decompressed_frame)
                
                # Call frame callback if provided
                if frame_callback:
                    frame_callback(decompressed_frame, self.stream_id, self.frames_processed)
                
                self.frames_processed += 1
                
                # No artificial delay - process frames as fast as possible
        
        except Exception as e:
            logger.error(f"Error in compression worker for {self.stream_id}: {e}")
        
        finally:
            # Cleanup
            if cap:
                cap.release()
            if writer:
                writer.release()
            
            self.is_running = False
            logger.info(f"Compression worker stopped for {self.stream_id}. Processed {self.frames_processed} frames")
    
    def get_compressed_frame(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get the next compressed frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Dictionary containing frame data or None if timeout
        """
        try:
            return self.compressed_frame_queue.get(timeout=timeout)
        except:
            return None
    
    def stop_compression(self):
        """Stop the streaming compression."""
        self.is_running = False
        if self.compression_thread and self.compression_thread.is_alive():
            self.compression_thread.join(timeout=5.0)
        
        logger.info(f"Stopped streaming compression for {self.stream_id}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary containing compression statistics
        """
        elapsed_time = time.time() - (self.compression_start_time or time.time())
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'stream_id': self.stream_id,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'queue_size': self.compressed_frame_queue.qsize(),
            'is_running': self.is_running
        }


def create_compressor(config: Dict[str, Any], stream_id: Optional[str] = None) -> VideoCompressor:
    """
    Factory function to create appropriate compressor instance.
    
    Args:
        config: Configuration dictionary
        stream_id: Stream ID for streaming compressor (optional)
        
    Returns:
        VideoCompressor or StreamingVideoCompressor instance
    """
    if stream_id:
        return StreamingVideoCompressor(config, stream_id)
    else:
        return VideoCompressor(config)

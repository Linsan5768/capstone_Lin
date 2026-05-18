#!/usr/bin/env python3
"""
Smart Cattle Detection System
Capable of determining movement direction, intelligently locating hips, and handling fast-moving targets
"""

import cv2
import numpy as np
import time
from collections import deque
from enhanced_cattle_detector import EnhancedCattleDetector
from config import *


class SmartCattleDetector(EnhancedCattleDetector):
    """
    Smart Cattle Detector
    Inherits from EnhancedCattleDetector, adding smart direction judgment and motion tracking
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", confidence_threshold: float = 0.3):
        """
        Initialize the smart detector
        
        Args:
            model_path: YOLO model path
            confidence_threshold: Detection confidence threshold (lower to increase detection rate)
        """
        super().__init__(model_path, confidence_threshold)
        
        # Motion tracking parameters
        self.tracking_history = {}  # Tracking history
        self.max_tracking_frames = 10  # Maximum tracking frames
        self.motion_threshold = 20  # Motion threshold (pixels)
        
        # Smart hip detection parameters
        self.direction_history = deque(maxlen=5)  # 方向历史
        self.hip_side_ratio = 0.3  # 髋部区域比例
        
        # Fast motion detection parameters
        self.fast_motion_threshold = 50  # Fast motion threshold
        self.detection_interval = 1  # Detection interval (detect every N frames)
        self.interpolation_frames = 3  # Interpolation frames
        
    def detect_cattle_smart(self, frame: np.ndarray, frame_id: int = 0) -> list:
        """
        Smart cattle detection
        
        Args:
            frame: Input frame
            frame_id: Frame ID
            
        Returns:
            List of detection results
        """
        # Basic detection
        detections = self.detect_cattle_enhanced(frame)
        
        # Intelligently process each detection result
        smart_detections = []
        for detection in detections:
            smart_detection = self._process_detection_smart(detection, frame, frame_id)
            if smart_detection:
                smart_detections.append(smart_detection)
        
        # Update tracking history
        self._update_tracking_history(smart_detections, frame_id)
        
        # Handle fast-moving targets
        smart_detections = self._handle_fast_motion(smart_detections, frame_id)
        
        return smart_detections
    
    def _process_detection_smart(self, detection: dict, frame: np.ndarray, frame_id: int) -> dict:
        """
        Intelligently process a single detection result
        
        Args:
            detection: Original detection result
            frame: Input frame
            frame_id: Frame ID
            
        Returns:
            Intelligently processed detection result
        """
        # Boundary check
        detection = self._check_boundaries(detection, frame.shape)
        
        # Smart hip detection
        smart_hip = self._detect_hip_smart(detection, frame, frame_id)
        detection['hip_top'] = smart_hip
        
        # Determine movement direction
        direction = self._determine_movement_direction(detection, frame_id)
        detection['movement_direction'] = direction
        
        # Smart ROI adjustment
        detection = self._adjust_roi_smart(detection, direction)
        
        return detection
    
    def _check_boundaries(self, detection: dict, frame_shape: tuple) -> dict:
        """
        Check and correct boundaries
        
        Args:
            detection: Detection result
            frame_shape: Frame shape (height, width, channels)
            
        Returns:
            Corrected detection result
        """
        height, width = frame_shape[:2]
        
        # Correct body region boundaries
        body_box = detection['body_box']
        x1, y1, x2, y2 = body_box
        
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        detection['body_box'] = (x1, y1, x2, y2)
        
        # Correct ROI region boundaries
        roi_box = detection['roi_box']
        rx1, ry1, rx2, ry2 = roi_box
        
        rx1 = max(0, min(rx1, width))
        ry1 = max(0, min(ry1, height))
        rx2 = max(0, min(rx2, width))
        ry2 = max(0, min(ry2, height))
        
        detection['roi_box'] = (rx1, ry1, rx2, ry2)
        
        # Correct hip top point boundaries
        hip_top = detection['hip_top']
        hx, hy = hip_top
        hx = max(0, min(hx, width))
        hy = max(0, min(hy, height))
        
        detection['hip_top'] = (hx, hy)
        
        return detection
    
    def _detect_hip_smart(self, detection: dict, frame: np.ndarray, frame_id: int) -> tuple:
        """
        Smart hip detection
        
        Args:
            detection: Detection result
            frame: Input frame
            frame_id: Frame ID
            
        Returns:
            Intelligently detected hip top point coordinates
        """
        body_box = detection['body_box']
        x1, y1, x2, y2 = body_box
        
        # Get body region
        body_region = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if body_region.size == 0:
            return detection['hip_top']
        
        # Determine movement direction
        direction = self._determine_movement_direction(detection, frame_id)
        
        # Intelligently select hip region based on movement direction
        if direction == 'left_to_right':
            # Moving left to right, hip is on the right
            hip_region = self._extract_hip_region_right(body_region)
        elif direction == 'right_to_left':
            # Moving right to left, hip is on the left
            hip_region = self._extract_hip_region_left(body_region)
        else:
            # Direction unclear, use default method
            hip_region = self._extract_hip_region_default(body_region)
        
        # Detect the highest point in the hip region
        hip_point = self._detect_hip_in_region(hip_region)
        
        # Convert to original image coordinates
        if direction == 'left_to_right':
            # Hip is on the right
            hip_x = x1 + hip_point[0] + int((x2 - x1) * (1 - self.hip_side_ratio))
        elif direction == 'right_to_left':
            # Hip is on the left
            hip_x = x1 + hip_point[0]
        else:
            # Default position
            hip_x = x1 + hip_point[0] + int((x2 - x1) * self.hip_side_ratio)
        
        hip_y = y1 + hip_point[1]
        
        return (int(hip_x), int(hip_y))
    
    def _extract_hip_region_right(self, body_region: np.ndarray) -> np.ndarray:
        """Extract right-side hip region"""
        h, w = body_region.shape[:2]
        start_x = int(w * (1 - self.hip_side_ratio))
        return body_region[:, start_x:]
    
    def _extract_hip_region_left(self, body_region: np.ndarray) -> np.ndarray:
        """Extract left-side hip region"""
        h, w = body_region.shape[:2]
        end_x = int(w * self.hip_side_ratio)
        return body_region[:, :end_x]
    
    def _extract_hip_region_default(self, body_region: np.ndarray) -> np.ndarray:
        """Extract default hip region (upper part)"""
        h, w = body_region.shape[:2]
        end_y = int(h * 0.6)  # 上半部分60%
        return body_region[:end_y, :]
    
    def _detect_hip_in_region(self, hip_region: np.ndarray) -> tuple:
        """
        Detect the highest point in the hip region
        
        Args:
            hip_region: Hip region image
            
        Returns:
            Hip top point coordinates (relative to the region)
        """
        if hip_region.size == 0:
            return (0, 0)
        
        # Convert to grayscale
        if len(hip_region.shape) == 3:
            gray = cv2.cvtColor(hip_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hip_region
        
        # Use multiple methods to detect the highest point
        methods = [
            self._detect_top_by_contour,
            self._detect_top_by_edge,
            self._detect_top_by_gradient
        ]
        
        candidates = []
        for method in methods:
            try:
                point = method(gray)
                if point:
                    candidates.append(point)
            except:
                continue
        
        if not candidates:
            # If all methods fail, return the top-center of the region
            h, w = gray.shape
            return (w // 2, 0)
        
        # Select the point with the minimum Y coordinate (highest)
        best_point = min(candidates, key=lambda x: x[1])
        return best_point
    
    def _detect_top_by_contour(self, gray: np.ndarray) -> tuple:
        """Detect the highest point by contour"""
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find the highest point
        top_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        return top_point
    
    def _detect_top_by_edge(self, gray: np.ndarray) -> tuple:
        """Detect the highest point by edge"""
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find edge points
        edge_points = np.where(edges > 0)
        
        if len(edge_points[0]) == 0:
            return None
        
        # Find the highest point
        min_y_idx = np.argmin(edge_points[0])
        top_x = edge_points[1][min_y_idx]
        top_y = edge_points[0][min_y_idx]
        
        return (top_x, top_y)
    
    def _detect_top_by_gradient(self, gray: np.ndarray) -> tuple:
        """Detect the highest point by gradient"""
        # Calculate vertical gradient
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)
        
        # Find the point with the maximum gradient
        max_grad_idx = np.unravel_index(np.argmax(grad_y), grad_y.shape)
        
        return (max_grad_idx[1], max_grad_idx[0])
    
    def _determine_movement_direction(self, detection: dict, frame_id: int) -> str:
        """
        Determine movement direction
        
        Args:
            detection: Detection result
            frame_id: Frame ID
            
        Returns:
            Movement direction: 'left_to_right', 'right_to_left', 'unknown'
        """
        # Get the center point of the current detection
        body_box = detection['body_box']
        center_x = (body_box[0] + body_box[2]) / 2
        
        # Add to direction history
        self.direction_history.append((frame_id, center_x))
        
        # If history is insufficient, return unknown
        if len(self.direction_history) < 3:
            return 'unknown'
        
        # Calculate movement trend
        recent_positions = [pos[1] for pos in list(self.direction_history)[-3:]]
        
        # Calculate movement speed
        if len(recent_positions) >= 2:
            movement = recent_positions[-1] - recent_positions[0]
            
            if movement > self.motion_threshold:
                return 'left_to_right'
            elif movement < -self.motion_threshold:
                return 'right_to_left'
        
        return 'unknown'
    
    def _adjust_roi_smart(self, detection: dict, direction: str) -> dict:
        """
        Intelligently adjust ROI based on movement direction - hip box is always on the right
        
        Args:
            detection: Detection result
            direction: Movement direction
            
        Returns:
            Adjusted detection result
        """
        body_box = detection['body_box']
        x1, y1, x2, y2 = body_box
        
        # Hip box is always on the right (regardless of movement direction)
        # Right 30% is the hip ROI, left 70% is the rest of the body
        roi_x1 = x1 + int((x2 - x1) * (1 - self.hip_side_ratio))  # Right 30%
        roi_x2 = x2
        
        detection['roi_box'] = (roi_x1, y1, roi_x2, y2)
        return detection
    
    def _update_tracking_history(self, detections: list, frame_id: int):
        """Update tracking history"""
        for i, detection in enumerate(detections):
            track_id = f"cattle_{i}"
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = deque(maxlen=self.max_tracking_frames)
            
            # Record detection information
            track_info = {
                'frame_id': frame_id,
                'body_box': detection['body_box'],
                'hip_top': detection['hip_top'],
                'movement_direction': detection.get('movement_direction', 'unknown')
            }
            
            self.tracking_history[track_id].append(track_info)
    
    def _handle_fast_motion(self, detections: list, frame_id: int) -> list:
        """
        Handle fast-moving targets
        
        Args:
            detections: Current frame's detection results
            frame_id: Frame ID
            
        Returns:
            Processed detection results
        """
        # If fast motion is detected, perform interpolation
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detections.append(detection)
            
            # Check for fast motion
            if self._is_fast_motion(detection, frame_id):
                # Generate interpolated detection results
                interpolated = self._interpolate_detection(detection, frame_id)
                if interpolated:
                    enhanced_detections.extend(interpolated)
        
        return enhanced_detections
    
    def _is_fast_motion(self, detection: dict, frame_id: int) -> bool:
        """Determine if it is fast motion"""
        # Fast motion judgment logic can be added here
        # E.g., check position change between consecutive frames
        return False  # Temporarily return False
    
    def _interpolate_detection(self, detection: dict, frame_id: int) -> list:
        """Interpolate to generate detection results for intermediate frames"""
        # Interpolation logic can be added here
        return []
    
    def process_video_smart(self, video_path: str, output_path: str = None, show_preview: bool = True):
        """
        Intelligently process video
        
        Args:
            video_path: Input video path
            output_path: Output video path
            show_preview: Whether to show preview
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        print(f"使用模型: {self.model.ckpt_path}")
        print(f"置信度阈值: {self.confidence_threshold}")
        print("-" * 60)
        
        # Set up output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Smart detection
                detections = self.detect_cattle_smart(frame, frame_count)
                
                # Draw detection results
                result_frame = self.draw_detections_smart(frame, detections)
                
                # Add frame information
                info_text = f"Frame: {frame_count}/{total_frames} | Smart Detections: {len(detections)}"
                cv2.putText(result_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
                
                # Add FPS information
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(result_frame, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
                
                # Save output video
                if out:
                    out.write(result_frame)
                
                # Show results
                if show_preview:
                    cv2.imshow('Smart Cattle Detection', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Show progress
                if frame_count % 30 == 0:
                    print(f"Processing progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - {current_fps:.1f} FPS")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Output statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\nSmart detection complete!")
            print(f"Total Frames: {frame_count}")
            print(f"Total Time: {total_time:.2f} 秒")
            print(f"Average FPS: {avg_fps:.2f}")
            if output_path:
                print(f"Video Output Path: {output_path}")
    
    def draw_detections_smart(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Intelligently draw detection results
        
        Args:
            frame: Input frame
            detections: List of detection results
            
        Returns:
            Frame with detection results drawn
        """
        result_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            # Draw body region (blue)
            body_box = detection['body_box']
            cv2.rectangle(result_frame, 
                         (int(body_box[0]), int(body_box[1])), 
                         (int(body_box[2]), int(body_box[3])), 
                         COLORS['body'], 2)
            
            # Draw smart ROI region (green)
            roi_box = detection['roi_box']
            cv2.rectangle(result_frame, 
                         (int(roi_box[0]), int(roi_box[1])), 
                         (int(roi_box[2]), int(roi_box[3])), 
                         COLORS['roi'], 2)
            
            # Draw smart hip top point (red)
            hip_top = detection['hip_top']
            cv2.circle(result_frame, hip_top, 6, COLORS['hip_top'], -1)
            cv2.circle(result_frame, hip_top, 10, COLORS['hip_top'], 2)
            
            # Add smart information
            confidence = detection['confidence']
            direction = detection.get('movement_direction', 'unknown')
            
            # Confidence label
            label = f"Cattle {i+1}: {confidence:.2f}"
            cv2.putText(result_frame, label, 
                       (int(body_box[0]), int(body_box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
            
            # Movement direction label
            direction_label = f"Dir: {direction}"
            cv2.putText(result_frame, direction_label, 
                       (int(body_box[0]), int(body_box[1]) + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['roi'], 1)
            
            # Hip top point coordinates
            hip_label = f"Hip: ({hip_top[0]}, {hip_top[1]})"
            cv2.putText(result_frame, hip_label, 
                       (int(body_box[0]), int(body_box[1]) + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['hip_top'], 1)
        
        return result_frame


def main():
    """Main function - demonstrate the smart detection system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Cattle Detection System')
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Input video file path')
    parser.add_argument('--output', '-o', type=str,
                       help='Output video file path')
    parser.add_argument('--model', '-m', type=str, default='yolo11n.pt',
                       help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Create smart detector
    detector = SmartCattleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Process video
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {args.video}")
        return 1
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Smart Cattle Detection System Starting")
    print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} Frames")
    print(f"Using Model: {args.model}")
    print(f"Confidence Threshold: {args.confidence}")
    print("-" * 60)
    
    # Set up output video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Smart detection
            detections = detector.detect_cattle_smart(frame, frame_count)
            
            # Draw detection results
            result_frame = detector.draw_detections_smart(frame, detections)
            
            # Add frame information
            info_text = f"Frame: {frame_count}/{total_frames} | Smart Detections: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
            
            # Save output video
            if out:
                out.write(result_frame)
            
            # Show results
            cv2.imshow('Smart Cattle Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"Processing progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - {fps_current:.1f} FPS")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Output statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\nSmart detection complete!")
        print(f"Total Frames: {frame_count}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        if args.output:
            print(f"Output Video: {args.output}")


if __name__ == "__main__":
    main()
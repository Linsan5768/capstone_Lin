import cv2
import numpy as np
from ultralytics import YOLO
import torch
from scipy import ndimage
from skimage import measure, morphology, filters
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from collections import deque
from config import *


class EnhancedCattleDetector:
    """
    Enhanced cattle inspection system
    Has higher robustness and real-time performance
    """
    
    def __init__(self, model_path: str = YOLO_CONFIG["model_path"], 
                 confidence_threshold: float = YOLO_CONFIG["confidence_threshold"]):
        """
        Initialize the enhanced detection system
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Use the parameters in the configuration file
        self.roi_ratio = DETECTION_CONFIG["roi_ratio"]
        self.hip_search_radius = DETECTION_CONFIG["hip_search_radius"]
        self.min_contour_area = DETECTION_CONFIG["min_contour_area"]
        self.kernel_size = DETECTION_CONFIG["morphology_kernel_size"]
        
        # Time series cache, used to improve robustness
        self.hip_history = deque(maxlen=10)  # Save the highest point of the last 10 frames
        self.detection_history = deque(maxlen=5)  # Save the test results of the last 5 frames
        
        # Performance optimization
        self.use_gpu = PERFORMANCE_CONFIG["use_gpu"] and torch.cuda.is_available()
        if self.use_gpu:
            print("Use GPU acceleration")
        else:
            print("Use CPU processing")
    
    def detect_cattle_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """
        Enhanced version of cattle detection, with higher robustness
        """
        # Preprocessing images
        processed_frame = self._preprocess_frame(frame)
        
        # YOLO detection
        results = self.model(processed_frame, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Extract the ROI area
                    roi_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Enhanced ROI and body detection
                    roi_box, body_box = self._detect_roi_and_body_enhanced(roi_frame, x1, y1)
                    
                    # Enhanced hip height detection
                    hip_top = self._detect_hip_top_enhanced(roi_frame, roi_box, x1, y1)
                    
                    # Time series smoothing
                    if ROBUSTNESS_CONFIG["temporal_smoothing"]:
                        hip_top = self._temporal_smooth_hip(hip_top)
                    
                    detection = {
                        'body_box': body_box,
                        'roi_box': roi_box,
                        'hip_top': hip_top,
                        'confidence': confidence,
                        'original_box': (x1, y1, x2, y2),
                        'frame_timestamp': time.time()
                    }
                    detections.append(detection)
        
        # Update detection history
        self.detection_history.append(detections)
        
        return detections
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Image preprocessing to improve inspection quality
        """
        # Histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def _detect_roi_and_body_enhanced(self, roi_frame: np.ndarray, offset_x: float, offset_y: float) -> Tuple[Tuple, Tuple]:
        """
        Enhanced ROI and body area detection
        """
        if roi_frame.size == 0:
            return (0, 0, 0, 0), (0, 0, 0, 0)
        
        # Multichannel processing
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Use a combination of multiple methods
        # Method 1: Segmentation based on color space
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask1 = self._create_color_mask(hsv)
        
        # Method 2: Texture-based segmentation
        mask2 = self._create_texture_mask(gray)
        
        # Method 3: Edge-based segmentation
        mask3 = self._create_edge_mask(gray)
        
        # Merge multiple masks
        combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        # Morphological operation
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find outline
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            h, w = roi_frame.shape[:2]
            return (0, 0, w, h), (0, 0, w, h)
        
        # Choose the largest outline
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the body area
        x, y, w, h = cv2.boundingRect(largest_contour)
        body_box = (x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y)
        
        # Intelligent ROI area detection
        roi_box = self._smart_roi_detection(largest_contour, x, y, w, h, offset_x, offset_y)
        
        return roi_box, body_box
    
    def _create_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create a mask based on color"""
        # Define the HSV color range of cattle
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        mask1 = cv2.inRange(hsv, lower_brown, upper_brown)
        mask2 = cv2.inRange(hsv, lower_black, upper_black)
        
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_texture_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create a mask based on texture"""
        # Use a simplified version of local binary mode (LBP)
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # Create a mask based on the variance threshold
        _, mask = cv2.threshold(variance.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _create_edge_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create a mask based on the edge"""
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        
        # Combine the edges of different scales
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Expansion operation to connect the edge
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(combined_edges, kernel, iterations=2)
        
        return mask
    
    def _smart_roi_detection(self, contour: np.ndarray, x: int, y: int, w: int, h: int, 
                           offset_x: float, offset_y: float) -> Tuple:
        """
        Intelligent ROI area detection, based on contour shape analysis
        Modified to split left and right: the left side is the hip ROI, and the right side is the rest of the body
        """
        # Analyze the convex hull of the contour
        hull = cv2.convexHull(contour)
        
        # Find the highest point of the outline
        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
        
        # Determine the ROI area based on the contour shape
        # Left and right division: 30% of the left side is the hip ROI, and 70% of the right side is the rest of the body
        aspect_ratio = w / h
        if aspect_ratio < 0.5:  # Narrower outline
            roi_ratio = self.roi_ratio * 0.8
        elif aspect_ratio > 1.5:  # Wide outline
            roi_ratio = self.roi_ratio * 1.2
        else:
            roi_ratio = self.roi_ratio
        
        # Calculate the ROI area-split left and right
        roi_width = int(w * roi_ratio)  # Left width
        roi_x = x  # Start from the left
        
        roi_box = (roi_x + offset_x, y + offset_y, 
                  roi_x + roi_width + offset_x, y + h + offset_y)
        
        return roi_box
    
    def _detect_hip_top_enhanced(self, roi_frame: np.ndarray, roi_box: Tuple, 
                               offset_x: float, offset_y: float) -> Tuple[int, int]:
        """
        Enhanced version of hip high point detection, using a variety of methods to integrate
        """
        if roi_frame.size == 0:
            return (0, 0)
        
        # Extract the ROI area
        x1, y1, x2, y2 = roi_box
        roi_x1, roi_y1 = int(x1 - offset_x), int(y1 - offset_y)
        roi_x2, roi_y2 = int(x2 - offset_x), int(y2 - offset_y)
        
        # Make sure the coordinates are within the image range
        h, w = roi_frame.shape[:2]
        roi_x1 = max(0, min(roi_x1, w))
        roi_y1 = max(0, min(roi_y1, h))
        roi_x2 = max(0, min(roi_x2, w))
        roi_y2 = max(0, min(roi_y2, h))
        
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return (int(offset_x + w//2), int(offset_y + h//2))
        
        roi_region = roi_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi_region.size == 0:
            return (int(offset_x + w//2), int(offset_y + h//2))
        
        # Use a variety of methods to detect hip heights
        hip_candidates = []
        
        if ROBUSTNESS_CONFIG["multi_method_detection"]:
            # Method 1: Based on contour
            hip1 = self._detect_hip_by_contour_enhanced(roi_region, roi_x1, roi_y1)
            if hip1:
                hip_candidates.append(hip1)
            
            # Method 2: Based on the edge
            hip2 = self._detect_hip_by_edge_enhanced(roi_region, roi_x1, roi_y1)
            if hip2:
                hip_candidates.append(hip2)
            
            # Method 3: Based on morphology
            hip3 = self._detect_hip_by_morphology_enhanced(roi_region, roi_x1, roi_y1)
            if hip3:
                hip_candidates.append(hip3)
            
            # Method 4: Based on feature points
            hip4 = self._detect_hip_by_features(roi_region, roi_x1, roi_y1)
            if hip4:
                hip_candidates.append(hip4)
        
        if not hip_candidates:
            # If all methods fail, return to the top of the center of the ROI area
            center_x = roi_x1 + (roi_x2 - roi_x1) // 2
            center_y = roi_y1
            return (int(center_x + offset_x), int(center_y + offset_y))
        
        # Choose the best hip height
        best_hip = self._select_best_hip_point(hip_candidates)
        
        return (int(best_hip[0] + offset_x), int(best_hip[1] + offset_y))
    
    def _detect_hip_by_contour_enhanced(self, roi_region: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[int, int]]:
        """Enhanced contour detection"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # Multi-threshold binarization
            _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, binary2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Combine two binarization results
            binary = cv2.bitwise_or(binary1, binary2)
            
            # Morphological operation
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find outline
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Choose the largest outline
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Find the highest point of the outline
            top_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            
            return (top_point[0] + offset_x, top_point[1] + offset_y)
        except:
            return None
    
    def _detect_hip_by_edge_enhanced(self, roi_region: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[int, int]]:
        """Enhanced edge detection"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges3 = cv2.Canny(gray, 100, 200)
            
            # Combine the edges of different scales
            combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Find edge points
            edge_points = np.where(combined_edges > 0)
            
            if len(edge_points[0]) == 0:
                return None
            
            # Find the point with the smallest Y coordinate (the highest point)
            min_y_idx = np.argmin(edge_points[0])
            top_x = edge_points[1][min_y_idx]
            top_y = edge_points[0][min_y_idx]
            
            return (top_x + offset_x, top_y + offset_y)
        except:
            return None
    
    def _detect_hip_by_morphology_enhanced(self, roi_region: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[int, int]]:
        """Enhanced morphological detection"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operation
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels <= 1:
                return None
            
            # Choose the largest connected component
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            
            # Get the bounding box of the component
            x = stats[largest_component, cv2.CC_STAT_LEFT]
            y = stats[largest_component, cv2.CC_STAT_TOP]
            
            return (x + offset_x, y + offset_y)
        except:
            return None
    
    def _detect_hip_by_features(self, roi_region: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[int, int]]:
        """Hip height detection based on feature points"""
        try:
            gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
            
            # Use Harris corner detection
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            # Find the corner
            corner_points = np.where(corners > 0.01 * corners.max())
            
            if len(corner_points[0]) == 0:
                return None
            
            # Find the corner point with the smallest Y coordinate (the highest point)
            min_y_idx = np.argmin(corner_points[0])
            top_x = corner_points[1][min_y_idx]
            top_y = corner_points[0][min_y_idx]
            
            return (top_x + offset_x, top_y + offset_y)
        except:
            return None
    
    def _select_best_hip_point(self, candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Choose the best hip high point from multiple candidate points
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # Calculate the center of all candidate points
        center_x = sum(point[0] for point in candidates) / len(candidates)
        center_y = sum(point[1] for point in candidates) / len(candidates)
        
        # Choose the point closest to the center
        best_point = min(candidates, key=lambda p: 
                        ((p[0] - center_x)**2 + (p[1] - center_y)**2)**0.5)
        
        return best_point
    
    def _temporal_smooth_hip(self, current_hip: Tuple[int, int]) -> Tuple[int, int]:
        """
        The time series is smooth, and the stability of high-point detection is improved
        """
        self.hip_history.append(current_hip)
        
        if len(self.hip_history) < 3:
            return current_hip
        
        # Calculate the weighted average
        weights = np.linspace(0.5, 1.0, len(self.hip_history))
        weights = weights / weights.sum()
        
        avg_x = sum(hip[0] * w for hip, w in zip(self.hip_history, weights))
        avg_y = sum(hip[1] * w for hip, w in zip(self.hip_history, weights))
        
        return (int(avg_x), int(avg_y))
    
    def draw_detections_enhanced(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Enhanced version of the test result drawing
        """
        result_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            # Draw the body area (blue)
            body_box = detection['body_box']
            cv2.rectangle(result_frame, 
                         (int(body_box[0]), int(body_box[1])), 
                         (int(body_box[2]), int(body_box[3])), 
                         COLORS['body'], 2)
            
            # Draw the hip ROI area (green)
            roi_box = detection['roi_box']
            cv2.rectangle(result_frame, 
                         (int(roi_box[0]), int(roi_box[1])), 
                         (int(roi_box[2]), int(roi_box[3])), 
                         COLORS['roi'], 2)
            
            # Draw hip high points (red dots)
            hip_top = detection['hip_top']
            cv2.circle(result_frame, hip_top, 6, COLORS['hip_top'], -1)
            cv2.circle(result_frame, hip_top, 10, COLORS['hip_top'], 2)
            
            # Add confidence label
            confidence = detection['confidence']
            label = f"Cattle {i+1}: {confidence:.2f}"
            cv2.putText(result_frame, label, 
                       (int(body_box[0]), int(body_box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
            
            # Add hip height coordinates
            hip_label = f"Hip: ({hip_top[0]}, {hip_top[1]})"
            cv2.putText(result_frame, hip_label, 
                       (int(body_box[0]), int(body_box[1]) + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['hip_top'], 1)
        
        return result_frame
    
    def process_video_enhanced(self, video_path: str, output_path: str = None, show_preview: bool = True):
        """
        Enhanced video processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} 帧")
        print(f"Using the model: {YOLO_CONFIG['model_path']}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Set the output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["output_codec"])
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        fps_history = deque(maxlen=30)  # Save the FPS of the last 30 frames
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_start = time.time()
                
                # Detection of cattle
                detections = self.detect_cattle_enhanced(frame)
                
                # Draw test results
                result_frame = self.draw_detections_enhanced(frame, detections)
                
                # Add frame information
                if VIDEO_CONFIG["show_frame_count"]:
                    info_text = f"Frame: {frame_count}/{total_frames}"
                    cv2.putText(result_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
                
                if VIDEO_CONFIG["show_detection_count"]:
                    detection_text = f"Cattle: {len(detections)}"
                    cv2.putText(result_frame, detection_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
                
                if VIDEO_CONFIG["show_fps"]:
                    frame_time = time.time() - frame_start
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_history.append(current_fps)
                    avg_fps = sum(fps_history) / len(fps_history)
                    
                    fps_text = f"FPS: {avg_fps:.1f}"
                    cv2.putText(result_frame, fps_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
                
                # Save output video
                if out:
                    out.write(result_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow(VIDEO_CONFIG["preview_window_name"], result_frame)
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
            print(f"\nProcessing completed!")
            print(f"Total frames: {frame_count}")
            print(f"Total time: {total_time:.2f} 秒")
            print(f"Average FPS: {avg_fps:.2f}")
            if output_path:
                print(f"Video output path: {output_path}")


def main():
    """Main function-Demonstration to enhance the use of the system"""
    # Initialize the enhanced detection system
    detector = EnhancedCattleDetector()
    
    # Process video
    video_path = "DepthData_testing_ 2025-10-14 at 2.08.53 pm.mov"
    output_path = "enhanced_cattle_detection_output.mp4"
    
    try:
        detector.process_video_enhanced(video_path, output_path, show_preview=True)
    except Exception as e:
        print(f"Error while processing video: {e}")


if __name__ == "__main__":
    main()

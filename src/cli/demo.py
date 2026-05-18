#!/usr/bin/env python3
"""
Intelligent cattle inspection system with integrated AprilTag
Can detect AprilTag and calculate the pixel-to-centimeter conversion, and then calculate the hip height of the cow
Buttons - g - calibrate groundline, r - reset, q - quit
"""

import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from smart_cattle_detector import SmartCattleDetector
from config import *


class CattleAprilTagCsvLogger:
    
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.path = os.path.join("logs", f"apriltag_cattle_heights_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "timestamp_iso", "cattle_id", "hip_height_cm", "", 
            "confidence", "frame_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        ])
        self._f.flush()
        print(f"[CSV] Logging to: {self.path}")

    def log(self, cattle_id, hip_height_cm, confidence, frame_id, bbox):
        """Log a cattle measurement row."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = bbox if bbox is not None else (None, None, None, None)
        self._w.writerow([
            ts,
            int(cattle_id) if cattle_id is not None else "",
            f"{hip_height_cm:.2f}" if hip_height_cm is not None else "",
            "",  # Blank column for separation
            f"{confidence:.3f}" if confidence is not None else "",
            int(frame_id) if frame_id is not None else "",
            int(x1) if x1 is not None else "",
            int(y1) if y1 is not None else "",
            int(x2) if x2 is not None else "",
            int(y2) if y2 is not None else ""
        ])
        self._f.flush()

    def log_separator(self,reason="AUTO"):
        """Log a separator row to mark new cattle."""
        self._w.writerow([f"NEW_CATTLE_{reason}"] + [""] * 9)
        self._f.flush()

    def close(self):
        """Close the CSV file."""
        try:
            self._f.close()
        except Exception:
            pass


class AprilTagCattleDetector(SmartCattleDetector):
    """
    Intelligent cattle detector with integrated AprilTag
    """
    
    def __init__(self, model_path: str = "yolo11n-seg.pt", confidence_threshold: float = 0.3, 
                 detect_cows_only: bool = False, high_precision_mask: bool = True):
        """
        Initialize the AprilTag cattle detector (using the YOLO segmentation model)
        
        Args:
            model_path: YOLO split model path
            confidence_threshold: Detection confidence threshold
            detect_cows_only: Whether only cattle are detected (True = only cattle are detected, False = cattle and horses are detected)
            high_precision_mask: Whether to use high-precision mask processing (to improve the high measurement accuracy)
        """
        super().__init__(model_path, confidence_threshold)
        
        # Filter parameters (adjusted to looser conditions)
        self.min_confidence = 0.15  # Minimum confidence threshold
        self.min_box_size = 50     # Minimum inspection frame size
        self.edge_margin = 20      # Edge margin
        self.min_mask_area = 1000  # Minimum mask area
        
        # High-precision mask processing parameters
        self.high_precision_mask = high_precision_mask
        
        # Detection category settings
        self.detect_cows_only = detect_cows_only
        if detect_cows_only:
            self.target_classes = [19]  # Only detect cattle
        else:
            self.target_classes = [17, 19]  # Detect cattle and horses
        
        # Detection parameters
        self.apriltag_detector = None
        self.tag_distance_real = 1.0  # True distance between the two AprilTags (meters)
        self.pixels_per_cm = None  # Pixel to centimeter conversion ratio
        self.tag_positions = {}  # Store the detected AprilTag location
        
        # Initialize the AprilTag detector
        self._init_apriltag_detector()
        
        # Hip height calculation parameters
        self.ground_level = None  # Ground level (only set after calibration)
        self.ground_line_points = []  # Two points of interactive calibration [(x1,y1), (x2,y2)]
        self.ground_line_params = None  # (a, b, c) Linear parameters: ax + by + c = 0
        self.calibration_done = False  # Whether to complete the ground line calibration
        self.calibrate_ground = True  # Whether to enable interactive calibration
        self.hip_height_cm = {}  # Store each cow's hip height
        self.height_method = 'ground'  # 'ground' or 'bbox'
        
    def _init_apriltag_detector(self):
        """Initialise AprilTag detector"""
        try:
            # Attempt to import apriltag library
            import apriltag
            self.apriltag_detector = apriltag.Detector()
            print("AprilTag initialisation successful")
        except ImportError:
            print("Warning: apriltag library not installed，OpenCV's Aruco detector will be used")
            # Using OpenCV's Aruco detector as alternative
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.apriltag_detector = "aruco"
    
    def detect_apriltags(self, frame: np.ndarray) -> list:
        """
        Detecting AprilTag
        
        Args:
            frame: Input frame
            
        Returns:
            AprilTag test result list
        """
        if self.apriltag_detector is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = []
        
        if self.apriltag_detector == "aruco":
            # Using OpenCV's Aruco detection
            try:
                # Trying new version of OpenCV API
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                # Fall back to old OpenCV API
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                for i, corner in enumerate(corners):
                    tag_id = ids[i][0]
                    # Calculate the centre point of AprilTag
                    center = np.mean(corner[0], axis=0)
                    tags.append({
                        'id': tag_id,
                        'center': center,
                        'corners': corner[0],
                        'area': cv2.contourArea(corner[0])
                    })
        else:
            # Using apriltag Library
            detections = self.apriltag_detector.detect(gray)
            
            for detection in detections:
                tags.append({
                    'id': detection.tag_id,
                    'center': detection.center,
                    'corners': detection.corners,
                    'area': cv2.contourArea(detection.corners)
                })
        
        return tags
    
    def calculate_pixels_per_cm(self, tags: list) -> float:
        """
        Calculate pixel to centimeter conversion based on AprilTag 
        Using the vertical distance between the upper and lower legs
        
        Args:
            tags: AprilTag test result list
            
        Returns:
            Pixel to centimeter conversion ratio
        """
        if len(tags) < 2:
            return None
        
        # Select the upper and lower April Tags
        # Take the two points with the largest vertical distance 
        max_vertical = -1.0
        tag1, tag2 = None, None
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                y1 = float(tags[i]['center'][1])
                y2 = float(tags[j]['center'][1])
                vertical_gap = abs(y1 - y2)
                if vertical_gap > max_vertical:
                    max_vertical = vertical_gap
                    tag1, tag2 = tags[i], tags[j]

        if tag1 is None or tag2 is None or max_vertical <= 0:
            return None

        # Use vertical pixel distance （Δy）
        pixel_distance = max_vertical
        
        # Calculate the conversion ratio (pixels/cm)
        # Real life distance is 1m to 100cm
        pixels_per_cm = pixel_distance / (self.tag_distance_real * 100)
        
        return pixels_per_cm

    def _set_ground_line_from_points(self, p1: tuple, p2: tuple):
        """
        Set the horizontal ground line by clicking two points

        Take the average of the two points's y coords as the ground y

        """
        x1, y1 = p1
        x2, y2 = p2
        y0 = int(round((y1 + y2) / 2.0))
        # Leveled horizontal line: 0*x + 1*y - y0 = 0 => (a,b,c) = (0,1,-y0)
        a, b, c = 0.0, 1.0, -float(y0)
        self.ground_level = float(y0)
        self.ground_line_params = (a, b, c)
        self.ground_line_points = [(x1, y0), (x2, y0)]
        self.calibration_done = True

    def _run_interactive_calibration(self, frame, window_name: str = 'AprilTag Cattle Detection'):
        """
        Run a two point interactive ground line calibration

        Click two points to define the ground line

        Press S to skip
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        click_points = []

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_points.append((x, y))

        cv2.setMouseCallback(window_name, _on_mouse)
        info = frame.copy()
        cv2.putText(info, 'Click two points to set ground line (press S to skip)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        while len(click_points) < 2:
            preview = info.copy()
            for p in click_points:
                cv2.circle(preview, p, 6, (0, 255, 255), -1)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('s'), ord('S')):
                break
        if len(click_points) >= 2:
            self._set_ground_line_from_points(click_points[0], click_points[1])
        # If user skips, the fallback horizontal line solution is retained
        self.calibration_done = True
    
    def detect_cattle_with_apriltag(self, frame: np.ndarray, frame_id: int = 0) -> dict:
        """
        Detecting Cows and AprilTags

        Using the YOLO Segmentation Model
        
        Args:
            frame: Input Frame
            frame_id: Frame ID
            
        Returns:
            Dictionary that contains the cattle detection and AprilTag information
        """
        # Detect AprilTag
        tags = self.detect_apriltags(frame)
        
        # Updated pixel to centimeter conversion ratio
        if len(tags) >= 2:
            new_pixels_per_cm = self.calculate_pixels_per_cm(tags)
            if new_pixels_per_cm is not None:
                self.pixels_per_cm = new_pixels_per_cm
        
        # Using YOLO segmentation model to detect cattle
        cattle_detections = self._detect_cattle_with_segmentation(frame, frame_id)
        
        # Calcualte the Hip Height of each cow
        if self.pixels_per_cm is not None:
            cattle_detections = self._calculate_hip_heights(cattle_detections, frame)
        
        return {
            'cattle_detections': cattle_detections,
            'apriltags': tags,
            'pixels_per_cm': self.pixels_per_cm,
            'frame_id': frame_id
        }
    
    def _detect_cattle_with_segmentation(self, frame: np.ndarray, frame_id: int = 0) -> list:
        """
        Detecting Cattle

        Using the YOLO Segmentation Model
        
        Args:
            frame: Input Frame
            frame_id: Frame ID
            
        Returns:
            Cattle test results list
        """
        # Using YOLO model for segmentation detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        height, width = frame.shape[:2]
        
        for result in results:
            if result.masks is not None:  # Ensure there is segmentation mask
                for i, (box, mask, conf, cls) in enumerate(zip(result.boxes.xyxy, result.masks.data, result.boxes.conf, result.boxes.cls)):
                    # Handle cattle and horse classes (class 19=cow, class 17=horse for COCO dataset)
                    # Apply stricter filtering conditions
                    class_id = int(cls)
                    confidence = float(conf)

                    # The target class is processed according to the configuration and with high enough
                    # configuration level 
                    if class_id in self.target_classes and confidence >= self.min_confidence:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        
                        # Make sure coordinates are within image range
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(width, int(x2))
                        y2 = min(height, int(y2))
                        
                        # Add size filtering: avoid detection boxes that are too small
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        if box_width < self.min_box_size or box_height < self.min_box_size:
                            continue  # Skip detection boxes that are too small
                        
                        # Add position filtering： Avoid detection that are at the edge of the image
                        if (x1 < self.edge_margin or y1 < self.edge_margin or 
                            x2 > width - self.edge_margin or y2 > height - self.edge_margin):
                            continue  # Skip edge detection
                        
                        # Processing segmentation masks
                        mask_resized = mask.cpu().numpy()
                        if len(mask_resized.shape) == 3:
                            mask_resized = mask_resized[0]  # Take the first mask
                        
                        # Resize the mask to the original image
                        mask_full = np.zeros((height, width), dtype=np.uint8)
                        
                        if self.high_precision_mask:
                            # High Precision Mode：Use higher quality interpolation and finer processing
                            mask_resized_resized = cv2.resize(mask_resized, (width, height), interpolation=cv2.INTER_CUBIC)
                            
                            # Using a lower threshold to preserve more details
                            mask_full[mask_resized_resized > 0.2] = 255
                            
                            # Morphological operations to optimise mask edges（finer kernels）
                            kernel = np.ones((1, 1), np.uint8)  # smaller kernal，retaining finer details
                            mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
                            
                            # Edge refinement（finer）
                            mask_full = cv2.GaussianBlur(mask_full, (1, 1), 0)  # Samller blur kernel
                            mask_full[mask_full > 100] = 255  # lower threshold
                            mask_full[mask_full <= 100] = 0
                        else:
                            # Standard mode: fast processing
                            mask_resized_resized = cv2.resize(mask_resized, (width, height))
                            mask_full[mask_resized_resized > 0.5] = 255
                        
                        # Add mask quality filter：make sure the mask has enough area
                        mask_area = cv2.countNonZero(mask_full)
                        
                        if mask_area < self.min_mask_area:
                            continue  # Skip the detection of mask areas that are too small
                        
                        # Create detection results
                        detection = {
                            'body_box': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'cattle_mask': mask_full,
                            'hip_top': None,
                            'roi_box': None
                        }
                        
                        # Calculate hip ROI（right side 30%）
                        roi_width = int((x2 - x1) * 0.3)
                        roi_x1 = x2 - roi_width
                        detection['roi_box'] = (roi_x1, y1, x2, y2)
                        
                        # Find the highest point on the mask in the ROI area as the hip top
                        hip_top = self._find_hip_top_in_roi_mask(detection, mask_full)
                        detection['hip_top'] = hip_top
                        
                        detections.append(detection)
        
        return detections
    
    def _find_hip_top_in_roi_mask(self, detection: dict, mask: np.ndarray) -> tuple:
        """
        Find the highest point of the mask in the hip box (hip ROI)
        as the hip height point (high precision version)
        
        Args:
            detection: detection results
            mask: Segmentation mask
            
        Returns:
            Hip Height coordinates (The highest point of the mask)
        """
        roi_box = detection['roi_box']
        x1, y1, x2, y2 = roi_box
        
        # Make sure the coordinates are within the image range
        height, width = mask.shape
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        # Extract the hip box area
        hip_region = mask[y1:y2, x1:x2]
        
        if hip_region.size == 0:
            # If hip box area is empty，return the center point of the hip box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Method1: Use edge detection to find a more accurate top edge
        try:
            # Perform edge detection on the mask
            edges = cv2.Canny(hip_region, 50, 150)
            
            # Find all edge points
            edge_points = []
            for row in range(edges.shape[0]):
                for col in range(edges.shape[1]):
                    if edges[row, col] > 0:  # edge point
                        global_x = x1 + col
                        global_y = y1 + row
                        edge_points.append((global_x, global_y))
            
            if edge_points:
                # Select the uppermost edge point
                hip_top_edge = min(edge_points, key=lambda p: p[1])
                
                # Verify whether this point is on the mask
                if (hip_top_edge[0] < width and hip_top_edge[1] < height and 
                    mask[hip_top_edge[1], hip_top_edge[0]] > 0):
                    return hip_top_edge
        except:
            pass  # If the edge detection fails, fall back to the original method
        
        # Method2: Original method（as an alternative）
        hip_top_candidates = []
        
        # Scan the mask points within the hip box area，avoid borders
        for row in range(1, hip_region.shape[0] - 1):  # Avoid upper and lower boundaries
            for col in range(1, hip_region.shape[1] - 1):  # Avoid left and right boundaries
                if hip_region[row, col] > 0:  # On the mask
                    # Convert to global coordinates
                    global_x = x1 + col
                    global_y = y1 + row
                    hip_top_candidates.append((global_x, global_y))
        
        if not hip_top_candidates:
            # If no point is found inside the mask, fall back to searching for the bounding box
            for row in range(hip_region.shape[0]):
                for col in range(hip_region.shape[1]):
                    if hip_region[row, col] > 0:  # On the mask
                        global_x = x1 + col
                        global_y = y1 + row
                        hip_top_candidates.append((global_x, global_y))
        
        if not hip_top_candidates:
            # If still cannot find mask point, return to hip box center point
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Select the topmost point as the hip height point（with the smallest y coordinate）
        hip_top = min(hip_top_candidates, key=lambda p: p[1])
        
        return hip_top
    
    def _create_cattle_mask(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """
        Create a segmentation mask of the cow (Using the GrabCut algorithm)
        
        Args:
            frame: Input frame
            detection: Single cow detection results
            
        Returns:
            Cow segmentation mask
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Getting body area
        body_box = detection['body_box']
        x1, y1, x2, y2 = body_box
        
        # Ensure coordinates are within the image range
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        # Create a rectangular area
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        try:
            # Using the GrabCut algorithm for segmentation
            # Create foreground and background models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Implement GrabCut
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Creating the final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            mask2 = mask2 * 255
            
            # Use morphological opeartions to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            return mask2
            
        except Exception as e:
            # If GrabCut fails，fall back to a simpler rectangular mask
            print(f"GrabCut failed: {e}, using rectangle mask")
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask
    
    def _find_hip_top_in_mask(self, detection: dict, cattle_mask: np.ndarray) -> tuple:
        """
        Find the hip top point within the hip box area and on the cow mask
        
        Args:
            detection: Cattle test results
            cattle_mask: Cow segmentation mask
            
        Returns:
            Corrected hip top point coordinates
        """
        roi_box = detection['roi_box']
        x1, y1, x2, y2 = roi_box
        
        # Make sure the coordinates are within the image range
        height, width = cattle_mask.shape
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        # Extract the hip box area
        hip_region = cattle_mask[y1:y2, x1:x2]
        
        if hip_region.size == 0:
            # If the hip box area is empty, return to the original point 
            return detection['hip_top']
        
        # Find the highest point on the mask within the hip box area
        # Scan from bottom to top and find the first mask point
        hip_top_candidates = []
        
        for col in range(hip_region.shape[1]):
            for row in range(hip_region.shape[0] - 1, -1, -1):  # From bottom to top
                if hip_region[row, col] > 0:  # On the mask
                    # Convert to the global coordinates
                    global_x = x1 + col
                    global_y = y1 + row
                    hip_top_candidates.append((global_x, global_y))
                    break  # Stop when you find the highest point
        
        if not hip_top_candidates:
            # If no point is found on the mask, return to the original point
            return detection['hip_top']
        
        # Select the top point as the hip top
        hip_top = min(hip_top_candidates, key=lambda p: p[1])
        
        return hip_top
    
    def _calculate_hip_heights(self, detections: list, frame: np.ndarray) -> list:
        """
        Calcualte the hip height of each cow (Using YOLO segmentation mask)
        
        Args:
            detections: Cattle test results list
            frame: Input frame
            
        Returns:
            A list of test results taht contains the hip height information
        """
        if self.pixels_per_cm is None:
            return detections
        
        height, width = frame.shape[:2]
        
        # Estimate the ground level（Assumed to be in the bottom 80% of the image）
        if self.ground_level is None:
            self.ground_level = height * 0.8
        
        for i, detection in enumerate(detections):

            if 'cattle_mask' in detection:
                # Calculate hip height（Distance from the ground to the hip height）
                hip_height_pixels = self.ground_level - detection['hip_top'][1]
                hip_height_cm = (hip_height_pixels / self.pixels_per_cm) + 2 ## for the height weight
                
                # Store the hip height Information
                detection['hip_height_cm'] = hip_height_cm
                detection['hip_height_pixels'] = hip_height_pixels
                detection['ground_level'] = self.ground_level
                
                # Update tracking history
                track_id = f"cattle_{i}"
                if track_id not in self.hip_height_cm:
                    self.hip_height_cm[track_id] = deque(maxlen=10)
                self.hip_height_cm[track_id].append(hip_height_cm)
        
        return detections
    
    def draw_detections_with_apriltag(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw test results (including cattle, AprilTag, and split mask)
        
        Args:
            frame: input frame
            Result: dictionary of test results
            
        Returns:
            A frame of the test results is drawn
        """
        result_frame = frame.copy()
        
        # Draw AprilTags
        for tag in result['apriltags']:
            # Draw AprilTag outline
            corners = tag['corners'].astype(int)
            cv2.polylines(result_frame, [corners], True, (0, 255, 255), 2)
            
            # Draw in AprilTag ID
            center = tag['center'].astype(int)
            cv2.putText(result_frame, f"Tag {tag['id']}", 
                       (center[0] - 20, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Plotting cattle detection results
        detections = result['cattle_detections']
        for i, detection in enumerate(detections):
            # Paint the body area（Blue）
            body_box = detection['body_box']
            cv2.rectangle(result_frame, 
                         (int(body_box[0]), int(body_box[1])), 
                         (int(body_box[2]), int(body_box[3])), 
                         COLORS['body'], 2)
            
            # Draw the hip ROI region（green）- always on the right side
            roi_box = detection['roi_box']
            cv2.rectangle(result_frame, 
                         (int(roi_box[0]), int(roi_box[1])), 
                         (int(roi_box[2]), int(roi_box[3])), 
                         COLORS['roi'], 2)
            
            # Draw the cow segmentation mask（semi-transparent）
            if 'cattle_mask' in detection:
                mask = detection['cattle_mask']
                # Create a Colour Mask
                mask_colored = np.zeros_like(result_frame)
                mask_colored[mask > 0] = [100, 0, 100]  # Purple mask
                # Blend mask into result frame
                result_frame = cv2.addWeighted(result_frame, 0.8, mask_colored, 0.2, 0)
            
            # Draw the hip height point（red）- constrained within the hip box and mask
            hip_top = detection['hip_top']
            # Verify that the hip top is within the hip box
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
            if (roi_x1 <= hip_top[0] <= roi_x2 and roi_y1 <= hip_top[1] <= roi_y2):
                # Draw the hip height point
                cv2.circle(result_frame, hip_top, 6, COLORS['hip_top'], -1)
                cv2.circle(result_frame, hip_top, 10, COLORS['hip_top'], 2)
                
                # Draw a line from the hip top to the top border of the hip box（show constraints）
                cv2.line(result_frame, hip_top, (hip_top[0], int(roi_y1)), (255, 0, 0), 1)
            else:
                # If hip top is not in the hip box，mark it in yellow
                cv2.circle(result_frame, hip_top, 6, (0, 255, 255), -1)
                cv2.circle(result_frame, hip_top, 10, (0, 255, 255), 2)
            
            # Draw hip height information
            if 'hip_height_cm' in detection:
                hip_height = detection['hip_height_cm']
                height_text = f"Hip Height: {hip_height:.1f} cm"
                cv2.putText(result_frame, height_text, 
                           (int(body_box[0]), int(body_box[1]) - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['hip_top'], 2)
            
            # Plotting confidence
            confidence = detection['confidence']
            conf_text = f"Cattle {i+1}: {confidence:.2f}"
            cv2.putText(result_frame, conf_text, 
                       (int(body_box[0]), int(body_box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
            
            # Drawing mask information
            if 'cattle_mask' in detection:
                mask_text = f"Mask: ON"
                cv2.putText(result_frame, mask_text, 
                           (int(body_box[0]), int(body_box[1]) - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 100), 2)

        # Draw the conversion scale information 
        if result['pixels_per_cm'] is not None:
            ratio_text_shadow = f"Pixels/cm: {result['pixels_per_cm']:.3f}"
            cv2.putText(result_frame, ratio_text_shadow, (10, result_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_shadow'], 4)
            
            ratio_text = f"Pixels/cm: {result['pixels_per_cm']:.3f}"
            cv2.putText(result_frame, ratio_text, (10, result_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['debug_text'], 2)
        
        # Draw the ground line (yellow) - only display after the calibration is completed
        if self.calibration_done and self.ground_line_params is not None:
            a, b, c = self.ground_line_params
            h, w = result_frame.shape[:2]
            pts = []
            if abs(b) > 1e-6:
                y_left = int((-a * 0 - c) / b)
                y_right = int((-a * (w - 1) - c) / b)
                pts = [(0, y_left), (w - 1, y_right)]
            elif abs(a) > 1e-6:
                x_val = int((-c) / a)
                pts = [(x_val, 0), (x_val, h - 1)]
            if len(pts) == 2:
                cv2.line(result_frame, pts[0], pts[1], (0, 255, 255), 2)

        # Draw AprilTag information
        tag_text_shadow = f"AprilTags: {len(result['apriltags'])}"
        cv2.putText(result_frame, tag_text_shadow, (10, result_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)

        # Draw AprilTag information
        tag_text = f"AprilTags: {len(result['apriltags'])}"
        cv2.putText(result_frame, tag_text, (10, result_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 3)
        
        return result_frame
    
    def process_stream_with_apriltag(self, cap: cv2.VideoCapture, output_path: str = None, show_preview: bool = True):
        """
        Universal stream processing: 

        Supports video files and cameras
        """
        if not cap.isOpened():
            print("Error: Unable to open video stream")
            return False
        # Attempt to read properties；CAP_PROP_FRAME_COUNT 可may be 0 for webcams
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        print(f"AprilTag cattle detection system start")
        print(f"Video information: {width}x{height}, {fps} FPS, {total_frames if total_frames>0 else 'N/A'} frames")
        print(f"Using model: {self.model.ckpt_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"AprilTag real-wrold distance: {self.tag_distance_real} 米")
        print("-" * 60)

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        
        # --- Cattle tracking for CSV logging ---
        csv_logger = CattleAprilTagCsvLogger()
        cattle_id = 1  # Start with cattle ID 1
        last_cattle_bbox = None  # Previous cattle bounding box
        frames_since_last_detection = 0  # Counter for gap detection
        is_first_cattle = True  # Track if this is the first cattle
        csv_sep_mode = "AUTO"
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Remove automatic first frame calibration; only enter calibration when 'g' is pressed

                # Detect cattle and AprilTag
                result = self.detect_cattle_with_apriltag(frame, frame_count)

                # Draw test results
                result_frame = self.draw_detections_with_apriltag(frame, result)
                cv2.putText(result_frame, "Keys: g=manual groundline (2 clicks) | r=reset groundline | q/ESC=quit",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Frame information text shadow
                denom_total = total_frames if total_frames > 0 else frame_count
                # info_text_shadow = f"Frame: {frame_count}/{denom_total} | Cattle: {len(result['cattle_detections'])}"
                # cv2.putText(result_frame, info_text_shadow, (10, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)

                # Add frame information
                # info_text = f"Frame: {frame_count}/{denom_total} | Cattle: {len(result['cattle_detections'])}"
                # cv2.putText(result_frame, info_text, (10, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['debug_text'], 3)
                
                # FPS text shadow
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(result_frame, fps_text, (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)

                # Add FPS information
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(result_frame, fps_text, (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['debug_text'], 3)

                # Calculation method text shadow
                method_text_shadow = f"Formula: {'BBox' if self.height_method=='bbox' else 'Ground'}"
                cv2.putText(result_frame, method_text_shadow, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)

                # Display the selected height calculation method
                method_text = f"Formula: {'BBox' if self.height_method=='bbox' else 'Ground'}"
                cv2.putText(result_frame, method_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 3)
                
                # Hotkey text shadow
                tool_tips_shadow = f"Keys: g=manual groundline (2 clicks) | r=reset groundline | q/ESC=quit"
                cv2.putText(result_frame, tool_tips_shadow, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)

                # Hotkey Labels
                tool_tips = f"Keys: g=manual groundline (2 clicks) | r=reset groundline | t=CSV toggle | b=CSV manual break | q/ESC=quit"
                cv2.putText(result_frame, tool_tips, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['hotkeys'], 3)
                
                

                # ---- Cattle tracking and CSV logging ----
                current_bbox = None
                current_confidence = None
                current_hip_height = None

                # Get the largest/best detection for tracking
                if result['cattle_detections']:
                    best_detection = max(result['cattle_detections'], 
                                        key=lambda d: (d['body_box'][2] - d['body_box'][0]) * 
                                                     (d['body_box'][3] - d['body_box'][1]))
                    current_bbox = best_detection['body_box']
                    current_confidence = best_detection.get('confidence')
                    current_hip_height = best_detection.get('hip_height_cm')

                new_cattle_detected = False

                # Update frames counter
                if current_bbox is not None:
                    frames_since_last_detection = 0  # Reset counter when cattle detected
                    
                    # Check if this is a new cattle
                    if last_cattle_bbox is not None:
                        # Calculate bbox center movement
                        x1, y1, x2, y2 = current_bbox
                        last_x1, last_y1, last_x2, last_y2 = last_cattle_bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        last_center_x = (last_x1 + last_x2) / 2
                        last_center_y = (last_y1 + last_y2) / 2
                        
                        # Calculate distance moved
                        distance_moved = np.sqrt((center_x - last_center_x)**2 + 
                                                (center_y - last_center_y)**2)
                        
                        # New cattle if moved more than 150 pixels
                        if distance_moved > 150:
                            new_cattle_detected = True
                    else:
                        # First cattle detection
                        if not is_first_cattle:
                            new_cattle_detected = True
                else:
                    # No detection this frame, increment counter
                    frames_since_last_detection += 1

                # Handle new cattle detection
                if new_cattle_detected and csv_sep_mode == "AUTO":
                    cattle_id += 1
                    csv_logger.log_separator("AUTO")
                    print(f"[CSV] New cattle detected: ID {cattle_id}")

                # Log measurement if there is valid data
                if current_bbox is not None:
                    csv_logger.log(
                        cattle_id=cattle_id,
                        hip_height_cm=current_hip_height,
                        confidence=current_confidence,
                        frame_id=frame_count,
                        bbox=current_bbox
                    )

                # Update tracking variables
                if current_bbox is not None:
                    last_cattle_bbox = current_bbox
                    is_first_cattle = False

                if out:
                    out.write(result_frame)

                if show_preview:
                    window_name = 'AprilTag Cattle Detection'
                    cv2.imshow(window_name, result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key in (ord('g'), ord('G')):
                        # Reset the current ground line and start interactive calibration
                        self.ground_line_params = None
                        self.ground_line_points = []
                        self.ground_level = None
                        self.calibration_done = False
                        self._run_interactive_calibration(frame, window_name)
                    if key in (ord('t'), ord('T')):
                        csv_sep_mode = "MANUAL" if csv_sep_mode == "AUTO" else "AUTO"
                        print(f"[CSV] CSV separator mode: {csv_sep_mode}")

                    if key in (ord('b'), ord('B')):
                        if csv_sep_mode == "MANUAL":
                            csv_logger.log_separator("MANUAL")
                            cattle_id += 1
                            last_cattle_bbox = None
                            frames_since_last_detection = 0
                            print(f"[CSV] Manual break: New cattle ID {cattle_id}")

                if frame_count % 30 == 0:
                    print(f"Progress: {frame_count}/{denom_total} ({(frame_count/max(1,denom_total))*100:.1f}%) - {current_fps:.1f} FPS")
                    if result['pixels_per_cm'] is not None:
                        print(f"  Pixel/cm conversion ratio: {result['pixels_per_cm']:.3f}")
                    if result['cattle_detections']:
                        for i, det in enumerate(result['cattle_detections']):
                            if 'hip_height_cm' in det:
                                print(f"  Cattle {i+1} Hip Height: {det['hip_height_cm']:.1f} cm")
        finally:
            cap.release()
            if out:
                out.release()
            
            csv_logger.close()
            
            cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = frame_count / max(1.0, total_time)
            print(f"\nAprilTag cattle analysis complete!")
            print(f"Total frames: {frame_count}")
            print(f"Total time: {total_time:.2f} 秒")
            print(f"AverageFPS: {avg_fps:.2f}")
            if 'result' in locals() and result.get('pixels_per_cm') is not None:
                print(f"Final pixels/cm conversion ratio: {result['pixels_per_cm']:.3f}")
            if output_path:
                print(f"Output video: {output_path}")

    def process_video_with_apriltag(self, video_path: str, output_path: str = None, show_preview: bool = True):
        """Process a video file"""
        cap = cv2.VideoCapture(video_path)
        return self.process_stream_with_apriltag(cap, output_path, show_preview)

    def process_camera_with_apriltag(self, camera_index: int = 0, output_path: str = None, show_preview: bool = True):
        """Process webcam input (including OAK-D Pro in UVC mode)"""
        cap = cv2.VideoCapture(camera_index)
        return self.process_stream_with_apriltag(cap, output_path, show_preview)
    
    def process_oak_with_apriltag(self, output_path: str = None, show_preview: bool = True):
        """Process OAK-D camera (via DepthAi pipline)"""
        import depthai as dai
        print("[INFO] Starting OAK-D camera stream via DepthAI pipeline...")

        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(25)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam_rgb.video.link(xout.input)

        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            frame_count = 0
            start_time = time.time()

            csv_logger = CattleAprilTagCsvLogger()
            cattle_id = 1
            last_cattle_bbox = None
            frames_since_last_detection = 0
            is_first_cattle = True
            csv_sep_mode = "AUTO"

            window_name = "AprilTag Cattle Detection"
            while True:
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                frame_count += 1
                # Detect cattle and AprilTag
                result = self.detect_cattle_with_apriltag(frame, frame_count)

                # Draw detections
                result_frame = self.draw_detections_with_apriltag(frame, result)

                # --- HUD overlays to match video/camera mode ---
                # FPS (shadow + text)
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(result_frame, fps_text, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)
                cv2.putText(result_frame, fps_text, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['debug_text'], 3)

                # Height formula (shadow + text)
                method_text = f"Formula: {'BBox' if self.height_method=='bbox' else 'Ground'}"
                cv2.putText(result_frame, method_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)
                cv2.putText(result_frame, method_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text'], 3)

                # Hotkeys (shadow + text)
                tool_tips = "Keys: g=groundline | r=reset | t=CSV toggle | b=CSV break | q/ESC=quit"
                cv2.putText(result_frame, tool_tips, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['text_shadow'], 6)
                cv2.putText(result_frame, tool_tips, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['hotkeys'], 3)

                if show_preview:
                    cv2.imshow(window_name, result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # q or ESC
                        break
                    if key in (ord('g'), ord('G')):
                        # reset existing ground line and enter interactive calibration
                        self.ground_line_params = None
                        self.ground_line_points = []
                        self.ground_level = None
                        self.calibration_done = False
                        self._run_interactive_calibration(frame, window_name)
                    if key in (ord('r'), ord('R')):
                        # reset ground line to default (80% height)
                        self.ground_line_params = None
                        self.ground_line_points = []
                        self.ground_level = None
                        self.calibration_done = False
                    if key in (ord('t'), ord('T')):
                        csv_sep_mode = "MANUAL" if csv_sep_mode == "AUTO" else "AUTO"
                        print(f"[CSV] CSV separator mode: {csv_sep_mode}")

                    if key in (ord('b'), ord('B')):
                        if csv_sep_mode == "MANUAL":
                            csv_logger.log_separator("MANUAL")
                            cattle_id += 1
                            last_cattle_bbox = None
                            frames_since_last_detection = 0
                            print(f"[CSV] Manual break: New cattle ID {cattle_id}")

                # ---- Cattle tracking and CSV logging ----
                current_bbox = None
                current_confidence = None
                current_hip_height = None

                # Get the largest/best detection for tracking
                if result['cattle_detections']:
                    best_detection = max(result['cattle_detections'], 
                                        key=lambda d: (d['body_box'][2] - d['body_box'][0]) * 
                                                     (d['body_box'][3] - d['body_box'][1]))
                    current_bbox = best_detection['body_box']
                    current_confidence = best_detection.get('confidence')
                    current_hip_height = best_detection.get('hip_height_cm')

                new_cattle_detected = False

                # Update frames counter
                if current_bbox is not None:
                    frames_since_last_detection = 0  # Reset counter when cattle detected
                    
                    # Check if this is a new cattle
                    if last_cattle_bbox is not None:
                        # Calculate bbox center movement
                        x1, y1, x2, y2 = current_bbox
                        last_x1, last_y1, last_x2, last_y2 = last_cattle_bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        last_center_x = (last_x1 + last_x2) / 2
                        last_center_y = (last_y1 + last_y2) / 2
                        
                        # Calculate distance moved
                        distance_moved = np.sqrt((center_x - last_center_x)**2 + 
                                                (center_y - last_center_y)**2)
                        
                        # New cattle if moved more than 150 pixels
                        if distance_moved > 150:
                            new_cattle_detected = True
                    else:
                        # First cattle detection
                        if not is_first_cattle:
                            new_cattle_detected = True
                else:
                    # No detection this frame, increment counter
                    frames_since_last_detection += 1

                # Handle new cattle detection
                if new_cattle_detected and csv_sep_mode == "AUTO":
                    cattle_id += 1
                    csv_logger.log_separator("AUTO")
                    print(f"[CSV] New cattle detected: ID {cattle_id}")

                # Log measurement if there is valid data
                if current_bbox is not None:
                    csv_logger.log(
                        cattle_id=cattle_id,
                        hip_height_cm=current_hip_height,
                        confidence=current_confidence,
                        frame_id=frame_count,
                        bbox=current_bbox
                    )

                # Update tracking variables
                if current_bbox is not None:
                    last_cattle_bbox = current_bbox
                    is_first_cattle = False

                # Progress reporting
                if frame_count % 30 == 0:
                    print(f"处理进度: {frame_count} 帧 - {current_fps:.1f} FPS")
                    if result['pixels_per_cm'] is not None:
                        print(f"  像素/厘米转换比例: {result['pixels_per_cm']:.3f}")
                    if result['cattle_detections']:
                        for i, det in enumerate(result['cattle_detections']):
                            if 'hip_height_cm' in det:
                                print(f"  牛 {i+1} Hip Height: {det['hip_height_cm']:.1f} cm")

        # Cleanup
        csv_logger.close()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / max(1.0, total_time)
        print(f"\nOAK-D AprilTag cattle inspection completed!")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} second(s)")
        print(f"Average FPS: {avg_fps:.2f}")
        if 'result' in locals() and result.get('pixels_per_cm') is not None:
            print(f"Final pixel/cm conversion ratio: {result['pixels_per_cm']:.3f}")


def main():
    """Main function-demonstration of AprilTag cattle inspection system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AprilTag Cattle Detection System')
    parser.add_argument('--video', '-v', type=str, required=False,
                   help='Input video file path (can be omitted if using --oak or --camera')
    parser.add_argument('--output', '-o', type=str,
                       help='Output video file path')
    parser.add_argument('--model', '-m', type=str, default='yolo11n.pt',
                       help='path to YOLO model')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--tag-distance', '-d', type=float, default=1.0,
                       help='Real-world distance between AprilTags (meters)')
    parser.add_argument('--no-calibration', action='store_true',
                       help='Skip interactive ground line calibration and use default ground (80% of height)')
    parser.add_argument('--camera', action='store_true',
                       help='Use webcam as input source (Including OAK-D Pro in UVC mode)')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index (default = 0)')
    parser.add_argument('--oak', action='store_true',
                   help='Use OAK-D camera (DepthAI pipeline)')
    parser.add_argument('--height-method', choices=['ground', 'bbox'], default='ground',
                       help='Hip Height calculation method: ground=to the ground line, bbox=to the bottom of the bounding box')
    
    args = parser.parse_args()
    
    # Create AprilTag cattle detector
    detector = AprilTagCattleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Set real-world AprilTag distance
    detector.tag_distance_real = args.tag_distance
    detector.calibrate_ground = (not args.no_calibration)
    detector.height_method = args.height_method
    
    # Process video or camera
    if args.oak:
        detector.process_oak_with_apriltag(args.output)
    elif args.camera:
        detector.process_camera_with_apriltag(args.camera_index, args.output)
    else:
        detector.process_video_with_apriltag(args.video, args.output)


if __name__ == "__main__":
    main()

"""
Cattle Hip Height detection System Configuration File
"""

# YOLO model configurations
YOLO_CONFIG = {
    "model_path": "yolo11n.pt",  # can be switched out with  yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100
}

# Detection parameter configurations
DETECTION_CONFIG = {
    "roi_ratio": 0.3,  # Ratio of hip area to body area (left-right split)
    "hip_search_radius": 20,  # Hip height point search radius
    "min_contour_area": 100,  # Minimum area a hip region must have to be considered valid
    "morphology_kernel_size": 3,  # how much image is cleaned/smoothed during morphological opeartions
}

# Colour configurations (BGR format)
COLORS = {
    "roi": (0, 255, 0),      # Green - Hip ROI
    "body": (255, 0, 0),     # Blue - Body region  
    "hip_top": (0, 0, 255),  # Red - Hip height
    "text": (255, 255, 255), # White - Text
    "background": (0, 0, 0),  # Black - Background
    "debug_text": (245, 243, 113),
    "hotkeys": (171, 203, 232),
    "text_shadow": (0, 0, 0)
}

# Video Processing configurations
VIDEO_CONFIG = {
    "output_codec": "mp4v",
    "output_extension": ".mp4",
    "preview_window_name": "Cattle Detection",
    "show_fps": True,
    "show_frame_count": True,
    "show_detection_count": True
}

# Performance optimisation configurations
PERFORMANCE_CONFIG = {
    "use_gpu": True,  # GPU acceleration
    "batch_size": 1,  # Batch size (number of frames fed into model simultaneously)
    "max_resolution": 1280,  # Maximum processing resolution
    "skip_frames": 0,  # Frame skips（0 = No frame skipping）
}

# Robust configurations
ROBUSTNESS_CONFIG = {
    "multi_method_detection": True,  # Detecting Hip Height point using multiple methods
    "temporal_smoothing": True,  # Stablise hip height over multiple frames to reduce jitter
    "outlier_rejection": True,  # Outlier rejection
    "confidence_weighting": True,  # Confidence Weighting
}

# Real-time processing configurations
REALTIME_CONFIG = {
    "camera_id": 0,  # Default camera ID
    "frame_queue_size": 10,  # Frame queue size
    "result_queue_size": 10,  # Result queue size
    "show_fps": True,  # Show FPS
    "show_detection_info": True,  # Display detection information
    "show_hip_coordinates": True,  # Display hip height point coordinates
    "save_frames": False,  # Whether to save frames
    "realtime_playback": True,  # Real-time playback (no frame skips)
}

# This Python file uses the following encoding: utf-8
import sys
import os
import time
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread
import csv
from datetime import datetime
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QUrl, Qt, QTimer
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                                QHBoxLayout, QWidget, QPushButton,
                                QSlider, QStyle, QLabel, QMessageBox)

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cli.hip_height_from_video import process_frame
from src.external.sort import Sort

class ThreadedVideoWriter(Thread):
    def __init__(self, output_path, fourcc, fps, frame_size):
        super().__init__()
        self.output_path = output_path
        self.frame_queue = Queue(maxsize=128)  # Buffer 128 frames max
        self.running = True
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
    def run(self):
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1.0)  # 1 second timeout
                if frame is not None:
                    self.writer.write(frame)
                self.frame_queue.task_done()
            except Empty:
                continue
                
    def write(self, frame):
        if self.running:
            try:
                self.frame_queue.put(frame, block=False)  # Non-blocking put
            except Full:
                pass  # Drop frame if queue is full
                
    def stop(self):
        self.running = False
        self.join()  # Wait for remaining frames
        self.writer.release()

class VideoWindow(QMainWindow):
    """
    Main window for the Hip Height Measuring App.
    Contains video player, FPS counter, "camera control buttons", and export to csv functionalities.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hip Height Measuring App")
        self.resize(1105, 717)

        # FPS Counter
        self.frame_count = 0

        # Initialise videoplayer
        # self.media_player = QMediaPlayer()

        # Timer set up
        self.elapsed_time = 0

        # Setup ui and signals
        self.ui_setup()
        self.create_layouts()
        self.connect_signals()
        # self.load_video()

        # cv2 video capture
        self.cap = None

        self.det_model = YOLO("yolo11n-seg.pt")
        # the rows of the csv file
        self.measurements = []
        self.frame_idx = 0
        
        # init the cow id tracker
        self.tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)


    def ui_setup(self):
        """
        Initialise all widgets for the UI.
        """
        # Placeholder videoplayer
        # self.video_widget = QVideoWidget()
        # self.media_player.setVideoOutput(self.video_widget)

        # trying new videoplayer
        self.video_widget = QLabel("Loading a video or starting the camera")
        self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_widget.setMinimumSize(640, 480)

        # Placeholder cow label
        self.cow_label = QLabel("Cow X")
        self.cow_label.setStyleSheet("font-weight: bold;")
        self.cow_label.setFixedHeight(20)

        # Placeholder height label
        self.height_label = QLabel("Height: --- cm")
        self.height_label.setFixedHeight(14)
        self.height_label.setFixedWidth(95)

        # Play/pause button
        self.play_pause_button = QPushButton()
        self.play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_pause_button.setIcon(self.play_icon)

        # Slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)

        # FPS counter
        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(1000)
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-weight: bold;")
        self.fps_label.setFixedHeight(self.play_pause_button.sizeHint().height())

        # Timer + export setup
        self.timer = QTimer(self)
        self.timer.setInterval(10)

        self.timer_start_button = QPushButton("Camera On")
        self.stop_button = QPushButton("Camera Off")
        self.timer_label = QLabel("00:00:00.000")
        self.timer_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.timer_label.setFixedHeight(self.stop_button.sizeHint().height())
        self.reset_button = QPushButton("Reset Timer")
        self.export_button = QPushButton("Export")

        # Timer font formatting
        self.setup_timer_font()

    def setup_timer_font(self):
        """
        Configure font for timer so it doesn't jitter.
        """
        # Font config
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(14)
        self.timer_label.setFont(font)

        font_metrics = self.timer_label.fontMetrics()
        text_width = font_metrics.horizontalAdvance("00:00:00.000")
        self.timer_label.setMinimumWidth(text_width + 5)

        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def create_layouts(self):
        """
        Create layouts for UI widgets.
        """
        # Horizontal layouts
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(self.fps_label)

        cow_layout = QVBoxLayout()
        cow_layout.addWidget(self.cow_label)
        cow_layout.addWidget(self.height_label)
        cow_layout.addStretch()

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_widget, stretch=5)
        video_layout.addLayout(cow_layout, stretch=0)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.progress_slider)

        timer_layout = QHBoxLayout()
        timer_layout.addWidget(self.timer_start_button)
        timer_layout.addWidget(self.stop_button)
        timer_layout.addWidget(self.timer_label)
        timer_layout.addWidget(self.reset_button)
        timer_layout.addStretch()
        timer_layout.addWidget(self.export_button)

        # Vertical layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(fps_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(timer_layout)

        # Set central widget (main body of window)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def connect_signals(self):
        """
        Connects widget signals to their methods.
        """
        # FPS signal
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start()
        # self.media_player.videoSink().videoFrameChanged.connect(self.count_frame)

        # Videoplayer control signals
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        # self.progress_slider.sliderMoved.connect(self.set_player_position)

        # # Connect videoplayer signals to UI
        # self.media_player.playbackStateChanged.connect(self.update_button_icon)
        # self.media_player.positionChanged.connect(self.update_slider_position)
        # self.media_player.durationChanged.connect(self.set_slider_range)
        # self.media_player.mediaStatusChanged.connect(self.show_first_frame)

        # Timer signals
        # self.timer_start_button.clicked.connect(self.toggle_timer)
        # self.stop_button.clicked.connect(self.stop_timer)
        # self.timer.timeout.connect(self.update_timer_display)
        # self.reset_button.clicked.connect(self.reset_timer)

        # changing timer signals to be wired to OpenCV
        self.timer_start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)

        self.timer.timeout.connect(self.update_timer_display)
        self.reset_button.clicked.connect(self.reset_timer)

        # Export signals
        self.export_button.clicked.connect(self.export_to_csv)
        self.export_button.clicked.connect(self.stop_timer)

    # def load_video(self):
    #     """
    #     Load the video file - later to be camera stream.
    #     """
    #     # Placeholder for video input
    #     video_file = Path(__file__).parent / "videos/output.mp4"
    #     if video_file.exists():
    #         self.media_player.setSource(QUrl.fromLocalFile(str(video_file)))
    #     else:
    #         print("Error: placeholder.mp4 not found")

    def show_first_frame(self, status):
        """
        Load video thumbnail on startup.
        """
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.media_player.pause()

    def toggle_play_pause(self):
        """
        Play/pause the video.
        """
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def update_button_icon(self, state):
        """
        Update icon for play/pause button
        """
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setIcon(self.pause_icon)
        else:
            self.play_pause_button.setIcon(self.play_icon)

    def set_slider_range(self, duration):
        """
        Set range of slider according to the video duration.
        """
        self.progress_slider.setRange(0, duration)

    def update_slider_position(self, position):
        """
        Update slider handle as the video plays.
        """
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(position)
        self.progress_slider.blockSignals(False)

    def set_player_position(self, position):
        """
        Goes to the position in the video where the user has chosen.
        """
        self.media_player.setPosition(position)

    def toggle_timer(self):
        """
        Starts the timer at 0 if it is not already running.
        """
        if not self.timer.isActive():
            self.media_player.setVideoOutput(self.video_widget)
            self.elapsed_time = 0
            self.timer_label.setText("00:00:00.000")
            self.timer.start()

    # Timer
    def update_timer_display(self):
        """
        Update the timer label with the elapsed_time.
        """
        self.elapsed_time += self.timer.interval()

        total_seconds = self.elapsed_time // 1000

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = self.elapsed_time % 1000

        time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        self.timer_label.setText(time_string)

    def stop_timer(self):
        """
        Stop the timer.
        """
        self.timer.stop()

    def reset_timer(self):
        """
        Reset the timer to 0, only if it is already stopped.
        """
        if not self.timer.isActive():
            self.elapsed_time = 0
            self.timer_label.setText("00:00:00.000")

    def count_frame(self, frame):
        """
        Update the frame_count by 1 for each time the frame changes in a second.
        """
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.frame_count += 1

    def update_fps_display(self):
        """
        Update the FPS label to reflect the number of frames counted in the last second.
        """
        self.fps_label.setText(f"FPS: {self.frame_count}")
        self.frame_count = 0

    def export_to_csv(self):
        """
        Export placeholder csv file to Downloads folder.
        """
        self.stop_timer()
        try:
            downloads_path = Path.home()/"Downloads"
            downloads_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"export_{timestamp}.csv"
            file_path = downloads_path/file_name

            # CSV header
            header = ["Time", "Hip Height", "Median", "..."]
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setText("Export Successful!")
            msg_box.setInformativeText(f"File saving to:\n{file_path}")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()

        except Exception as e:
            error_box = QMessageBox(self)
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setText("Export Failed")
            error_box.setInformativeText(f"An error occurred:\n{e}")
            error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_box.exec()

    def start_processing(self):
        # start reading grames in with OpenCV and process with the backend yolo model

        # video demo or live camera

        # (live camera)
        # video_source = 0
        # self.cap = cv2.VideoCapture(video_source, cv2.CAP_AVFOUNDATION)

        # sample file
        video_path = ROOT / "videos" / "output.mp4"
        if not video_path.exists():
            print(f"Error: file not found: {video_path}")
            self.height_label.setText(f"Error: file not found: {video_path}")
            return

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            self.fps_label.setText("FPS: 0")
            self.height_label.setText("Error: cannot open video/camera")
            return

        # timeout every 33 milliseconds so every 16ms, next frame is called
        self.timer.setInterval(33)

        try:
            self.timer.timeout.disconnect(self.next_frame)
        except (TypeError, RuntimeError):
            pass

        self.timer.timeout.connect(self.next_frame)
        self.timer.start()

        # Initialize video writer for saving processed frames
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Save the video to saved_videos directory
        videos_dir = ROOT / "saved_videos"  
        videos_dir.mkdir(exist_ok=True)

        # Storage limit for saved videos == 200GB ==
        MAX_STORAGE_GB = 200
        BYTES_IN_GB = 1024 * 1024 * 1024
        
        total_size = 0
        video_files = []
        for f in videos_dir.glob("*.mp4"):
            total_size += f.stat().st_size
            video_files.append((f.stat().st_mtime, f))
            
        while total_size >= MAX_STORAGE_GB * BYTES_IN_GB and video_files:
            video_files.sort(key=lambda x: x[0])  
            oldest_file = video_files[0][1]
            print(f"Removing old video to free space: {oldest_file.name}")
            total_size -= oldest_file.stat().st_size
            oldest_file.unlink()
            video_files.pop(0)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(videos_dir / f"output_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = ThreadedVideoWriter(output_path, fourcc, fps, (width, height))
        self.video_writer.start()
        print(f"Saving video to: {output_path}")

        self.frame_idx = 0
        self.measurements.clear()
        self.fps_timer.start()

    def stop_processing(self):
        """Stop processing and release resources"""
        self.timer.stop()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Stop threaded video writer and wait for remaining frames
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.stop()  # This will join thread and release writer
            print("Video writer stopped and saved")
            self.video_writer = None
            
        self.fps_label.setText("FPS: 0")
        self.video_widget.clear()  

    def next_frame(self):
        # take a frame and put it into the backend
        if self.cap is None:
            return
        
        # ok is if the frame is read in
        # frame is the actual image data (Numpy array)
        ok, frame = self.cap.read()
        if not ok:
            self.stop_processing()
            return
        
        # fps coutner
        self.frame_idx += 1
        self.frame_count += 1 

        # backend processing the OpenCV frame

        results = process_frame(frame, self.det_model, self.tracker, conf_thresh=0.4)
       
        # if there are results
        # update labels
        if results["detected"]:
            id = results.get("id")
            if id is None:
                self.cow_label.setText(f"Cow X")
            else:
                self.cow_label.setText(f"Cow {id}")
                
            hip_y = results.get("hip_y")
            ground_y = results.get("ground_y")
            if hip_y is not None and ground_y is not None:
                height_px = max(0, ground_y - hip_y)
                # show on GUI
                self.height_label.setText(f"Height: {height_px} px")
            else:
                # detected cow but didnt compute height
                self.height_label.setText("Height: not detected")
            video = results.get("visualization", frame)
        else:
            self.height_label.setText("No cow detected")
            video = frame

        # Save video frame
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.write(video)

        # display the image in GUI
        # opencv uses BGR, QT expects RGB so convert
        rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_widget.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """Handle cleanup when application closes"""
        # Stop processing and release resources
        self.stop_processing()
        
        # Extra safety check for video writer
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.stop()  # This ensures thread is joined
            print("Final video writer cleanup on close")
            self.video_writer = None
            
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()

    sys.exit(app.exec())

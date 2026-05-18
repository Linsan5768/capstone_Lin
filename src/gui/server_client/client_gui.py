import socket
import struct  # To unpack the header
import sys
from typing_extensions import override

import cv2
import numpy as np
import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QCloseEvent
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

# Configuration variables (don't change except API URL when necessary)
LINUX_API_URL = "http://192.168.1.10:8000" # todo: set for prod

UDP_PORT = 9999 # we're using UDP since its faster. Don't use this to transfer measurement data
UDP_IP = "0.0.0.0" # specifies that we should accept UDP data from any connection

BUFFER_SIZE = 65536


class VideoReceiverThread(QThread):
    """
    Worker thread for receiving and reassembling UDP chunks.
    There isn't really any reason to change the implementation of this class unless
    you want to improve the frame emmision logic
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # AF_INET = internet socket (IP), SOCK_DGRAM = UDP socket
        self.sock.bind((UDP_IP, UDP_PORT)) # tell OS to reroute data arriving at port 9999 to the GUI
        self.sock.settimeout(1.0)
        self.frame_buffer = {}  # A dictionary to store chunks for each frame, keeps last 5

    def run(self):
        """
        Maintains the frame buffer and emits processed frames to the GUI
        """
        while self._run_flag:
            try:
                packet, addr = self.sock.recvfrom(BUFFER_SIZE) # blocking btw

                # Unpack header (3, 4-byte integers)
                header_size = 12
                header_data = packet[:header_size]
                chunk_data = packet[header_size:]

                frame_id, total_chunks, chunk_index = struct.unpack("III", header_data)

                # Create slotes for chunks in the dictionary
                if frame_id not in self.frame_buffer:
                    self.frame_buffer[frame_id] = [None] * total_chunks

                # Store the chunk
                if chunk_index < total_chunks:
                    self.frame_buffer[frame_id][chunk_index] = chunk_data

                # note: there may be a memory leak somewhere below this point

                # If all chunk slots are filled
                if not any(c is None for c in self.frame_buffer[frame_id]):
                    # Reassemble the frame from chunks byte-by-byte
                    full_frame_data = b"".join(self.frame_buffer[frame_id])

                    # Decode the JPEG frame
                    nparr = np.frombuffer(full_frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        self.change_pixmap_signal.emit(frame) # NOTE: sends frame as signal to GUI

                    # Free fram (otherwise all chunks stay in memory)
                    del self.frame_buffer[frame_id]

                # Remove oldest frame to prevent buffer from growing indefinitely
                # could also remove this if we want to keep the video stream
                # but probably easier to just get the video as a blob once we're done streaming
                if len(self.frame_buffer) > 5:
                    oldest_frame = min(self.frame_buffer.keys())
                    del self.frame_buffer[oldest_frame]

            except socket.timeout:
                continue  # Allows checking the _run_flag again
            except Exception as e:
                print(f"Error processing packet in VideoRecieverThread:\n{e}")

    def stop(self):
        """
        Called by the GUI when the thread should be stopped
        """
        self._run_flag = False
        self.wait()
        self.sock.close()


class VideoClientGUI(QWidget):
    """
    Class representing the GUI app, has buttons etc and function to update each video frame
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Stream Client")
        self.video_thread = None

        # --- Widgets ---
        self.video_label = QLabel("Press 'Start Stream' to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.stop_button.setEnabled(False)

        # --- Layout ---
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        # --- Signals and Slots ---
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

    def start_stream(self):
        """
        Makes API call to server to start the video stream and starts thread to start processsing it
        """

        try:
            # Get and validate response
            response = requests.post(f"{LINUX_API_URL}/start_stream")
            response.raise_for_status()  # Raise an exception for bad status codes
            print(response.json())

            # Thread logic
            self.video_thread = VideoReceiverThread()
            self.video_thread.change_pixmap_signal.connect(self.update_image) # subscribe to worker's signal; namely the frame
            self.video_thread.start()

            # Button logic
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.video_label.setText("Waiting for video stream...")

        except requests.exceptions.RequestException as e:
            self.video_label.setText(f"Error starting stream: {e}")
            print(f"API Error: {e}")

    def stop_stream(self):
        """
        Makes API call to server to stop the video stream and stops thread processing it
        """

        try:
            response = requests.post(f"{LINUX_API_URL}/stop_stream")
            response.raise_for_status()
            print(response.json())

        except requests.exceptions.RequestException as e:
            # you should still stop the thread even if the API request failed
            self.video_label.setText(f"API Error on stop: {e}")
            print(f"API Error: {e}")

        finally:
            if self.video_thread:
                self.video_thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.video_label.setText("Stream stopped.")

    def update_image(self, cv_img):
        """
        Updates the video_label with a new opencv image. Called when a signal from VideoRecieverThread is emitted.
        Has no None input handling

        Args:
            cv_img: A Matlike object representing the frame to be displayed
        """
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap)

    @override
    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Called when attempting to close the window. Stops stream on window close"""

        # a0 is the event. I have to name it that way because this is actually an override function
        if self.stop_button.isEnabled():
            self.stop_stream()

        if a0 is not None:
            a0.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoClientGUI()
    window.show()
    sys.exit(app.exec())

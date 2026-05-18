import threading
import cv2
import socket
import time
import struct # To pack integers into bytes
from fastapi import FastAPI, HTTPException

WINDOWS_COMPUTER_IP = '192.168.1.10' # ip of computer to send messages to
PORT = 9999 # should be the same port as the port in client_gui. Please move this to a config.py so we can set universally
JPEG_QUALITY = 80 # 80%
TARGET_FPS = 30  # Target frame rate (should probably be around 24? depends on camera)
CHUNK_SIZE = 4096 # Size of each chunk in bytes

class StreamManager:
    """
    Class representing the server running on the Jetson Nano.
    Handles video streaming, and will also be responsible for calling the algorithm functions to annotate the video stream.

    There should be no reason to change this class unless improving streaming logic, changing quality, or adding in the
    ML function call.
    """
    def __init__(self):
        self.streaming_active = threading.Event()
        self.streaming_thread = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def video_streaming_task(self):
        """Captures video, annotates, chunks, and sends frames via UDP."""

        print("I am going to connect to the camera!!!")
        cap = cv2.VideoCapture(1)

        print("The camera has been started!!!")
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        print("Starting video stream...")
        frame_id = 0
        while self.streaming_active.is_set():

            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # PLACEHOLDER FOR ML ANNOTATION. REPLACE WITH CALL TO ANNOTATION ALGORITHM
            # will probably look like frame = processframe(frame)
            # when the video stream is stopped, the processing will not happen
            # -----------------------------------------
            cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
            cv2.putText(frame, "ML Annotation", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # -----------------------------------------

            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                continue

            data = encoded_frame.tobytes()
            total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

            # Send the frame in chunks
            for i in range(total_chunks):
                chunk = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]

                # This should correlate with the recipient logic in client_gui
                header = struct.pack('III', frame_id, total_chunks, i)

                try:
                    self.sock.sendto(header + chunk, (WINDOWS_COMPUTER_IP, PORT))
                except Exception as e:
                    print(f"Error sending chunk: {e}")
                    break

            frame_id += 1

            # --- FPS CONTROL LOGIC ---
            # this is kinda shoddy but it does work :)
            #
            # without this the while loop sends all the frame data at once (basically plays the video as fast as possible)
            # leads to errors in the GUI
            elapsed_time = time.time() - frame_start_time
            sleep_time = (1.0 / TARGET_FPS) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # outside of while loop
        cap.release()
        print("Streaming stopped and resources released.")

    def start(self):
        """Starts video stereaming and processing"""
        if not self.streaming_thread or not self.streaming_thread.is_alive():
            self.streaming_active.set()
            self.streaming_thread = threading.Thread(target=self.video_streaming_task)
            self.streaming_thread.start()
            return {"message": "Video stream started."}
        else:
            raise HTTPException(status_code=400, detail="Stream is already running.")

    def stop(self):
        """Stops video stereaming and processing"""
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_active.clear()
            self.streaming_thread.join()
            self.streaming_thread = None
            return {"message": "Video stream stopped."}
        else:
            raise HTTPException(status_code=400, detail="Stream is not currently running.")

# TO RUN: uvicorn stream_server:app --host 0.0.0.0 --port 8000
# note: you can't put this in a if __name__ == __main__ statement because then uvicorn can't find app

app = FastAPI()
stream_manager = StreamManager()

@app.post("/start_stream")
async def start_stream_endpoint():
    return stream_manager.start()

@app.post("/stop_stream")
async def stop_stream_endpoint():
    return stream_manager.stop()

# add more endpoints here for additional functions
# like get status (server offline / online)
# get measurements from a cow (the entire csv thing sent over, probably as a json)
# etc.

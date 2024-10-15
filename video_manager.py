import cv2
from PIL import Image, ImageTk
import tkinter as tk
import os
import threading
import time
import pickle
import re
from datetime import datetime, timedelta


class VideoManager:
    def __init__(self, log_func):
        self.log = log_func
        self.cap = None
        self.total_frames = 0
        self.is_playing = False
        self.label = None
        self.scrollbar = None
        self.display_tracking = False
        self.mosquito_tracks = None

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            self.log(f"Error opening video file: {video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 1:
            self.log(f"Error: No frames found in the video {video_path}")
            return False

        return True

    def set_display_label(self, label):
        self.label = label

    def set_scrollbar(self, scrollbar):
        self.scrollbar = scrollbar

    def set_display_tracking(self, display_tracking):
        self.display_tracking = display_tracking

    def load_tracking_data(self, tracking_file):
        with open(tracking_file, 'rb') as f:
            self.mosquito_tracks = pickle.load(f)

    def play_video(self, time_btw_frames=0.04):
        if self.is_playing:
            self.log("Video is already playing.")
            return
        self.is_playing = True
        threading.Thread(target=self._play_video, args=(time_btw_frames,)).start()

    def _play_video(self, time_btw_frames):
        while self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                break
            if self.display_tracking and self.mosquito_tracks:
                frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame = self.overlay_tracking(frame, frame_idx)
            self._display_frame(frame)
            frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.update_scrollbar(frame_index)
            time.sleep(time_btw_frames)

    def pause_video(self):
        self.is_playing = False

    def show_frame(self, frame_index):
        if self.cap is None or not self.cap.isOpened():
            self.log("No video loaded or video capture is not opened.")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            if self.display_tracking and self.mosquito_tracks:
                frame = self.overlay_tracking(frame, frame_index)
            self._display_frame(frame)

    def _display_frame(self, frame):
        if not self.label or frame is None:
            self.log("Frame or video display label is not available.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        frame_rgb_resized = self.resize_image(frame_rgb, label_width, label_height)
        img = Image.fromarray(frame_rgb_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def resize_image(self, img, width, height):
        aspect_ratio = img.shape[1] / img.shape[0]
        if width / height > aspect_ratio:
            width = int(height * aspect_ratio)
        else:
            height = int(width / aspect_ratio)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def overlay_tracking(self, frame, frame_idx):
        # Drawing overlay logic goes here
        for id, data in self.mosquito_tracks.objects.items():
            t_start, t_end = data["start"], data["end"]
            if t_start <= frame_idx <= t_end:
                t_relative = frame_idx - t_start+2
                try:
                    centroid = data["coordinates"][t_relative]
                    state = data["state"][t_relative]
                    color = (0, 0, 255) if state == 0 else (255, 0, 0)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, color, 1)
                except Exception as e:
                    continue
                    #print(e)
                    #self.log(f"Error: {e}")
        return frame

    def next_frame(self):
        if not self.is_playing and self.cap and self.cap.isOpened():
            current_frame = self.scrollbar.get()
            if current_frame < self.total_frames - 1:
                self.show_frame(current_frame + 1)
                self.update_scrollbar(current_frame + 1)

    def previous_frame(self):
        if not self.is_playing and self.cap and self.cap.isOpened():
            current_frame = self.scrollbar.get()
            if current_frame > 0:
                self.show_frame(current_frame - 1)
                self.update_scrollbar(current_frame - 1)

    def update_scrollbar(self, frame_index):
        self.scrollbar.set(frame_index)
    
    def get_total_frames(self):
        return self.total_frames

    def get_datetime_from_file_name(self, video_name):
            try:
                # General regex pattern for extracting date and time and video number
                if video_name.endswith(".mp4"):
                    match = re.match(r'.*_(\d{6})_[^_]*_(\d{6})_v(\d+)\.mp4', video_name)
                else:
                    video_name = video_name+".mp4"
                    match = re.match(r'.*_(\d{6})_[^_]*_(\d{6})_v(\d+)\.mp4', video_name)

                if not match:
                    raise ValueError("No matching pattern found")

                date_part = match.group(1)  # YYMMDD
                time_part = match.group(2)  # HHMMSS
                video_number = int(match.group(3))  # Video number as an integer

                # Parse the date and time components
                YY = int(date_part[:2])
                MM = int(date_part[2:4])
                DD = int(date_part[4:6])
                HH = int(time_part[:2])
                MI = int(time_part[2:4])
                SS = int(time_part[4:6])

                # Base time
                base_time = datetime(2000 + YY, MM, DD, HH, MI, SS)
                
                # Compute the actual start time by adding the video number offset
                start_time = base_time + timedelta(minutes=20 * (video_number - 1))
                return start_time
            except Exception as e:
                print(f"Error parsing date and time from video name '{video_name}': {e}")
                return None
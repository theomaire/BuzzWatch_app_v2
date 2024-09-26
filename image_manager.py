import os
from PIL import Image, ImageTk
import tkinter as tk

class ImageManager:
    def __init__(self, log_func):
        self.log = log_func
        self.resize_factor = 0.75

    def display_image(self, image_path, label):
        try:
            img = Image.open(image_path)
            img = img.resize((int(img.width * self.resize_factor), int(img.height * self.resize_factor)))
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)
        except Exception as e:
            self.log(f"Error displaying image {image_path}: {str(e)}")

    def update_image_list(self, listbox, path):
        image_files = [f for f in os.listdir(path) if f.endswith('.png')]
        image_files.sort()
        listbox.delete(0, tk.END)
        for image in image_files:
            listbox.insert(tk.END, image)
        self.log("Image list updated.")

    def update_median_image_list(self, listbox, path):
        image_files = [f for f in os.listdir(path) if f.endswith('.png')]
        image_files.sort()
        listbox.delete(0, tk.END)
        for image in image_files:
            listbox.insert(tk.END, image)
        self.log("Median image list updated.")

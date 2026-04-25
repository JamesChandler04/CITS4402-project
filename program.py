import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
from pathlib import Path
from typing import Optional
from enum import Enum

class ImgType(Enum):
    Original = "original"
    Processed = "processed"

MAX_WIDTH = 300

class ImageGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("Image GUI")

        # Set window to correct size at the start, including padding between images
        self.master.grid_rowconfigure(0, minsize=MAX_WIDTH + 5)
        self.master.grid_columnconfigure(0, minsize=MAX_WIDTH + 5)
        self.master.grid_columnconfigure(1, minsize=MAX_WIDTH + 5)

        # Original image
        self.original_image_label = tk.Label(master)
        self.original_image_label.grid(row=0, column=0, padx=0, pady=0)

        # Image after processing
        self.processed_image_label = tk.Label(master)
        self.processed_image_label.grid(row=0, column=1, padx=0, pady=0)

        # Processing time label
        self.processing_time_label = tk.Label(master, text="Single Image Processed In:")
        self.processing_time_label.grid(row=1, column=0, columnspan=2, sticky="w")

        # Faces found label
        self.faces_found_label = tk.Label(master, text="Faces Detected:")
        self.faces_found_label.grid(row=2, column=0, columnspan=2, sticky="w")

        # Load image button
        self.load_button = tk.Button(master, text="Single Image", command=self.load_single_image)
        self.load_button.grid(row=3, column=0)

        # Bulk processing button
        self.bulk_button = tk.Button(master, text="Bulk Processing", command=self.bulk_processing)
        self.bulk_button.grid(row=3, column=1)


    def load_single_image(self) -> None:
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        
        self.original_image = Image.open(file_path)

        self.display_image(self.original_image, ImgType.Original)

        processed_image = self.process_image(self.original_image)
        self.display_image(processed_image, ImgType.Processed)

    def process_image(self, img: Image.Image) -> Image.Image:

        photo = ImageTk.PhotoImage(img)

        # TODO: Process image

        print(f"Processed image: ", img.info)

        return img

    def bulk_processing(self) -> None:
        folder_path = filedialog.askdirectory(title="Select Folder for Bulk Processing")
        print(f"Selected folder for bulk processing: {folder_path}")

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        for filename in os.listdir(folder_path):
            if Path(filename).suffix.lower() in image_extensions:
                file_path = os.path.join(folder_path, filename)
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, ImgType.Original)
                processed_image = self.process_image(self.original_image)
                self.display_image(processed_image, ImgType.Processed)

    def display_image(self, img: Image.Image, type: ImgType) -> None:
        width, height = img.size

        if width > MAX_WIDTH:
            new_height = int(height * (MAX_WIDTH / width))
            img = img.resize((MAX_WIDTH, new_height))

        photo = ImageTk.PhotoImage(img)

        if type == ImgType.Original:
            self.original_image_label.configure(image=photo)
            self.original_image_label.image = photo  # type: ignore [attr-defined]
        elif type == ImgType.Processed:
            self.processed_image_label.configure(image=photo)
            self.processed_image_label.image = photo  # type: ignore [attr-defined]
        
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()

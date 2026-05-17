import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
import os
from pathlib import Path
from enum import Enum
import time


class ImgType(Enum):
    Original = "original"
    Processed = "processed"


MAX_HEIGHT = 225
MAX_WIDTH = 300


class ImageGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("PROJECT Image GUI")

        # Load face detector
        self.net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )

        # Set window to correct size at the start, including padding between images
        self.master.grid_rowconfigure(1, minsize=MAX_HEIGHT + 5)
        self.master.grid_columnconfigure(0, minsize=MAX_WIDTH + 5)
        self.master.grid_columnconfigure(1, minsize=MAX_WIDTH + 5)

        # Original image label
        self.original_text = tk.Label(master, text="Original")
        self.original_text.grid(row=0, column=0)

        # Processed image label
        self.processed_text = tk.Label(master, text="Processed")
        self.processed_text.grid(row=0, column=1)

        # Original image
        self.original_image_label = tk.Label(master)
        self.original_image_label.grid(row=1, column=0, padx=0, pady=0)

        # Image after processing
        self.processed_image_label = tk.Label(master)
        self.processed_image_label.grid(row=1, column=1, padx=0, pady=0)

        # Processing time label
        self.processing_time_label = tk.Label(
            master, text="Single Image Processed In:")
        self.processing_time_label.grid(
            row=2, column=0, columnspan=2, sticky="w")

        # Faces found label
        self.faces_found_label = tk.Label(master, text="Faces Detected:")
        self.faces_found_label.grid(row=3, column=0, columnspan=2, sticky="w")

        # Load image button
        self.load_button = tk.Button(
            master, text="Single Image", command=self.load_single_image)
        self.load_button.grid(row=4, column=0)

        # Bulk processing button
        self.bulk_button = tk.Button(
            master, text="Bulk Processing", command=self.bulk_processing)
        self.bulk_button.grid(row=4, column=1)

    def load_single_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )

        if not file_path:
            return

        # Load original image
        original_image = Image.open(file_path)

        # Display original
        self.display_image(original_image, ImgType.Original)

        # Process image
        start_time = time.time()

        processed_image, faces = self.find_faces(file_path)

        end_time = time.time()

        # Update labels
        self.processing_time_label.configure(
            text=f"Single Image Processed In: {end_time - start_time:.2f} seconds"
        )

        self.faces_found_label.configure(
            text=f"Faces Detected: {len(faces)}"
        )

        # processed_image, face_photos = self.process_image(original_image)
        # end_time = time.time()
        # self.processing_time_label.configure(
        #     text=f"Single Image Processed In: {end_time - start_time:.2f} seconds")
        # self.faces_found_label.configure(
        #     text=f"Faces Detected: {len(face_photos)}")

        # Display processed image
        self.display_image(processed_image, ImgType.Processed)

    def find_faces(self, file_path: str):

        image = cv2.imread(file_path)

        (h, w) = image.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Run detector
        self.net.setInput(blob)

        detections = self.net.forward()

        faces = []

        # Loop over detections
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            # Ignore weak detections
            # Test to see if this threshold can be increased for fewer potential false positives (there are none from the sample images tho)
            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")

                faces.append((startX, startY, endX, endY))

                # Draw rectangle
                cv2.rectangle(
                    image,
                    (startX, startY),
                    (endX, endY),
                    (0, 255, 0),
                    2
                )

                # Confidence text
                text = f"{confidence * 100:.1f}%"

                y = startY - 10 if startY - 10 > 10 else startY + 10

                cv2.putText(
                    image,
                    text,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2
                )

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        output_pil = Image.fromarray(image_rgb)

        return output_pil, faces

    def process_image(self, img: Image.Image) -> tuple[Image.Image, list[Image.Image]]:
        photo = ImageTk.PhotoImage(img)
        face_photos = self.find_faces(img)

        print(f"Processed image: {photo} and found {len(face_photos)} faces.")

        # TODO: Put face photos into image.

        return img, face_photos

    def bulk_processing(self) -> None:
        folder_path = filedialog.askdirectory(
            title="Select Folder for Bulk Processing")
        print(f"Selected folder for bulk processing: {folder_path}")

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        for filename in os.listdir(folder_path):
            if Path(filename).suffix.lower() in image_extensions:
                file_path = os.path.join(folder_path, filename)
                original_image = Image.open(file_path)
                self.display_image(original_image, ImgType.Original)
                processed_image, new_faces = self.process_image(original_image)
                self.display_image(processed_image, ImgType.Processed)

    def display_image(self, img: Image.Image, type: ImgType) -> None:
        width, height = img.size

        if width > MAX_WIDTH:
            new_height = int(height * (MAX_WIDTH / width))
            img = img.resize((MAX_WIDTH, new_height))

        photo = ImageTk.PhotoImage(img)

        if type == ImgType.Original:
            self.original_image_label.configure(image=photo)
            # type: ignore [attr-defined]
            self.original_image_label.image = photo
        elif type == ImgType.Processed:
            self.processed_image_label.configure(image=photo)
            # type: ignore [attr-defined]
            self.processed_image_label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()

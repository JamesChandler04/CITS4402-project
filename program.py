import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import mediapipe as mp
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

        # Mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
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

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

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

        clean_image = image.copy()

        # Draw bounding boxes and confidence labels on the display image only
        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            text = f"{detections[0, 0, faces.index((startX, startY, endX, endY)), 2] * 100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # # Draw rectangle
            # cv2.rectangle(
            #     image,
            #     (startX, startY),
            #     (endX, endY),
            #     (0, 255, 0),
            #     2
            # )

            # # Confidence text
            # text = f"{confidence * 100:.1f}%"

            # y = startY - 10 if startY - 10 > 10 else startY + 10

            # cv2.putText(
            #     image,
            #     text,
            #     (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.45,
            #     (0, 255, 0),
            #     2
            # )

        image, aligned_faces = self.detect_landmarks(image, clean_image, faces)
        image = self.place_faces_on_corners(image, aligned_faces)

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_pil = Image.fromarray(image_rgb)

        return output_pil, faces

    def align_face(self, image, left_eye, right_eye, nose):

        # Source landmark points
        src_points = np.float32([
            left_eye,
            right_eye,
            nose
        ])

        # Destination landmark points
        dst_points = np.float32([
            [40, 40],
            [85, 40],
            [63, 70]
        ])

        # Compute affine transform
        M = cv2.getAffineTransform(src_points, dst_points)

        # Warp image
        aligned_face = cv2.warpAffine(image, M, (125, 125))

        return aligned_face

    # def get_landmarks(self, face_landmarks, w, h):

    #     LEFT_EYE = 33
    #     RIGHT_EYE = 263
    #     NOSE_TIP = 1

    #     left_eye = face_landmarks.landmark[LEFT_EYE]
    #     right_eye = face_landmarks.landmark[RIGHT_EYE]
    #     nose = face_landmarks.landmark[NOSE_TIP]

    #     lx = int(left_eye.x * w)
    #     ly = int(left_eye.y * h)

    #     rx = int(right_eye.x * w)
    #     ry = int(right_eye.y * h)

    #     nx = int(nose.x * w)
    #     ny = int(nose.y * h)

    #     return (lx, ly), (rx, ry), (nx, ny)

    def detect_landmarks(self, image, clean_image, face_boxes):
        """
        Detects landmarks, draws them on the display image, and produces
        clean aligned face thumbnails from the unmodified clean_image.
        """
        image_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        aligned_faces = []
        h, w = image.shape[:2]

        if not results.multi_face_landmarks:
            return image, aligned_faces

        for face_landmarks in results.multi_face_landmarks:
            LEFT_EYE = 33
            RIGHT_EYE = 263
            NOSE_TIP = 1

            lx = int(face_landmarks.landmark[LEFT_EYE].x * w)
            ly = int(face_landmarks.landmark[LEFT_EYE].y * h)
            rx = int(face_landmarks.landmark[RIGHT_EYE].x * w)
            ry = int(face_landmarks.landmark[RIGHT_EYE].y * h)
            nx = int(face_landmarks.landmark[NOSE_TIP].x * w)
            ny = int(face_landmarks.landmark[NOSE_TIP].y * h)

            left_eye = (lx, ly)
            right_eye = (rx, ry)
            nose = (nx, ny)

            # Draw landmarks on the display image
            cv2.circle(image, left_eye,  4, (0, 255, 0), -1)  # green
            cv2.circle(image, right_eye, 4, (0, 0, 255), -1)  # red
            cv2.circle(image, nose,      4, (255, 0, 0), -1)  # blue

            # FIX 3 cont: Align from the clean full image so warpAffine has
            # full context. The 125x125 output window acts as the crop.
            aligned_face = self.align_face(
                clean_image, left_eye, right_eye, nose)

            # Draw target landmarks on aligned face thumbnail
            cv2.circle(aligned_face, (40, 40), 4,
                       (0, 255, 0), -1)  # green - left eye
            cv2.circle(aligned_face, (85, 40), 4,
                       (0, 0, 255), -1)  # red   - right eye
            cv2.circle(aligned_face, (63, 70), 4,
                       (255, 0, 0), -1)  # blue  - nose

            aligned_faces.append(aligned_face)

        return image, aligned_faces

    def place_faces_on_corners(self, image, aligned_faces):

        h, w = image.shape[:2]

        positions = [
            (0, 0),                 # top-left
            (w - 125, 0),           # top-right
            (0, h - 125),           # bottom-left
            (w - 125, h - 125)      # bottom-right
        ]

        for face, (x, y) in zip(aligned_faces, positions):

            if face.shape[0] != 125 or face.shape[1] != 125:
                continue

            image[y:y+125, x:x+125] = face

        return image

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

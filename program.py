"""
CITS4402 Computer Vision Project 2026
Title: Face Detection and Matching

Group Member 1 Name: James
Group Member 1 Student Number: 

Group Member 2 Name: Hamza
Group Member 2 Student Number: 

Group Member 3 Name: James
Group Member 3 Student Number: 
"""

# pylint: disable=no-member

import os
import time
import tkinter as tk
from enum import Enum
from tkinter import filedialog, messagebox

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk


class ImgType(Enum):
    original = "original"
    processed = "processed"


maxHeight = 320
maxWidth = 420


class ImageGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("CITS4402 Project - Face Detection and Matching")
        self.master.geometry("1100x760")
        self.master.minsize(980, 680)

        # ----------------------------
        # 1) Load face detector and landmark detector
        #    Model paths are built relative to the script location
        #    so the files can be found reliably.
        # ----------------------------
        scriptFolder = os.path.dirname(os.path.abspath(__file__))
        protoPath = os.path.join(scriptFolder, "models", "deploy.prototxt")
        modelPath = os.path.join(
            scriptFolder,
            "models",
            "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        )

        if not os.path.exists(protoPath):
            raise FileNotFoundError(f"Could not find model file: {protoPath}")

        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Could not find model file: {modelPath}")

        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # ----------------------------
        # 2) Create main title
        # ----------------------------
        self.titleLabel = tk.Label(
            master,
            text="Face Detection and Matching Project GUI",
            font=("Arial", 16, "bold"),
        )
        self.titleLabel.pack(pady=10)

        # ----------------------------
        # 3) Create image display area
        #    The input image is shown on the left and the processed
        #    image is shown on the right.
        # ----------------------------
        self.imageFrame = tk.Frame(master)
        self.imageFrame.pack(pady=10)

        self.originalFrame = tk.LabelFrame(
            self.imageFrame,
            text="Input Image",
            padx=10,
            pady=10,
        )
        self.originalFrame.pack(side="left", padx=10)

        self.processedFrame = tk.LabelFrame(
            self.imageFrame,
            text="Processed Image",
            padx=10,
            pady=10,
        )
        self.processedFrame.pack(side="left", padx=10)

        # Fixed-size container for original image
        self.originalImageContainer = tk.Frame(
            self.originalFrame,
            width=maxWidth,
            height=maxHeight,
            bg="white",
            relief="sunken",
            bd=1,
        )
        self.originalImageContainer.pack()
        self.originalImageContainer.pack_propagate(False)

        self.originalImageLabel = tk.Label(
            self.originalImageContainer,
            text="Input image will appear here",
            bg="white",
        )
        self.originalImageLabel.pack(fill="both", expand=True)

        # Fixed-size container for processed image
        self.processedImageContainer = tk.Frame(
            self.processedFrame,
            width=maxWidth,
            height=maxHeight,
            bg="white",
            relief="sunken",
            bd=1,
        )
        self.processedImageContainer.pack()
        self.processedImageContainer.pack_propagate(False)

        self.processedImageLabel = tk.Label(
            self.processedImageContainer,
            text="Processed image will appear here",
            bg="white",
        )
        self.processedImageLabel.pack(fill="both", expand=True)

        # ----------------------------
        # 4) Create status and result display area
        #    This section shows processing information and results.
        # ----------------------------
        self.textFrame = tk.Frame(master)
        self.textFrame.pack(fill="x", padx=20, pady=10)

        self.statusVar = tk.StringVar()
        self.statusVar.set("Ready.")

        self.resultVar = tk.StringVar()
        self.resultVar.set("No processing yet.")

        self.statusTitleLabel = tk.Label(
            self.textFrame,
            text="Status",
            font=("Arial", 11, "bold"),
            anchor="w",
        )
        self.statusTitleLabel.pack(fill="x")

        self.statusLabel = tk.Label(
            self.textFrame,
            textvariable=self.statusVar,
            anchor="w",
            justify="left",
            relief="sunken",
            padx=8,
            pady=8,
        )
        self.statusLabel.pack(fill="x", pady=(0, 10))

        self.resultTitleLabel = tk.Label(
            self.textFrame,
            text="Results",
            font=("Arial", 11, "bold"),
            anchor="w",
        )
        self.resultTitleLabel.pack(fill="x")

        self.resultLabel = tk.Label(
            self.textFrame,
            textvariable=self.resultVar,
            anchor="w",
            justify="left",
            relief="sunken",
            padx=8,
            pady=8,
        )
        self.resultLabel.pack(fill="x")

        # ----------------------------
        # 5) Create project buttons
        #    The GUI contains one button for loading a single image
        #    and one button for bulk processing.
        # ----------------------------
        self.buttonFrame = tk.Frame(master)
        self.buttonFrame.pack(pady=20)

        self.loadButton = tk.Button(
            self.buttonFrame,
            text="Single Image",
            width=18,
            command=self.load_single_image,
        )
        self.loadButton.grid(row=0, column=0, padx=40)

        self.bulkButton = tk.Button(
            self.buttonFrame,
            text="Bulk Processing",
            width=18,
            command=self.bulk_processing,
        )
        self.bulkButton.grid(row=0, column=1, padx=40)

    # ----------------------------
    # 6) Load a single image
    #    This opens a file dialog, displays the input image,
    #    processes it, and then displays the processed result.
    # ----------------------------
    def load_single_image(self) -> None:
        filePath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
        )

        if not filePath:
            self.statusVar.set("No image selected.")
            self.resultVar.set("Waiting for image selection.")
            return

        try:
            originalImage = Image.open(filePath)
        except Exception as error:
            messagebox.showerror("Load Error", f"Could not open the selected image.\n{error}")
            self.statusVar.set("Failed to load image.")
            self.resultVar.set("Please try another file.")
            return

        self.display_image(originalImage, ImgType.original)

        startTime = time.time()
        processedImage, faces = self.find_faces(filePath)
        endTime = time.time()

        self.display_image(processedImage, ImgType.processed)

        fileName = os.path.basename(filePath)
        processingTime = endTime - startTime
        faceCount = len(faces)

        self.statusVar.set(f"Loaded and processed image successfully: {fileName}")
        self.resultVar.set(
            f"Single image processed in {processingTime:.2f} seconds.\n"
            f"Faces detected: {faceCount}"
        )

    # ----------------------------
    # 7) Detect faces in the image
    #    This applies the face detector, draws face boxes and
    #    confidence labels, then performs landmark detection and
    #    aligned face display.
    # ----------------------------
    def find_faces(self, filePath: str):
        image = cv2.imread(filePath)

        if image is None:
            raise ValueError("Could not read the selected image using OpenCV.")

        imageHeight, imageWidth = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        confidenceValues = []

        # ----------------------------
        # 8) Collect valid face detections
        #    Weak detections are ignored using the confidence threshold.
        # ----------------------------
        for indexValue in range(detections.shape[2]):
            confidenceValue = detections[0, 0, indexValue, 2]

            if confidenceValue > 0.5:
                box = detections[0, 0, indexValue, 3:7] * np.array(
                    [imageWidth, imageHeight, imageWidth, imageHeight]
                )
                startX, startY, endX, endY = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(imageWidth - 1, endX)
                endY = min(imageHeight - 1, endY)

                if endX > startX and endY > startY:
                    faces.append((startX, startY, endX, endY))
                    confidenceValues.append(confidenceValue)

        cleanImage = image.copy()

        # ----------------------------
        # 9) Draw face boxes and confidence labels
        #    These are shown on the processed image.
        # ----------------------------
        for indexValue, (startX, startY, endX, endY) in enumerate(faces):
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            confidenceText = f"{confidenceValues[indexValue] * 100:.1f}%"
            textY = startY - 10 if startY - 10 > 10 else startY + 15

            cv2.putText(
                image,
                confidenceText,
                (startX, textY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2,
            )

        image, alignedFaces = self.detect_landmarks(image, cleanImage, faces)
        image = self.place_faces_on_corners(image, alignedFaces)

        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputPil = Image.fromarray(imageRgb)

        return outputPil, faces

    # ----------------------------
    # 10) Align a face using three landmarks
    #     This maps the left eye, right eye, and nose tip to
    #     fixed target positions in a 125 x 125 output image.
    # ----------------------------
    def align_face(self, image, leftEye, rightEye, nose):
        srcPoints = np.float32([leftEye, rightEye, nose])

        dstPoints = np.float32(
            [
                [40, 40],
                [85, 40],
                [63, 70],
            ]
        )

        transformMatrix = cv2.getAffineTransform(srcPoints, dstPoints)
        alignedFace = cv2.warpAffine(image, transformMatrix, (125, 125))

        return alignedFace

    # ----------------------------
    # 11) Detect facial landmarks
    #     This detects the left eye, right eye, and nose tip,
    #     draws them on the processed image, and creates aligned
    #     face thumbnails.
    # ----------------------------
    def detect_landmarks(self, image, cleanImage, faceBoxes):
        imageRgb = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imageRgb)

        alignedFaces = []
        imageHeight, imageWidth = image.shape[:2]

        if not results.multi_face_landmarks:
            return image, alignedFaces

        for faceLandmarks in results.multi_face_landmarks:
            leftEyeIndex = 33
            rightEyeIndex = 263
            noseTipIndex = 1

            leftX = int(faceLandmarks.landmark[leftEyeIndex].x * imageWidth)
            leftY = int(faceLandmarks.landmark[leftEyeIndex].y * imageHeight)

            rightX = int(faceLandmarks.landmark[rightEyeIndex].x * imageWidth)
            rightY = int(faceLandmarks.landmark[rightEyeIndex].y * imageHeight)

            noseX = int(faceLandmarks.landmark[noseTipIndex].x * imageWidth)
            noseY = int(faceLandmarks.landmark[noseTipIndex].y * imageHeight)

            leftEye = (leftX, leftY)
            rightEye = (rightX, rightY)
            nose = (noseX, noseY)

            # ----------------------------
            # 12) Draw landmarks on the original detected face
            #     green = left eye
            #     red = right eye
            #     blue = nose
            # ----------------------------
            cv2.circle(image, leftEye, 4, (0, 255, 0), -1)
            cv2.circle(image, rightEye, 4, (0, 0, 255), -1)
            cv2.circle(image, nose, 4, (255, 0, 0), -1)

            alignedFace = self.align_face(cleanImage, leftEye, rightEye, nose)

            # ----------------------------
            # 13) Draw target landmarks on aligned face thumbnail
            # ----------------------------
            cv2.circle(alignedFace, (40, 40), 4, (0, 255, 0), -1)
            cv2.circle(alignedFace, (85, 40), 4, (0, 0, 255), -1)
            cv2.circle(alignedFace, (63, 70), 4, (255, 0, 0), -1)

            alignedFaces.append(alignedFace)

        return image, alignedFaces

    # ----------------------------
    # 14) Place aligned face thumbnails at the image corners
    # ----------------------------
    def place_faces_on_corners(self, image, alignedFaces):
        imageHeight, imageWidth = image.shape[:2]

        positions = [
            (0, 0),
            (imageWidth - 125, 0),
            (0, imageHeight - 125),
            (imageWidth - 125, imageHeight - 125),
        ]

        for faceImage, (xValue, yValue) in zip(alignedFaces, positions):
            if faceImage.shape[0] != 125 or faceImage.shape[1] != 125:
                continue

            image[yValue:yValue + 125, xValue:xValue + 125] = faceImage

        return image

    # ----------------------------
    # 15) Prepare output folder
    #    This creates the Processed_Images folder if needed and
    #    clears old files from it.
    # ----------------------------
    def prepare_output_folder(self, baseFolder: str) -> str:
        outputFolder = os.path.join(os.path.dirname(baseFolder), "Processed_Images")

        if os.path.exists(outputFolder):
            for fileName in os.listdir(outputFolder):
                filePath = os.path.join(outputFolder, fileName)
                if os.path.isfile(filePath):
                    os.remove(filePath)
        else:
            os.makedirs(outputFolder)

        return outputFolder

    # ----------------------------
    # 16) Save cropped aligned faces
    # ----------------------------
    def save_cropped_faces(self, croppedFaces: list, outputFolder: str, identity: int, faceCounter: list) -> None:
        for faceImage in croppedFaces:
            fileName = f"Identity_{identity}_face_{faceCounter[0]}.jpg"
            savePath = os.path.join(outputFolder, fileName)
            cv2.imwrite(savePath, faceImage)
            faceCounter[0] += 1

    # ----------------------------
    # 17) Bulk processing button
    #    This button is kept in the interface, and currently
    #    displays a message.
    # ----------------------------
    def bulk_processing(self) -> None:
        self.statusVar.set("Bulk Processing button pressed.")
        self.resultVar.set("Bulk processing is currently unavailable.")
        messagebox.showinfo(
            "Information",
            "Bulk processing is currently unavailable.",
        )

    # ----------------------------
    # 18) Display an image in the GUI
    #    The image is resized to fit the fixed display area while
    #    keeping its aspect ratio.
    # ----------------------------
    def display_image(self, img: Image.Image, imageType: ImgType) -> None:
        widthValue, heightValue = img.size

        scaleValue = min(maxWidth / widthValue, maxHeight / heightValue)
        newWidth = int(widthValue * scaleValue)
        newHeight = int(heightValue * scaleValue)

        resizedImage = img.resize((newWidth, newHeight))
        photoImage = ImageTk.PhotoImage(resizedImage)

        if imageType == ImgType.original:
            self.originalImageLabel.configure(image=photoImage, text="")
            self.originalImageLabel.image = photoImage

        elif imageType == ImgType.processed:
            self.processedImageLabel.configure(image=photoImage, text="")
            self.processedImageLabel.image = photoImage


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
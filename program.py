"""
CITS4402 Computer Vision Project 2026
Title: Face Detection and Matching

Group Member 1 Name: James
Group Member 1 Student Number: 23348041

Group Member 2 Name: Hamza
Group Member 2 Student Number:

Group Member 3 Name: George
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
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from PIL import Image, ImageTk

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except ImportError:
    AgglomerativeClustering = None
    silhouette_score = None


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

        landmarkerPath = os.path.join(
            scriptFolder, "models", "face_landmarker.task")
        if not os.path.exists(landmarkerPath):
            raise FileNotFoundError(
                f"Could not find model file: {landmarkerPath}")

        baseOptions = mp_tasks.BaseOptions(model_asset_path=landmarkerPath)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=baseOptions,
            num_faces=4,
            min_face_detection_confidence=0.5,
        )
        self.faceMesh = mp_vision.FaceLandmarker.create_from_options(options)

        # ----------------------------
        # 2) Prepare face feature extractor and clustering state
        #    buffalo_l embeddings will be used later for identity
        #    clustering in bulk processing.
        # ----------------------------
        self.faceEmbeddingApp = None
        self.faceRecognizer = None
        self.bulkFaceRecords = []

        # ----------------------------
        # 3) Create main title
        # ----------------------------
        self.titleLabel = tk.Label(
            master,
            text="Face Detection and Matching Project GUI",
            font=("Arial", 16, "bold"),
        )
        self.titleLabel.pack(pady=10)

        # ----------------------------
        # 4) Create image display area
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
        # 5) Create status and result display area
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
        # 6) Create project buttons
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
    # 7) Load a single image
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
            messagebox.showerror(
                "Load Error",
                f"Could not open the selected image.\n{error}",
            )
            self.statusVar.set("Failed to load image.")
            self.resultVar.set("Please try another file.")
            return

        self.display_image(originalImage, ImgType.original)

        startTime = time.time()
        processedImage, faces, rawFaceCount, _alignedFacesClean = self.find_faces(
            filePath)
        endTime = time.time()

        self.display_image(processedImage, ImgType.processed)

        fileName = os.path.basename(filePath)
        processingTime = endTime - startTime
        validatedFaceCount = len(faces)

        self.statusVar.set(
            f"Loaded and processed image successfully: {fileName}")
        self.resultVar.set(
            f"Single image processed in {processingTime:.2f} seconds.\n"
            f"Raw detections: {rawFaceCount}\n"
            f"Skin validated faces: {validatedFaceCount}"
        )

    # ----------------------------
    # 8) Create a skin mask from the colour image
    #    The image is converted to YCrCb and thresholded using
    #    a standard skin-colour range. The mask is then cleaned
    #    using morphology.
    # ----------------------------
    def create_skin_mask(self, image):
        yCrCbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        lowerSkin = np.array([0, 133, 77], dtype=np.uint8)
        upperSkin = np.array([255, 173, 127], dtype=np.uint8)

        skinMask = cv2.inRange(yCrCbImage, lowerSkin, upperSkin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        return skinMask

    # ----------------------------
    # 9) Compute skin ratio inside a detected face box
    #    Both the full box and the central region are checked.
    # ----------------------------
    def compute_skin_ratio(self, skinMask, faceBox):
        startX, startY, endX, endY = faceBox

        faceMask = skinMask[startY:endY, startX:endX]

        if faceMask.size == 0:
            return 0.0, 0.0

        overallRatio = np.count_nonzero(faceMask) / faceMask.size

        boxHeight, boxWidth = faceMask.shape[:2]
        xMargin = int(0.2 * boxWidth)
        yMargin = int(0.2 * boxHeight)

        centerMask = faceMask[
            yMargin:boxHeight - yMargin,
            xMargin:boxWidth - xMargin,
        ]

        if centerMask.size == 0:
            centerRatio = overallRatio
        else:
            centerRatio = np.count_nonzero(centerMask) / centerMask.size

        return overallRatio, centerRatio

    # ----------------------------
    # 10) Validate face detections using skin colour evidence
    #     A detection is kept if enough skin pixels are present
    #     inside the face box. If none survive, the raw detections
    #     are kept to avoid dropping all faces from one image.
    # ----------------------------
    def filter_face_boxes_with_skin(self, faceBoxes, skinMask):
        validatedFaces = []

        for faceBox in faceBoxes:
            overallRatio, centerRatio = self.compute_skin_ratio(
                skinMask, faceBox)

            if overallRatio >= 0.08 and centerRatio >= 0.12:
                validatedFaces.append(faceBox)
            elif overallRatio >= 0.14:
                validatedFaces.append(faceBox)

        if len(faceBoxes) > 0 and len(validatedFaces) == 0:
            return faceBoxes

        return validatedFaces

    # ----------------------------
    # 11) Detect faces in the image
    #     This applies the face detector, uses skin colour
    #     segmentation to validate detections, then performs
    #     landmark detection and aligned face display.
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

        rawFaces = []

        # ----------------------------
        # 12) Collect raw face detections
        #     Weak detections are ignored using the confidence threshold.
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
                    rawFaces.append((startX, startY, endX, endY))

        cleanImage = image.copy()

        # ----------------------------
        # 13) Create skin mask and validate detections
        #     The final face boxes are the detections supported
        #     by skin-colour evidence.
        # ----------------------------
        skinMask = self.create_skin_mask(cleanImage)
        faces = self.filter_face_boxes_with_skin(rawFaces, skinMask)

        # ----------------------------
        # 14) Draw final validated face boxes
        #     These are shown on the processed image.
        # ----------------------------
        for startX, startY, endX, endY in faces:
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

        image, alignedFacesDisplay, alignedFacesClean = self.detect_landmarks(
            image, cleanImage, faces)
        image = self.place_faces_on_corners(image, alignedFacesDisplay)

        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputPil = Image.fromarray(imageRgb)

        return outputPil, faces, len(rawFaces), alignedFacesClean

    # ----------------------------
    # 15) Expand a detected face box
    #     This gives the landmark detector more face context
    #     around the forehead, cheeks, and chin so the aligned
    #     corner image captures more of the whole face.
    # ----------------------------
    def expand_face_box(self, faceBox, imageWidth, imageHeight):
        startX, startY, endX, endY = faceBox

        faceWidth = endX - startX
        faceHeight = endY - startY

        xMargin = int(0.35 * faceWidth)
        topMargin = int(0.50 * faceHeight)
        bottomMargin = int(0.35 * faceHeight)

        newStartX = max(0, startX - xMargin)
        newStartY = max(0, startY - topMargin)
        newEndX = min(imageWidth, endX + xMargin)
        newEndY = min(imageHeight, endY + bottomMargin)

        return newStartX, newStartY, newEndX, newEndY

    # ----------------------------
    # 16) Compute eye centre
    #     The eye centre is calculated from multiple landmarks
    #     instead of a single point, which gives more stable
    #     alignment and reduces stretching.
    # ----------------------------
    def compute_eye_center(self, faceLandmarks, eyeIndices, cropWidth, cropHeight):
        eyePoints = []

        for indexValue in eyeIndices:
            pointX = faceLandmarks[indexValue].x * cropWidth
            pointY = faceLandmarks[indexValue].y * cropHeight
            eyePoints.append((pointX, pointY))

        eyePoints = np.array(eyePoints, dtype=np.float32)
        centerX = int(np.mean(eyePoints[:, 0]))
        centerY = int(np.mean(eyePoints[:, 1]))

        return centerX, centerY

    # ----------------------------
    # 17) Align a face using three landmarks
    #     A similarity transform is used so the aligned face is
    #     rotated and scaled without the strong shear that causes
    #     stretched corner thumbnails.
    # ----------------------------
    def align_face(self, faceImage, rightEye, leftEye, nose):
        rightEyePoint = np.float32(rightEye)
        leftEyePoint = np.float32(leftEye)
        nosePoint = np.float32(nose)

        if rightEyePoint[0] > leftEyePoint[0]:
            rightEyePoint, leftEyePoint = leftEyePoint, rightEyePoint

        eyeDistance = np.linalg.norm(rightEyePoint - leftEyePoint)

        if eyeDistance < 10:
            return None

        srcPoints = np.float32(
            [rightEyePoint, leftEyePoint, nosePoint]).reshape(-1, 1, 2)
        dstPoints = np.float32(
            [
                [40, 40],
                [85, 40],
                [63, 70],
            ]
        ).reshape(-1, 1, 2)

        transformMatrix, _ = cv2.estimateAffinePartial2D(srcPoints, dstPoints)

        if transformMatrix is None:
            return None

        alignedFace = cv2.warpAffine(
            faceImage,
            transformMatrix,
            (125, 125),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return alignedFace

    # ----------------------------
    # 18) Detect facial landmarks
    #     Landmarks are detected inside each detected face region.
    #     One aligned-face list is prepared for GUI display with
    #     landmarks, and another clean list is prepared for saving.
    # ----------------------------
    def detect_landmarks(self, image, cleanImage, faceBoxes):
        alignedFacesDisplay = []
        alignedFacesClean = []
        imageHeight, imageWidth = image.shape[:2]

        firstEyeIndices = [33, 133, 159, 145, 158, 153]
        secondEyeIndices = [362, 263, 386, 374, 385, 380]
        noseTipIndex = 1

        for faceBox in faceBoxes:
            expandedStartX, expandedStartY, expandedEndX, expandedEndY = self.expand_face_box(
                faceBox,
                imageWidth,
                imageHeight,
            )

            faceCrop = cleanImage[
                expandedStartY:expandedEndY,
                expandedStartX:expandedEndX,
            ].copy()

            if faceCrop.size == 0:
                continue

            faceCropRgb = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2RGB)
            mpImage = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=faceCropRgb)
            results = self.faceMesh.detect(mpImage)

            if not results.face_landmarks:
                continue

            faceLandmarks = results.face_landmarks[0]

            cropHeight, cropWidth = faceCrop.shape[:2]

            firstEye = self.compute_eye_center(
                faceLandmarks,
                firstEyeIndices,
                cropWidth,
                cropHeight,
            )

            secondEye = self.compute_eye_center(
                faceLandmarks,
                secondEyeIndices,
                cropWidth,
                cropHeight,
            )

            noseX = int(faceLandmarks[noseTipIndex].x * cropWidth)
            noseY = int(faceLandmarks[noseTipIndex].y * cropHeight)
            nose = (noseX, noseY)

            # ----------------------------
            # 19) Determine which eye is on the left side of the image
            #     The leftmost eye in the image is placed at x = 40
            #     and the rightmost eye is placed at x = 85.
            # ----------------------------
            if firstEye[0] < secondEye[0]:
                rightEyeLocal = firstEye
                leftEyeLocal = secondEye
            else:
                rightEyeLocal = secondEye
                leftEyeLocal = firstEye

            rightEyeGlobal = (
                expandedStartX + rightEyeLocal[0],
                expandedStartY + rightEyeLocal[1],
            )
            leftEyeGlobal = (
                expandedStartX + leftEyeLocal[0],
                expandedStartY + leftEyeLocal[1],
            )
            noseGlobal = (
                expandedStartX + nose[0],
                expandedStartY + nose[1],
            )

            # ----------------------------
            # 20) Draw landmarks on the processed image
            #     red = right eye
            #     green = left eye
            #     blue = nose
            # ----------------------------
            cv2.circle(image, rightEyeGlobal, 4, (0, 0, 255), -1)
            cv2.circle(image, leftEyeGlobal, 4, (0, 255, 0), -1)
            cv2.circle(image, noseGlobal, 4, (255, 0, 0), -1)

            alignedFaceClean = self.align_face(
                faceCrop, rightEyeLocal, leftEyeLocal, nose)

            if alignedFaceClean is None:
                continue

            alignedFaceDisplay = alignedFaceClean.copy()

            # ----------------------------
            # 21) Draw target landmarks on aligned face thumbnail
            #     red = right eye
            #     green = left eye
            #     blue = nose
            # ----------------------------
            cv2.circle(alignedFaceDisplay, (40, 40), 4, (0, 0, 255), -1)
            cv2.circle(alignedFaceDisplay, (85, 40), 4, (0, 255, 0), -1)
            cv2.circle(alignedFaceDisplay, (63, 70), 4, (255, 0, 0), -1)

            alignedFacesDisplay.append(alignedFaceDisplay)
            alignedFacesClean.append(alignedFaceClean)

        return image, alignedFacesDisplay, alignedFacesClean

    # ----------------------------
    # 22) Place aligned face thumbnails at the image corners
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
            if faceImage is None:
                continue

            if faceImage.shape[0] != 125 or faceImage.shape[1] != 125:
                continue

            image[yValue:yValue + 125, xValue:xValue + 125] = faceImage

        return image

    # ----------------------------
    # 23) Prepare output folder
    #     This creates the Processed_Images folder if needed and
    #     clears old files from it.
    # ----------------------------
    def prepare_output_folder(self, baseFolder: str) -> str:
        outputFolder = os.path.join(
            os.path.dirname(baseFolder), "Processed_Images")

        if os.path.exists(outputFolder):
            for fileName in os.listdir(outputFolder):
                filePath = os.path.join(outputFolder, fileName)
                if os.path.isfile(filePath):
                    os.remove(filePath)
        else:
            os.makedirs(outputFolder)

        return outputFolder

    # ----------------------------
    # 24) Load pretrained face feature model
    #     buffalo_l is used here so aligned faces can be converted
    #     into embeddings for the later identity clustering step.
    # ----------------------------
    def initialize_face_embedder(self) -> bool:
        if self.faceRecognizer is not None:
            return True

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except Exception as error:
            messagebox.showerror(
                "Model Load Error",
                f"Could not import insightface correctly.\n{error}",
            )
            self.statusVar.set("Face feature model is unavailable.")
            self.resultVar.set("Check insightface / numpy installation.")
            return False

        try:
            self.statusVar.set(
                "Loading pretrained face feature model. First run may take time.")
            self.resultVar.set(
                "Please wait while buffalo_l model is prepared.")
            self.master.update_idletasks()

            self.faceEmbeddingApp = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self.faceEmbeddingApp.prepare(ctx_id=0, det_size=(640, 640))

            if "recognition" not in self.faceEmbeddingApp.models:
                raise RuntimeError(
                    "Recognition model was not loaded from buffalo_l.")

            self.faceRecognizer = self.faceEmbeddingApp.models["recognition"]
            return True

        except Exception as error:
            messagebox.showerror(
                "Model Load Error",
                f"Could not initialize pretrained face feature model.\n{error}",
            )
            self.statusVar.set("Failed to load face feature model.")
            self.resultVar.set("Bulk feature extraction could not start.")
            return False

    # ----------------------------
    # 25) Extract embedding from one aligned face
    #     The aligned face is converted into a normalized feature
    #     vector which will later be used for identity clustering.
    # ----------------------------
    def extract_face_embedding(self, faceImage):
        if self.faceRecognizer is None:
            return None

        try:
            resizedFace = cv2.resize(faceImage, (112, 112))
            featureOutput = self.faceRecognizer.get_feat(resizedFace)
            featureVector = np.array(featureOutput, dtype=np.float32).flatten()

            featureNorm = np.linalg.norm(featureVector)
            if featureNorm > 0:
                featureVector = featureVector / featureNorm

            return featureVector
        except Exception:
            return None

    # ----------------------------
    # 26) Build cosine-distance matrix
    #     The embeddings are already normalized, so cosine
    #     distance can be computed from their dot products.
    # ----------------------------
    def build_distance_matrix(self):
        featureMatrix = np.array(
            [record["embedding"] for record in self.bulkFaceRecords],
            dtype=np.float32,
        )

        featureNorms = np.linalg.norm(featureMatrix, axis=1, keepdims=True)
        featureNorms[featureNorms == 0] = 1.0
        normalizedFeatures = featureMatrix / featureNorms

        similarityMatrix = normalizedFeatures @ normalizedFeatures.T
        similarityMatrix = np.clip(similarityMatrix, -1.0, 1.0)

        distanceMatrix = 1.0 - similarityMatrix
        np.fill_diagonal(distanceMatrix, 0.0)

        return distanceMatrix

    # ----------------------------
    # 27) Run agglomerative clustering for one threshold
    #     This helper supports both newer and older versions of
    #     scikit-learn.
    # ----------------------------
    def run_agglomerative(self, distanceMatrix, thresholdValue):
        try:
            clusterModel = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="complete",
                distance_threshold=thresholdValue,
            )
        except TypeError:
            clusterModel = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage="complete",
                distance_threshold=thresholdValue,
            )

        labels = clusterModel.fit_predict(distanceMatrix)
        return labels

    # ----------------------------
    # 28) Choose the best automatic clustering result
    #     Several distance thresholds are tried. The threshold
    #     with the best silhouette score is selected, with a
    #     penalty for too many singleton clusters.
    # ----------------------------
    def cluster_face_records(self):
        if AgglomerativeClustering is None or silhouette_score is None:
            messagebox.showerror(
                "Missing Library",
                "Please install scikit-learn before running identity clustering.",
            )
            self.statusVar.set("Identity clustering is unavailable.")
            self.resultVar.set("Install scikit-learn first.")
            return [], 0, None

        faceCount = len(self.bulkFaceRecords)

        if faceCount == 0:
            return [], 0, None

        if faceCount == 1:
            return [1], 1, None

        distanceMatrix = self.build_distance_matrix()

        thresholdCandidates = np.arange(0.10, 0.85, 0.02)
        bestScore = -999999.0
        bestLabels = None
        bestThreshold = None

        for thresholdValue in thresholdCandidates:
            try:
                labels = self.run_agglomerative(distanceMatrix, thresholdValue)
            except Exception:
                continue

            uniqueLabels = np.unique(labels)
            clusterCount = len(uniqueLabels)

            if clusterCount < 2 or clusterCount > faceCount:
                continue

            try:
                silhouetteValue = silhouette_score(
                    distanceMatrix,
                    labels,
                    metric="precomputed",
                )
            except Exception:
                continue

            singletonCount = 0
            for labelValue in uniqueLabels:
                labelMembers = np.sum(labels == labelValue)
                if labelMembers == 1:
                    singletonCount += 1

            singletonRatio = singletonCount / faceCount

            # ----------------------------
            # Prefer well-separated clusters but penalize solutions
            # that make almost every face its own identity.
            # ----------------------------
            combinedScore = silhouetteValue - (0.20 * singletonRatio)

            if combinedScore > bestScore:
                bestScore = combinedScore
                bestLabels = labels.copy()
                bestThreshold = float(thresholdValue)

        if bestLabels is None:
            fallbackThreshold = 0.70
            bestLabels = self.run_agglomerative(
                distanceMatrix, fallbackThreshold)
            bestThreshold = fallbackThreshold

        identityNumbers = []
        labelToIdentity = {}
        nextIdentity = 1

        for labelValue in bestLabels:
            if labelValue not in labelToIdentity:
                labelToIdentity[labelValue] = nextIdentity
                nextIdentity += 1
            identityNumbers.append(labelToIdentity[labelValue])

        uniqueIdentityCount = len(set(identityNumbers))

        return identityNumbers, uniqueIdentityCount, bestThreshold

    # ----------------------------
    # 29) Save clustered faces
    #     Output files are saved without landmarks using the
    #     required format Identity_N_face_M.jpg
    # ----------------------------
    def save_clustered_faces(self, outputFolder: str, identityNumbers: list) -> int:
        identityFaceCounts = {}
        savedCount = 0

        for record, identityNumber in zip(self.bulkFaceRecords, identityNumbers):
            if identityNumber not in identityFaceCounts:
                identityFaceCounts[identityNumber] = 0

            identityFaceCounts[identityNumber] += 1

            fileName = f"Identity_{identityNumber}_face_{identityFaceCounts[identityNumber]}.jpg"
            savePath = os.path.join(outputFolder, fileName)
            cv2.imwrite(savePath, record["faceImage"])
            savedCount += 1

        return savedCount

    # ----------------------------
    # 30) Bulk processing button
    #     This opens a folder, processes all images in it, extracts
    #     aligned faces, computes embeddings, clusters them into
    #     identities, saves the clustered faces, and reports the
    #     final identity count on the GUI.
    # ----------------------------
    def bulk_processing(self) -> None:
        folderPath = filedialog.askdirectory(
            title="Select Folder for Bulk Processing")

        if not folderPath:
            self.statusVar.set("No folder selected.")
            self.resultVar.set("Waiting for folder selection.")
            return

        if not self.initialize_face_embedder():
            return

        imageExtensions = [".png", ".jpg", ".jpeg",
                           ".bmp", ".gif", ".tif", ".tiff"]
        imageFiles = []

        for fileName in os.listdir(folderPath):
            fullPath = os.path.join(folderPath, fileName)
            extension = os.path.splitext(fileName)[1].lower()

            if os.path.isfile(fullPath) and extension in imageExtensions:
                imageFiles.append(fullPath)

        imageFiles.sort()

        if len(imageFiles) == 0:
            messagebox.showwarning(
                "No Images", "No image files were found in the selected folder.")
            self.statusVar.set("No valid images found in selected folder.")
            self.resultVar.set("Please choose another folder.")
            return

        outputFolder = self.prepare_output_folder(folderPath)

        totalImagesProcessed = 0
        totalRawDetections = 0
        totalValidatedFaces = 0
        totalEmbeddingsExtracted = 0
        self.bulkFaceRecords = []

        startTime = time.time()

        for filePath in imageFiles:
            try:
                originalImage = Image.open(filePath)
                processedImage, faces, rawFaceCount, alignedFacesClean = self.find_faces(
                    filePath)

                self.display_image(originalImage, ImgType.original)
                self.display_image(processedImage, ImgType.processed)
                self.master.update_idletasks()

                fileName = os.path.basename(filePath)

                for faceIndex, faceImage in enumerate(alignedFacesClean, start=1):
                    embeddingVector = self.extract_face_embedding(faceImage)

                    if embeddingVector is not None:
                        self.bulkFaceRecords.append(
                            {
                                "sourceFile": fileName,
                                "faceIndex": faceIndex,
                                "faceImage": faceImage.copy(),
                                "embedding": embeddingVector,
                            }
                        )
                        totalEmbeddingsExtracted += 1

                totalImagesProcessed += 1
                totalRawDetections += rawFaceCount
                totalValidatedFaces += len(faces)

            except Exception:
                continue

        identityNumbers, uniqueIdentityCount, bestThreshold = self.cluster_face_records()

        totalSavedFaces = 0
        if len(identityNumbers) > 0:
            totalSavedFaces = self.save_clustered_faces(
                outputFolder, identityNumbers)

        endTime = time.time()
        processingTime = endTime - startTime

        if bestThreshold is None:
            thresholdText = "N/A"
        else:
            thresholdText = f"{bestThreshold:.2f}"

        self.statusVar.set(
            f"Bulk processing completed successfully. Output folder: {outputFolder}"
        )
        self.resultVar.set(
            f"Total {totalImagesProcessed} image(s) processed in {processingTime:.2f} seconds.\n"
            f"{totalValidatedFaces} face(s) detected corresponding to {uniqueIdentityCount} unique identit(y/ies).\n"
            f"Raw detections: {totalRawDetections}\n"
            f"Face embeddings extracted: {totalEmbeddingsExtracted}\n"
            f"Clustered faces saved: {totalSavedFaces}\n"
            f"Chosen clustering threshold: {thresholdText}"
        )

        messagebox.showinfo(
            "Bulk Processing Complete",
            f"Processed {totalImagesProcessed} image(s), detected {totalValidatedFaces} face(s), and found {uniqueIdentityCount} unique identit(y/ies).",
        )

    # ----------------------------
    # 31) Display an image in the GUI
    #     The image is resized to fit the fixed display area while
    #     keeping its aspect ratio.
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

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2

class ImageGUI:
   
    def __init__(self, master):
        self.master = master
        self.master.title("Image GUI")
    
        # Create a frame for the GUI and center it
        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Create a border for the GUI
        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.grid(row=0, column=0, sticky="nsew")
        
        # Create a "Load Image" button
        self.load_button = tk.Button(self.border, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.TOP, padx=5, pady=5)
        # Create a label to display the chosen image
        self.image_label = tk.Label(self.border)
        self.image_label.pack(side=tk.TOP, padx=5, pady=5)
        
    
        # Create a Scale widget for the kernel size
        self.kernel_size_var = tk.IntVar()
        self.kernel_size_scale = tk.Scale(self.border, from_=1, to=31, orient=tk.HORIZONTAL, label="Kernel Size", variable=self.kernel_size_var)
        self.kernel_size_scale.pack(side=tk.RIGHT, padx=5, pady=5)
    
        # Create a Scale widget for the standard deviation
        self.sigma_var = tk.DoubleVar()
        self.sigma_scale = tk.Scale(self.border, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sigma", variable=self.sigma_var)
        self.sigma_scale.pack(side=tk.LEFT, padx=5, pady=5)
    
        # Create a "Histogram Equalize" button
        self.histogram_button = tk.Button(self.border, text="Histogram Equalize", command=self.histogram_equalize)
        self.histogram_button.pack(side=tk.TOP, padx=5, pady=5)
        # Create a label to display the equalized image
        self.equalized_label = tk.Label(self.border)
        self.equalized_label.pack(side=tk.TOP, padx=5, pady=5)
    
            # Create a "Low Pass" button
        self.low_pass_button = tk.Button(self.border, text="Low Pass ->>", command=self.low_pass_filter)
        self.low_pass_button.pack(side=tk.LEFT, padx=5, pady=5)
        
            # Create a "High Pass" button
        self.high_pass_button = tk.Button(self.border, text="<<- High Pass", command=self.high_pass_filter)
        self.high_pass_button.pack(side=tk.RIGHT, padx=5, pady=5)
        # Create a label to display the filtered image
        self.filtered_label = tk.Label(self.border)
        self.filtered_label.pack(side=tk.TOP, padx=5, pady=5)

     
    def load_image(self):
        # Open a file selection dialog box to choose an image file
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        
        # Load the chosen image using PIL
        self.original_image = Image.open(file_path)
        
        # Resize the image to fit in the label
        width, height = self.original_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        self.original_image = self.original_image.resize((new_width, new_height))
        
        # Convert the image to Tkinter format and display it on the left side
        photo = ImageTk.PhotoImage(self.original_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        

    def histogram_equalize(self):
        # Convert the original image to grayscale
        grayscale_image = self.original_image.convert('L')
        
        # Perform histogram equalization using OpenCV
        np_image = np.array(grayscale_image)
        equalized_image = cv2.equalizeHist(np_image)
        
        # Convert the equalized image back to PIL format
        pil_image = Image.fromarray(equalized_image)
        
        # Resize the image to fit in the label
        width, height = pil_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        pil_image = pil_image.resize((new_width, new_height))
        
        # Convert the image to Tkinter format and display it on the right side
        photo = ImageTk.PhotoImage(pil_image)
        self.equalized_label.configure(image=photo)
        self.equalized_label.image = photo
        
        
    def high_pass_filter(self):
        # Get the kernel size and sigma from the scale widgets
        ksize = self.kernel_size_var.get()
    
        # Convert the original image to grayscale
        grayscale_image = self.original_image.convert('L')
    
        # Convert the grayscale image to a numpy array
        np_image = np.array(grayscale_image)
    
        # Apply a Laplacian filter to the image
        filtered_image = cv2.Laplacian(np_image, cv2.CV_64F, ksize=ksize)
    
        # Normalize the filtered image to 0-255 range
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
        # Convert the filtered image back to PIL format
        pil_image = Image.fromarray(filtered_image)
    
        # Resize the image to fit in the label
        width, height = pil_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        pil_image = pil_image.resize((new_width, new_height))
    
        # Convert the image to Tkinter format and display it on the right side
        photo = ImageTk.PhotoImage(pil_image)
        self.filtered_label.configure(image=photo)
        self.filtered_label.image = photo
        
        
    def low_pass_filter(self):
        # Get the kernel size and sigma from the scale widgets
        ksize = self.kernel_size_var.get()
        sigma = self.sigma_var.get()
    
        # Convert the original image to grayscale
        grayscale_image = self.original_image.convert('L')
    
        # Convert the grayscale image to a numpy array
        np_image = np.array(grayscale_image)
    
        # Apply a Gaussian filter to the image
        filtered_image = cv2.GaussianBlur(np_image, (ksize, ksize), sigma)
    
        # Convert the filtered image back to PIL format
        pil_image = Image.fromarray(filtered_image)
    
        # Resize the image to fit in the label
        width, height = pil_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        pil_image = pil_image.resize((new_width, new_height))
    
        # Convert the image to Tkinter format and display it on the right side
        photo = ImageTk.PhotoImage(pil_image)
        self.filtered_label.configure(image=photo)
        self.filtered_label.image = photo
  
        
            
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()

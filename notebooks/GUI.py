import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

class PlantDiseaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Classification")
        self.root.geometry("700x750")
        self.root.configure(bg="#1e1e1e")  # Dark background
        
        # Print Python version
        print(f"Python version: {sys.version}")
        print(f"TensorFlow version: {tf.__version__}")
        
        try:
            # Load the model
            model_path = 'plant_disease_model.h5'
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}\nPlease make sure the model file exists in the same directory.")
                self.root.destroy()
                return
                
            # Use tf.keras instead of keras directly
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
            return
        
        # Define class names
        self.class_names = {
            0: 'Pepper bell Bacterial spot',
            1: 'Pepper bell healthy',
            2: 'Potato Early blight',
            3: 'Potato Late blight',
            4: 'Potato healthy',
            5: 'Tomato Bacterial spot',
            6: 'Tomato Early blight',
            7: 'Tomato Late blight',
            8: 'Tomato Leaf Mold',
            9: 'Tomato Septoria leaf spot',
            10: 'Tomato Spider mites',
            11: 'Tomato Target Spot',
            12: 'Tomato Yellow Leaf Curl Virus',
            13: 'Tomato mosaic virus',
            14: 'Tomato healthy'
        }
        
        # Use a modern theme if available
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        # Configure dark theme styles
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff', font=('Segoe UI', 13))
        style.configure('Header.TLabel', 
                       font=('Segoe UI', 22, 'bold'), 
                       foreground='#4CAF50',  # Green
                       background='#2d2d2d')
        style.configure('Result.TLabel', 
                       font=('Segoe UI', 16, 'bold'), 
                       background='#2d2d2d', 
                       foreground='#2196F3')  # Blue
        style.configure('Confidence.TLabel', 
                       font=('Segoe UI', 14, 'bold'), 
                       background='#2d2d2d', 
                       foreground='#4CAF50')  # Green
        style.configure('TButton', 
                       font=('Segoe UI', 12, 'bold'), 
                       padding=8, 
                       background='#4CAF50',  # Green
                       foreground='white',
                       relief='flat')
        style.map('TButton', 
                 background=[('active', '#45a049')],  # Darker green
                 foreground=[('active', 'white')])
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="30 20 30 20", style='TFrame')
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Logo at the top (centered)
        logo_path = 'logo.png'
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((120, 120), Image.ANTIALIAS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(self.main_frame, image=self.logo_photo, bg='#1e1e1e', borderwidth=0)
            logo_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        # else:
        #     logo_label = tk.Label(self.main_frame, text="THE OPTIMIZERS", font=('Segoe UI', 20, 'bold'), fg='#4CAF50', bg='#1e1e1e')
        #     logo_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Plant Disease Classification", style='Header.TLabel', anchor='center')
        title_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky='ew')
        
        # Upload button
        self.upload_btn = ttk.Button(self.main_frame, text="Upload Image", command=self.upload_image, style='TButton')
        self.upload_btn.grid(row=2, column=0, columnspan=2, pady=15, sticky='ew')
        
        # Image display frame (rounded corners effect by color contrast)
        self.image_frame = tk.Frame(self.main_frame, 
                                  width=350, 
                                  height=350, 
                                  bg='#2d2d2d',  # Dark gray
                                  highlightbackground='#4CAF50',  # Green border
                                  highlightthickness=2)
        self.image_frame.grid(row=3, column=0, columnspan=2, pady=20)
        self.image_frame.grid_propagate(False)
        self.image_label = ttk.Label(self.image_frame, background='#2d2d2d')
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Prediction result
        self.result_label = ttk.Label(self.main_frame, text="", style='Result.TLabel', anchor='center')
        self.result_label.grid(row=4, column=0, columnspan=2, pady=(10, 5), sticky='ew')
        
        # Confidence score
        self.confidence_label = ttk.Label(self.main_frame, text="", style='Confidence.TLabel', anchor='center')
        self.confidence_label.grid(row=5, column=0, columnspan=2, pady=(0, 10), sticky='ew')
        
        # Footer
        self.footer_label = ttk.Label(self.main_frame, 
                                    text="Developed by The Optimizers", 
                                    font=('Segoe UI', 11, 'bold'), 
                                    background='#1e1e1e', 
                                    foreground='#4CAF50')
        self.footer_label.grid(row=6, column=0, columnspan=2, pady=(30, 0), sticky='ew')
        
    def upload_image(self):
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )
            
            if file_path:
                # Load and preprocess image
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("Error", "Failed to load image. Please try another image file.")
                    return
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image for display
                display_img = cv2.resize(img, (340, 340))
                display_img = Image.fromarray(display_img)
                display_img = ImageTk.PhotoImage(display_img)
                
                # Update image display
                self.image_label.configure(image=display_img)
                self.image_label.image = display_img
                
                # Preprocess image for prediction
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Make prediction with error handling
                try:
                    prediction = self.model.predict(img, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class] * 100
                    
                    # Color for healthy/disease
                    if 'healthy' in self.class_names[predicted_class].lower():
                        result_fg = '#4CAF50'  # Green for healthy
                    else:
                        result_fg = '#f44336'  # Red for disease
                    self.result_label.configure(
                        text=f"Prediction: {self.class_names[predicted_class]}",
                        foreground=result_fg
                    )
                    self.confidence_label.configure(
                        text=f"Confidence: {confidence:.2f}%"
                    )
                except Exception as pred_error:
                    messagebox.showerror("Prediction Error", f"Error during prediction: {str(pred_error)}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    try:
        # Configure TensorFlow to use CPU if GPU is not available
        tf.config.set_visible_devices([], 'GPU')
        
        root = tk.Tk()
        app = PlantDiseaseGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    main()

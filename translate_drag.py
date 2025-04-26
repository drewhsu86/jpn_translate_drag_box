import tkinter as tk
import pyautogui
import keyboard
import sys
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from scipy.spatial import cKDTree
from deep_translator import GoogleTranslator
# fix fugashi for manga_ocr: https://github.com/pypa/pip/issues/10605 (mecab dll)
from manga_ocr import MangaOcr
# import easyocr

class HotKeyManager:
    def __init__(self):
        self.mocr = MangaOcr()
        keyboard.add_hotkey('esc', self.on_quit_program)
        self.drag_hotkey = 'ctrl+alt+q'
        # Set easyOCR reader
        # self.easy_reader = easyocr.Reader([self.convert_to_easy_lang(self.lang)], gpu=False)
        # Register the hotkey to trigger the 'on_initiate_drag' method
        keyboard.add_hotkey(f'{self.drag_hotkey}', self.on_initiate_drag)
        print(f"Hotkey listening... Press {self.drag_hotkey} to initiate the drag.")
        keyboard.wait()  # Keep the program running and waiting for hotkeys

    def on_quit_program(self):
        sys.exit()

    def on_initiate_drag(self):
        print('Initiating drag box...')
        self.create_drag_box()

    def create_drag_box(self):
        # Create the root window (transparent)
        root = tk.Tk()
        root.attributes('-alpha', 0.3)  # Make window transparent
        root.attributes('-topmost', True)  # Keep window on top
        root.overrideredirect(True)  # Removes window borders
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Full screen

        # Canvas for drawing the box
        canvas = tk.Canvas(root, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)

        # Variables to track the start and end of the drag
        start_x = start_y = None
        rect = None

        def on_button_press(event):
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x_root, event.y_root  # Record starting point of the drag
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='blue', width=2)

        def on_mouse_drag(event):
            nonlocal rect
            # Update the rectangle's size while dragging
            canvas.coords(rect, start_x, start_y, event.x_root, event.y_root)

        def on_button_release(event):
            # Finalize the rectangle, we can also capture the image here
            nonlocal start_x, start_y
            x1 = min(start_x, event.x_root)
            y1 = min(start_y, event.y_root)
            x2 = max(start_x, event.x_root)
            y2 = max(start_y, event.y_root)

            print("Drag finished. Coordinates:", start_x, start_y, event.x_root, event.y_root)
            # root.quit()  # Close the window after dragging
            root.destroy()

            # Capture the region
            image = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
            # image.show()

            # ocr_text = self.perform_ocr(image)
            self.perform_ocr_jpn_vert(image)

        # Bind mouse events
        canvas.bind('<ButtonPress-1>', on_button_press)
        canvas.bind('<B1-Motion>', on_mouse_drag)
        canvas.bind('<ButtonRelease-1>', on_button_release)

        # Run the Tkinter event loop
        root.mainloop()
    
    def process_image_for_ocr(self, image):
        scale_factor = 1.5
        # process image and use it to pull text
        img = np.array(image)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.adaptiveThreshold(
            img, 
            maxValue=255, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY_INV,  # Invert for black text on white
            blockSize=21, 
            C=7
        )
        img_processed = Image.fromarray(img)
        return img_processed
    
    def perform_ocr_jpn_vert(self, image):
        image_processed = self.process_image_for_ocr(image)
        text = self.mocr(image_processed)
        print(text)
        translated = GoogleTranslator(source='ja', target='en').translate(text)
        self.popup_window_text(translated)

    def popup_window_text(self, text):
        text_width = 300
        text_height = 200

        window = tk.Tk()
        window.title("Translation")
        window.geometry(f"{text_width}x{text_height}")  # Set window size

        # Add a label to display text
        text_label = tk.Label(window, 
                              text=text, 
                              font=("Arial", 12), 
                              wraplength=int(0.8*text_width),  # Wrap text
                              justify="center")
        text_label.pack(pady=20)  # Add some padding

        # Function to close the window
        def close_window():
            window.destroy()

        # Create a close button
        close_button = tk.Button(window, text="Close", command=close_window)
        close_button.pack(pady=20)  # Add some padding

        # Start the main event loop
        window.mainloop()

# Start the hotkey manager
HotKeyManager()

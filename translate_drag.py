import tkinter as tk
import pyautogui
import keyboard
import sys
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
from scipy.spatial import cKDTree
# import easyocr

class HotKeyManager:
    def __init__(self, set_lang = 'jpn'):
        self.lang = set_lang
        self.kernel_size = 3
        keyboard.add_hotkey('esc', self.on_quit_program)
        self.drag_hotkey = 'ctrl+alt+q'
        self.lang_hotkey = 'ctrl+alt+e'
        self.kernel_size_hotkey = 'ctrl+alt+k'
        # Set easyOCR reader
        # self.easy_reader = easyocr.Reader([self.convert_to_easy_lang(self.lang)], gpu=False)
        # Register the hotkey to trigger the 'on_initiate_drag' method
        keyboard.add_hotkey(f'{self.drag_hotkey}', self.on_initiate_drag)
        keyboard.add_hotkey(f'{self.lang_hotkey}', self.on_change_language)
        keyboard.add_hotkey(f'{self.kernel_size_hotkey}', self.on_change_kernel_size)
        print(f"Hotkey listening... Press {self.drag_hotkey} to initiate the drag.")
        keyboard.wait()  # Keep the program running and waiting for hotkeys

    def on_quit_program(self):
        sys.exit()

    def on_initiate_drag(self):
        print('Initiating drag box...')
        self.create_drag_box()

    def on_change_language(self):
        print("Available: eng, jpn")
        new_lang = input("Please type in the code for your desired language: ")
        self.lang = new_lang
        # self.easy_reader = easyocr.Reader([self.convert_to_easy_lang(self.lang)], gpu=False)

    # def convert_to_easy_lang(self, code):
    #     if code.lower() == 'eng':
    #         return 'en'
    #     if 'jpn' in code.lower():
    #         return 'ja'

    def on_change_kernel_size(self):
        input_str = input("Please type in a positive integer for kernel size")
        if isinstance(input_str, str) and input_str.isdigit():
            self.kernel_size = int(input_str)

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
    
    def perform_ocr(self, image):
        image_processed = self.process_image_for_ocr(image)
        image_processed.show()
        # Convert image to text
        # easy_text = self.easy_reader.readtext(np.array(image_processed), detail=0, paragraph=True, decoder='greedy')
        # text = ''.join(easy_text)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image_processed, lang=self.lang, config=custom_config)
        print("Extracted text:", text)
        return text
    
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
        bounding_boxes = self.gen_bounding_boxes(image_processed)
        bounding_boxes = self.merge_gen_bounding_boxes(image_processed, bounding_boxes)

    def gen_bounding_boxes(self, image_processed):
        image_processed = image_processed.convert("RGB")
        # try dilating before doing contours
        dilation_size = self.kernel_size
        dilation_iter = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        dilated = cv2.dilate(np.array(image_processed), kernel, iterations=dilation_iter)

        edges = cv2.Canny(dilated, threshold1=100, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes for contours
        bounding_boxes = []

        draw = ImageDraw.Draw(image_processed)

        contour_thres = 10
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > contour_thres and h > contour_thres:  # Filter small contours (adjust threshold)
                bounding_boxes.append((x, y, w, h))
                draw.rectangle((x, y, x+w, y+h), outline="red", width=2)
        image_processed.show()

        return bounding_boxes
    
    def merge_gen_bounding_boxes(self, image_processed, bounding_boxes):
        bounding_boxes = merge_touching_boxes(bounding_boxes)
        image_processed = image_processed.convert("RGB")
        draw = ImageDraw.Draw(image_processed)  
        for (x, y, w, h) in bounding_boxes:
            draw.rectangle((x, y, x+w, y+h), outline="blue", width=2)
        image_processed.show()


def is_point_in_bbox(point, bbox):
    px, py = point
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h

# merge boxes written by AI (chatgpt, Grok)
def merge_touching_boxes(boxes, min_gap=0):
    """Merge bounding boxes that overlap or touch, iteratively.
    Args:
        boxes: List of [x, y, w, h] where x, y is top-left, w, h is width, height.
        min_gap: Allow boxes to be merged if gap between them is <= min_gap (pixels).
    Returns:
        List of merged [x, y, w, h] boxes.
    """
    if not boxes:
        return []

    def boxes_touch(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (max(x1, x2) <= min(x1 + w1, x2 + w2) + min_gap and
                max(y1, y2) <= min(y1 + h1, y2 + h2) + min_gap)

    def merge_group(group_boxes):
        x_min = min(b[0] for b in group_boxes)
        y_min = min(b[1] for b in group_boxes)
        x_max = max(b[0] + b[2] for b in group_boxes)
        y_max = max(b[1] + b[3] for b in group_boxes)
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    # Iteratively merge until no changes
    current_boxes = boxes.copy()
    while True:
        groups = []
        used = set()

        # Group touching boxes
        for i in range(len(current_boxes)):
            if i not in used:
                group = [i]
                used.add(i)
                for j in range(i + 1, len(current_boxes)):
                    if j not in used and boxes_touch(current_boxes[i], current_boxes[j]):
                        group.append(j)
                        used.add(j)
                groups.append(group)

        # Create new list of merged boxes
        new_boxes = [merge_group([current_boxes[i] for i in g]) for g in groups]

        # If no boxes were merged (same number of boxes), we're done
        if len(new_boxes) == len(current_boxes):
            break
        current_boxes = new_boxes

    return current_boxes

# Start the hotkey manager
if len(sys.argv) < 2:
    HotKeyManager()
else:
    HotKeyManager(sys.argv[1])
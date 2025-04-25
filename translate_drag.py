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
        keyboard.add_hotkey('esc', self.on_quit_program)
        self.drag_hotkey = 'ctrl+alt+q'
        self.lang_hotkey = 'ctrl+alt+e'
        # Set easyOCR reader
        # self.easy_reader = easyocr.Reader([self.convert_to_easy_lang(self.lang)], gpu=False)
        # Register the hotkey to trigger the 'on_initiate_drag' method
        keyboard.add_hotkey(f'{self.drag_hotkey}', self.on_initiate_drag)
        keyboard.add_hotkey(f'{self.lang_hotkey}', self.on_change_language)
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
        

    def gen_bounding_boxes(self, image_processed):
        image_processed = image_processed.convert("RGB")
        # try dilating before doing contours
        dilation_size = 2
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

        bounding_boxes_merged = merge_bbox_jpn(bounding_boxes)
        for bbox in bounding_boxes_merged:
            (x, y, w, h) = bbox
            draw.rectangle((x, y, x+w, y+h), outline="blue", width=2)
        image_processed.show()

        return bounding_boxes_merged

def merge_bbox_jpn(bboxes):
    # sort boxes based on center point
    # right to left (higher to lower)
    # top to bottom (lower to higher)
    bboxes = sorted(bboxes, key=lambda box: -box[0] - box[2]/2)
    bboxes = sorted(bboxes, key=lambda box: box[1] + box[3]/2)

    # Create a list of bounding box centers
    bbox_centers = np.array([(x + w / 2, y + h / 2) for (x, y, w, h) in bboxes])

    # Create a KD-tree from the centers
    kdtree = cKDTree(bbox_centers)

    merge_queue = [] # indices of boxes that have already been merged

    for i in range(len(bboxes)):
        (x, y, w, h) = bboxes[i]
        bbox_center = bbox_centers[i]
        # check aspect ratio
        # if height is not within certain ratio of width
        # its probably a malformed character
        ratio = 1.2
        candidate_ratio = 1.2
        center_thres_ratio = 0.9

        # check if boxes are too tall or flat
        # proper box height or width is most likely the higher value

        if w > ratio*h:
            # if height > ratio*width it's too tall
            # tall boxes we need to check the left
            left_center = bbox_center + np.array([w, 0])
            
            left_indices = kdtree.query_ball_point(left_center, r=w)

            left_min_index = -1
            if left_indices:
                left_min_index = min(left_indices, key=lambda j: np.linalg.norm(bbox_centers[j] - left_center) + (float('inf') if i == j else 0))

            if left_min_index < 0 or left_min_index in merge_queue:
                continue

            # check the left 
            # if they have a bad aspect ratio and are close enough to the 
            # proposed center point, we will merge
            center_thres = center_thres_ratio*w

            if np.linalg.norm(bbox_centers[left_min_index] - left_center) <= center_thres:
                (lx, ly, lw, lh) = bboxes[left_min_index]
                if lw > candidate_ratio*lh:
                    # this would be merged, otherwise ignore
                    merge_queue.append(i)
                    merge_queue.append(left_min_index)

        elif h > ratio*w:
            # if width > ratio*height it's too flat
            # flat boxes we check below
            bot_center = bbox_center + np.array([h, 0])
            
            bot_indices = kdtree.query_ball_point(bot_center, r=h)

            bot_min_index = -1
            if bot_indices:
                bot_min_index = min(bot_indices, key=lambda j: np.linalg.norm(bbox_centers[j] - bot_center) + (float('inf') if i == j else 0))

            if bot_min_index < 0 or bot_min_index in merge_queue:
                continue

            # check the bot 
            # if they have a bad aspect ratio and are close enough to the 
            # proposed center point, we will merge
            center_thres = center_thres_ratio*h

            if np.linalg.norm(bbox_centers[bot_min_index] - bot_center) <= center_thres:
                (lx, ly, lw, lh) = bboxes[bot_min_index]
                if lh > candidate_ratio*lw:
                    # this would be merged, otherwise ignore
                    merge_queue.append(i)
                    merge_queue.append(bot_min_index)

    merge_queue = sorted(merge_queue)
    print(merge_queue)
    post_merge_bboxes = []
    if len(merge_queue) == 0:
        return bboxes
    
    needs_merge = merge_queue[0]
    for i in range(len(bboxes)):
        if i == needs_merge:
            if i + 1 < len(merge_queue) - 1:
                needs_merge += 1
        else:
            post_merge_bboxes.append(bboxes[i])
    
    # the merge queue should have pairs so we should be able to iterate up
    # in steps of 2
    for i in range(0, len(merge_queue), 2):
        (x1, y1, w1, h1) = bboxes[merge_queue[i]]
        (x2, y2, w2, h2) = bboxes[merge_queue[i+1]]

        x = min(x1, x2)
        y = min(y1, y2)
        
        # Compute width and height of the merged box
        w = max(x1 + w1, x2 + w2)- x
        h = max(y1 + h1, y2 + h2) - y
        merged_bbox = (x, y, w, h )

        post_merge_bboxes.append(merged_bbox)
    return post_merge_bboxes


def is_point_in_bbox(point, bbox):
    px, py = point
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h

# Start the hotkey manager
if len(sys.argv) < 2:
    HotKeyManager()
else:
    HotKeyManager(sys.argv[1])
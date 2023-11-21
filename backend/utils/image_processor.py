import cv2
import math
import os
from enum import Enum
from pydantic import BaseModel
from typing import Callable
from utils.config import get_settings

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 600
# BASE_FOLDER = "../first_project/assets/base"
# IMAGE_FOLDER = "../first_project/assets/base/original"
SETTINGS = get_settings()
IMAGE_FOLDER = SETTINGS.images_path

class Type_Enum(str, Enum):
    RESIZE = "resize"
    CROP = "crop"
    GRAYSCALE = "grayscale"
    REMOVE_BACKGROUND = "remove_background"

class ImageProcessor(BaseModel):
    function: Callable
    output_path: str
    output_prefix: str
    message: str


def resize_and_save(image, output_path):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(output_path, img)
    return img

def grayscale_and_save(image, output_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)
    return gray

process_type = {"resize": ImageProcessor(function=resize_and_save, output_path=IMAGE_FOLDER, output_prefix="", message="Resized"),
                 "crop": None,
                   "grayscale": ImageProcessor(function=grayscale_and_save, output_path=str(IMAGE_FOLDER + "/processed"), output_prefix="", message="Grayscaled"),
                     "remove_background": None}

def process_images(type: Type_Enum=Type_Enum.RESIZE):
    process = process_type[type]
    process_function = process.function
    shape = ""
    if os.path.exists(IMAGE_FOLDER):
        for filename in os.listdir(IMAGE_FOLDER):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and not filename.startswith('resized_'):
                input_path = os.path.join(IMAGE_FOLDER, filename)
                output_path = os.path.join(process.output_path, process.output_prefix + filename)
                
                image = cv2.imread(input_path)
                processed_image = process_function(image, output_path)
                
                print(f"{process.message} and saved {filename} {processed_image.shape}")
                if filename.startswith("input"):
                    shape = processed_image.shape
        return shape
    else:
        print(f"Folder {IMAGE_FOLDER} does not exist.")

# if __name__ == "__main__":
    
#     process_images(Type_Enum.GRAYSCALE)

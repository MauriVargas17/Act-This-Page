import io
import cv2
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends
)
from fastapi.responses import Response
import numpy as np
from functools import cache
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from middlewares.inference import Comparator
from utils.config import get_settings
from utils.responses import StatusResponse
from pydantic import BaseModel
from middlewares.inference import Comparison
from utils.image_processor import ImageProcessor, Type_Enum, process_images

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

IMAGES_PATH = SETTINGS.images_path

images_dict = {
    0: '../first_project/assets/base/original/harrypotter-og.png',
    1: '../first_project/assets/base/original/lalaland-og.png',
    2: '../first_project/assets/base/original/msdoubtfire-og.png',
    3: '../first_project/assets/base/original/starwars-og.png',
    4: '../first_project/assets/base/original/taxidriver-og.png',
    5: '../first_project/assets/base/original/theshowman-og.png',
    6: '../first_project/assets/base/original/titanic-og.png',
}

@cache    
def get_comparator():
    print("Creating comparator...")
    return Comparator()
def upload_and_compare(file1, file2, comparator, mod_name):
    img_stream_1 = io.BytesIO(file1.file.read())
    with open(file2, "rb") as file:
        image_content = file.read()
        img_stream_2 = io.BytesIO(image_content)
        if file1.content_type.split("/")[0] != "image":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
                detail="Not an image"
            )
        img_obj_1 = Image.open(img_stream_1)
        img_obj_1 = np.array(img_obj_1)
        img_obj_2 = Image.open(img_stream_2)
        img_obj_2 = np.array(img_obj_2)

        img_route_1 = f"{IMAGES_PATH}/input-{mod_name}.png"
        img_route_2 = f"{IMAGES_PATH}/target-{mod_name}.png"

        img_obj_1 = cv2.cvtColor(img_obj_1, cv2.COLOR_RGB2BGR)
        img_obj_2 = cv2.cvtColor(img_obj_2, cv2.COLOR_RGB2BGR)

        cv2.imwrite(img_route_1, img_obj_1)
        cv2.imwrite(img_route_2, img_obj_2)

        process_images()
        process_images(Type_Enum.GRAYSCALE)

        img_route_1 = f"{IMAGES_PATH}/processed/input-{mod_name}.png"
        img_route_2 = f"{IMAGES_PATH}/processed/target-{mod_name}.png"

        print("Images saved")

    return comparator.compare(image_1=img_route_1, image_2=img_route_2)

@app.get("/")
def read_root():
    return {"message": "Hello, CORS is enabled"}
@app.get("/status")
async def status():
    return StatusResponse(status="ok", model_name=SETTINGS.name_of_model)

@app.post("/predict")
def compare_images(
    file_1: UploadFile = File(...), 
    file_2: int = 0, 
    file_number: int = 1,
    comparator: Comparator = Depends(get_comparator)
) -> Comparison:
    file_2_image = images_dict[file_2]
    print(file_2_image)
    print("Comparing images...")
    result = upload_and_compare(file_1, file_2_image, comparator, file_number)
    print(f"{result.similarity*100}% similarity between {file_1.filename} and target")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True) 
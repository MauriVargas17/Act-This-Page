import time
import numpy as np
from pydantic import BaseModel
from utils.config import get_settings
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import cache
from utils.responses import Comparison

SETTINGS = get_settings()

class Comparator:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path=f'../first_project/backend/model/{SETTINGS.name_of_model}') 
        self.options = vision.ImageEmbedderOptions(
    base_options=self.base_options, l2_normalize=True, quantize=True)       
    def embed(self, image_1, image_2):
        with vision.ImageEmbedder.create_from_options(self.options) as embedder:
            first_image = mp.Image.create_from_file(image_1)
            second_image = mp.Image.create_from_file(image_2)
            first_embedding_result = embedder.embed(first_image)
            second_embedding_result = embedder.embed(second_image)
            return first_embedding_result, second_embedding_result   
    def compare(self, image_1, image_2) -> Comparison:

        initial_time = time.time()
        first_embedding_result, second_embedding_result = self.embed(image_1, image_2)
        similarity = vision.ImageEmbedder.cosine_similarity(
                first_embedding_result.embeddings[0],
                second_embedding_result.embeddings[0])
            
        final_time = time.time()
        result = Comparison(
                similarity=np.round(similarity, 4),
                image_1_embedding=first_embedding_result.embeddings[0].embedding,
                image_2_embedding=second_embedding_result.embeddings[0].embedding,
                time_ms=np.round((final_time - initial_time) * 1000, 2)
            )
        return result
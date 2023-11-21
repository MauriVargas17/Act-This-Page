from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache
import os

from dotenv import load_dotenv

load_dotenv("../first_project/backend/utils/.env")

class Settings(BaseSettings):
    #model_config = SettingsConfigDict(env_file=".env")
    images_path: str = "../first_project/assets/images"
    api_name: str = "Object Detection Service"
    revision: str = "local"
    api_version: str = "1.0.0"
    name_of_model: str = "mobilenet_v3_large.tflite"
    log_level: str = "DEBUG" 

    class Config:
        env_file = ".env"


@cache
def get_settings():
    print(f"API_NAME: {os.getenv('API_NAME')}")
    print("getting settings...")
    return Settings()
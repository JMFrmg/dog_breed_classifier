import io
import pickle
from logging.config import dictConfig
import logging
from PIL import Image
from typing import Optional

from fastapi import FastAPI, File
import torch

from .config import CONFIG
from .log_config import LogConfig
from .model import get_model, get_transform

from fastapi.logger import logger as fastapi_logger

app = FastAPI()


# Logging
dictConfig(LogConfig().dict())
logger = logging.getLogger("dog_breed")

gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers

fastapi_logger.handlers = gunicorn_error_logger.handlers

if __name__ != "__main__":
    fastapi_logger.setLevel(gunicorn_logger.level)
else:
    fastapi_logger.setLevel(logging.DEBUG)

# Load model only once when app start
@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    # Load model and pretrained weights
    model = get_model()
    model.load_state_dict(torch.load(CONFIG["WEIGHTS_PATH"], map_location=CONFIG["DEVICE"]))
    logger.info("Model and weights loaded.")

    model.eval()
    model.to(CONFIG["DEVICE"])

    # Images preprocessing
    transform = get_transform()

    # Class id to class name dictionnary
    with open(CONFIG["CLASS_MAP_PATH"], "rb") as f:
        id_to_class = pickle.load(f)


    # add model and preprocess tools too app state
    app.package = {
        "id_to_class": id_to_class,
        "transform": transform,
        "model": model
    }

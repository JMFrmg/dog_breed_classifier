import torch

GLOBAL_CONFIG = {
    "WEIGHTS_PATH": "/code/app/model_files/inference_model.pt",
    "CLASS_MAP_PATH": "/code/app/model_files/reverse_class_map.pickle",
    "USE_CUDA_IF_AVAILABLE": True,
    "GPU_ID": 0
}

def get_config() -> dict:
    """
    Get config
    :return: dict
    """

    config = GLOBAL_CONFIG.copy()

    if config["USE_CUDA_IF_AVAILABLE"]:
        config["DEVICE"] = torch.device("cuda:" + str(config["GPU_ID"]) if torch.cuda.is_available() else "cpu")
    else:
        config["DEVICE"] = torch.device("cpu")


    return config

CONFIG = get_config()
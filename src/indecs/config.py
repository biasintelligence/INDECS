import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

load_dotenv()

class Config:
    REPO_ID = os.getenv("REPO_ID")
    WEIGHTS_FILE = os.getenv("WEIGHTS_FILE")
    MODEL_DIR = os.getenv("MODEL_DIR")
    DARKNET_LIB_PATH = os.getenv("DARKNET_LIB_PATH")

    @staticmethod
    def get_weights_path():
        local_path = os.path.join(Config.MODEL_DIR, Config.WEIGHTS_FILE)
        if os.path.exists(local_path):
            print("Loading weights from local file.")
            return local_path
        else:
            print("Downloading weights from Hugging Face Hub.")
            return hf_hub_download(repo_id=Config.REPO_ID, filename=Config.WEIGHTS_FILE)

    @staticmethod
    def get_config_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model", "yolov4-grouse.cfg")

    @staticmethod
    def get_data_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model", "sagegrouse.data")

    @staticmethod
    def get_names_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model", "sagegrouse.names")

    @staticmethod
    def download_model():
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
        snapshot_download(repo_id=Config.REPO_ID, local_dir=Config.MODEL_DIR)
        print(f"Model downloaded to {Config.MODEL_DIR}")
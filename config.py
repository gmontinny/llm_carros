from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # VLM
    vlm_provider: str = "openai"
    openai_api_key: str = ""
    vlm_openai_model: str = "gpt-4o"
    gemini_api_key: str = ""
    vlm_gemini_model: str = "gemini-2.5-pro"
    llama_model: str = "llava"
    llama_base_url: str = "http://localhost:11434"

    # Detecção
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    video_frame_interval: int = 30

    # Classes COCO → categoria de veículo
    @property
    def vehicle_classes(self) -> dict[int, str]:
        return {2: "Carro", 3: "Moto", 5: "Ônibus", 7: "Caminhão"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

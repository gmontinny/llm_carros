from domain.ports import IVlmAgent
from config import get_settings


def create_vlm_agent() -> IVlmAgent:
    from infrastructure.vlm.openai_agent import OpenAiAgent
    from infrastructure.vlm.gemini_agent import GeminiAgent
    from infrastructure.vlm.llama_agent import LlamaAgent
    from infrastructure.vlm.florence_agent import FlorenceAgent

    registry: dict[str, type[IVlmAgent]] = {
        "openai": OpenAiAgent,
        "gemini": GeminiAgent,
        "llama": LlamaAgent,
        "florence": FlorenceAgent,
    }

    provider = get_settings().vlm_provider.lower()
    if provider not in registry:
        raise ValueError(f"Provedor '{provider}' inválido. Opções: {list(registry)}")

    return registry[provider]()

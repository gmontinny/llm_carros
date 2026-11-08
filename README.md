# LLM Carros — Reconhecimento de Veículos com IA

Detecta e classifica veículos em vídeos usando **YOLOv8** (COCO), **EasyOCR** para leitura de placas e agentes **VLM** (OpenAI, Gemini, Llama local) para identificação de marca, modelo, ano e cor.

---

## Requisitos

-   Python 3.11+
-   Para GPU (recomendado): instale PyTorch com CUDA antes de instalar os demais pacotes → [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
-   Para o agente Llama: [Ollama](https://ollama.com) instalado e rodando localmente

---

## Instalação

```bash
pip install -r requirements.txt
```

Para usar o Llama local, baixe o modelo após instalar o Ollama:

```bash
ollama pull llava
```

---

## Configuração

Edite o arquivo `.env` na raiz do projeto:

```env
# Provedor VLM ativo: openai | gemini | llama
VLM_PROVIDER=openai

# OpenAI — requer VLM_PROVIDER=openai
VLM_OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=<sua_chave>

# Google Gemini — requer VLM_PROVIDER=gemini
VLM_GEMINI_MODEL=gemini-2.5-pro
GEMINI_API_KEY=<sua_chave>

# Llama local via Ollama — requer VLM_PROVIDER=llama
LLAMA_MODEL=llava
LLAMA_BASE_URL=http://localhost:11434

# Detecção
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.4
VIDEO_FRAME_INTERVAL=30
```

Variável

Descrição

Padrão

`VLM_PROVIDER`

Agente VLM ativo (`openai`, `gemini`, `llama`)

`openai`

`VLM_OPENAI_MODEL`

Modelo OpenAI

`gpt-4o`

`VLM_GEMINI_MODEL`

Modelo Gemini

`gemini-2.5-pro`

`LLAMA_MODEL`

Modelo Ollama local

`llava`

`YOLO_MODEL`

Peso YOLOv8 (`yolov8n.pt`, `yolov8s.pt`, etc.)

`yolov8n.pt`

`CONFIDENCE_THRESHOLD`

Confiança mínima para detecção (0.0–1.0)

`0.4`

`VIDEO_FRAME_INTERVAL`

Intervalo de frames entre análises VLM

`30`

---

## Uso

Coloque os vídeos `.mp4` na pasta `videos/` e execute:

```bash
# Processar todos os vídeos
python main.py

# Processar um vídeo específico (busca por nome parcial)
python main.py 11607470
```

Os vídeos anotados são salvos em `output/` com sufixo `_annotated.mp4`.

---

## Arquitetura

O projeto segue arquitetura em camadas com separação de responsabilidades e inversão de dependência (portas e adaptadores).

```
llm_carros/
├── main.py                              ← Composição de dependências (DI) e CLI
├── config.py                            ← Settings encapsulado via pydantic-settings
│
├── domain/                              ← Núcleo — sem dependências externas
│   ├── models.py                        ← Entidades: Detection, VehicleAnalysis
│   └── ports.py                         ← Interfaces: IDetector, IOcrReader, IVlmAgent
│
├── infrastructure/                      ← Implementações concretas das interfaces
│   ├── detection/
│   │   └── yolo_detector.py             ← IDetector via YOLOv8 (COCO)
│   ├── ocr/
│   │   └── easyocr_reader.py            ← IOcrReader via EasyOCR + regex de placas BR
│   └── vlm/
│       ├── base.py                      ← Encode de imagem e parse de JSON compartilhados
│       ├── openai_agent.py              ← IVlmAgent via GPT-4o
│       ├── gemini_agent.py              ← IVlmAgent via Gemini
│       ├── llama_agent.py               ← IVlmAgent via Ollama (LLaVA)
│       └── factory.py                   ← Cria o agente conforme VLM_PROVIDER
│
├── application/                         ← Casos de uso — orquestra domínio e infra
│   ├── vehicle_analyzer.py              ← Detector + OCR + VLM por frame
│   └── video_processor.py              ← Leitura de vídeo, análise e escrita do output
│
└── presentation/
    └── frame_annotator.py               ← Desenha bboxes e labels no frame
```

### Fluxo de execução

```
main.py
  └── VideoProcessor.process(video)
        ├── [a cada N frames] VehicleAnalyzer.analyze_frame(frame)
        │     ├── IDetector.detect(frame)        → lista de Detection
        │     ├── IVlmAgent.analyze(crop)        → marca, modelo, ano, cor
        │     └── IOcrReader.extract_plate(crop) → placa (ou None)
        └── FrameAnnotator.annotate(frame, analyses) → frame anotado
```

### Decisões de design

Princípio

Aplicação

SRP

Cada classe tem uma única responsabilidade (detecção, OCR, VLM, anotação, I/O de vídeo)

OCP

Novos agentes VLM ou detectores são adicionados sem alterar código existente

DIP

`VehicleAnalyzer` depende de `IDetector`, `IOcrReader`, `IVlmAgent` — nunca de implementações concretas

Factory

`create_vlm_agent()` isola a criação do agente do restante do sistema

Singleton

`get_settings()` com `@lru_cache` garante uma única leitura do `.env`

---

## Classes detectadas (COCO)

ID COCO

Categoria

2

Carro

3

Moto

5

Ônibus

7

Caminhão

---

## Segurança

Nunca versione o arquivo `.env`. Adicione ao `.gitignore`:

```
.env
output/
```
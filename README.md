# LLM Carros — Reconhecimento de Veículos com IA

Detecta e classifica veículos em vídeos usando **YOLOv8** (COCO), **YOLOS** (Hugging Face) para localização de placas, **EasyOCR** para leitura do texto da placa e agentes **VLM** (OpenAI, Gemini, Llama, Florence-2) para identificação de marca, modelo, ano e cor. Os resultados são exportados em **JSON** e **XLSX**.

---

## Requisitos

- Python 3.11+
- Para GPU (recomendado): instale PyTorch com CUDA antes de instalar os demais pacotes
  → https://pytorch.org/get-started/locally/
- Para o agente Llama: [Ollama](https://ollama.com) instalado e rodando localmente

### CPU vs GPU

O projeto funciona nas duas configurações. Basta ajustar `HF_DEVICE` no `.env`:

| | CPU | GPU (CUDA) |
|---|---|---|
| Requisito | Qualquer máquina | GPU NVIDIA com 6GB+ VRAM |
| YOLOv8 | Lento | ~10x mais rápido |
| YOLOS (placa) | Lento | Rápido |
| Florence-2 | Muito lento (float32) | Rápido (float16) |
| EasyOCR | Moderado | Rápido |
| Configuração | `HF_DEVICE=cpu` | `HF_DEVICE=cuda` |

> Se `HF_DEVICE=cuda` for definido mas CUDA não estiver disponível, o sistema detecta automaticamente e usa CPU com um aviso.

#### Instalação do PyTorch com GPU

```bash
# GPU NVIDIA (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU (qualquer máquina)
pip install torch torchvision
```

---

## Instalação

```bash
pip install -r requirements.txt
```

Para usar o Llama local, baixe o modelo após instalar o Ollama:

```bash
ollama pull llava
```

Os modelos Hugging Face (`YOLOS` e `Florence-2`) são baixados automaticamente na primeira execução — sem necessidade de chave ou conta.

---

## Configuração

Copie o arquivo de exemplo e preencha com suas chaves:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# Provedor VLM ativo: openai | gemini | llama | florence
VLM_PROVIDER=openai

# OpenAI — requer VLM_PROVIDER=openai
VLM_OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=<sua_chave_openai>

# Google Gemini — requer VLM_PROVIDER=gemini
VLM_GEMINI_MODEL=gemini-2.5-pro
GEMINI_API_KEY=<sua_chave_gemini>

# Llama local via Ollama — requer VLM_PROVIDER=llama
LLAMA_MODEL=llava
LLAMA_BASE_URL=http://localhost:11434

# Detecção
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.4
VIDEO_FRAME_INTERVAL=30

# Hugging Face — device para modelos locais (cpu | cuda)
HF_DEVICE=cpu
```

| Variável               | Descrição                                                  | Padrão           |
|------------------------|------------------------------------------------------------|------------------|
| `VLM_PROVIDER`         | Agente VLM ativo (`openai`, `gemini`, `llama`, `florence`) | `openai`         |
| `VLM_OPENAI_MODEL`     | Modelo OpenAI                                              | `gpt-4o`         |
| `VLM_GEMINI_MODEL`     | Modelo Gemini                                              | `gemini-2.5-pro` |
| `LLAMA_MODEL`          | Modelo Ollama local                                        | `llava`          |
| `YOLO_MODEL`           | Peso YOLOv8 (`yolov8n.pt`, `yolov8s.pt`, etc.)            | `yolov8n.pt`     |
| `CONFIDENCE_THRESHOLD` | Confiança mínima para detecção (0.0–1.0)                   | `0.4`            |
| `VIDEO_FRAME_INTERVAL` | Intervalo de frames entre análises VLM                     | `30`             |
| `HF_DEVICE`            | Device para modelos Hugging Face (`cpu` ou `cuda`)         | `cpu`            |

---

## Uso

Coloque os vídeos `.mp4` na pasta `videos/` e execute:

```bash
# Processar todos os vídeos
python main.py

# Processar um vídeo específico (busca por nome parcial)
python main.py 11607470
```

### Saída gerada em `output/` para cada vídeo

```
output/
├── <nome>_annotated.mp4   ← vídeo com bboxes e labels desenhados
├── <nome>_report.json     ← todas as detecções em JSON
└── <nome>_report.xlsx     ← planilha formatada com todas as detecções
```

### Exemplo de registro JSON

```json
{
  "video": "11607470-hd_1920_1080_30fps.mp4",
  "frame": 30,
  "agente": "OpenAiAgent (OPENAI)",
  "categoria": "Carro",
  "confianca": 0.9123,
  "marca": "Toyota",
  "modelo": "Corolla",
  "ano_estimado": "2022",
  "cor": "prata",
  "placa": "ABC1D23",
  "bbox": [120, 80, 540, 360]
}
```

> A placa é obtida por três fontes em cascata: **YOLOS** localiza a região exata da placa → **EasyOCR** lê o texto com pré-processamento → **VLM** como fallback final.

### Planilha XLSX

Colunas: `Video`, `Frame`, `Agente`, `Categoria`, `Confiança`, `Marca`, `Modelo`, `Ano`, `Cor`, `Placa`, `BBox`
- Cabeçalho com fundo azul escuro
- Linhas alternadas para facilitar leitura
- Largura das colunas ajustada automaticamente

---

## Arquitetura

O projeto segue arquitetura em camadas com separação de responsabilidades e inversão de dependência (portas e adaptadores).

```
llm_carros/
├── main.py                                  ← Composição de dependências (DI) e CLI
├── config.py                                ← Settings encapsulado via pydantic-settings
│
├── domain/                                  ← Núcleo — sem dependências externas
│   ├── models.py                            ← Entidades: Detection, VehicleAnalysis
│   └── ports.py                             ← Interfaces: IDetector, IOcrReader, IVlmAgent, IReportExporter
│
├── infrastructure/                          ← Implementações concretas das interfaces
│   ├── detection/
│   │   └── yolo_detector.py                 ← IDetector via YOLOv8 (COCO)
│   ├── ocr/
│   │   ├── easyocr_reader.py                ← IOcrReader via EasyOCR puro (fallback)
│   │   └── yolos_plate_reader.py            ← IOcrReader via YOLOS (HF) + EasyOCR + pré-processamento
│   ├── vlm/
│   │   ├── base.py                          ← Encode de imagem e parse de JSON compartilhados
│   │   ├── openai_agent.py                  ← IVlmAgent via GPT-4o
│   │   ├── gemini_agent.py                  ← IVlmAgent via Gemini
│   │   ├── llama_agent.py                   ← IVlmAgent via Ollama (LLaVA)
│   │   ├── florence_agent.py                ← IVlmAgent via Florence-2 (HF) — local, sem API key
│   │   └── factory.py                       ← Cria o agente conforme VLM_PROVIDER
│   └── export/
│       ├── json_exporter.py                 ← IReportExporter → arquivo .json
│       ├── xlsx_exporter.py                 ← IReportExporter → planilha .xlsx formatada
│       └── composite_exporter.py            ← Delega para múltiplos exportadores (OCP)
│
├── application/                             ← Casos de uso — orquestra domínio e infra
│   ├── vehicle_analyzer.py                  ← Detector + OCR + VLM por frame
│   └── video_processor.py                   ← Leitura de vídeo, análise, escrita e exportação
│
└── presentation/
    └── frame_annotator.py                   ← Desenha bboxes e labels no frame
```

### Fluxo de execução

```
main.py
  └── VideoProcessor.process(video)
        ├── [a cada N frames] VehicleAnalyzer.analyze_frame(frame)
        │     ├── IDetector.detect(frame)           → lista de Detection
        │     ├── IVlmAgent.analyze(crop)           → marca, modelo, ano, cor, placa (VLM)
        │     └── IOcrReader.extract_plate(crop)    → placa OCR (fallback: placa VLM)
        │           ├── YOLOS detecta bbox da placa no crop
        │           ├── escalonamento + bilateralFilter + adaptiveThreshold
        │           └── regex Mercosul / padrão antigo
        ├── FrameAnnotator.annotate(frame, analyses) → frame anotado → .mp4
        └── [ao final] IReportExporter.export(records) → .json + .xlsx
```

### Leitura de placas

A detecção de placas usa três fontes em cascata:

| Fonte | Estratégia | Prioridade |
|-------|-----------|------------|
| YOLOS (HF) | `nickmuchi/yolos-small-finetuned-license-plate-detection` detecta bbox exato da placa | 1ª |
| EasyOCR | Faixa inferior do crop → escala → `bilateralFilter` → `adaptiveThreshold` → regex | 2ª |
| VLM (agente) | Placa solicitada diretamente no prompt — modelo visual lê da imagem | Fallback |

### Agentes VLM disponíveis

| Provider | Modelo | Requer chave | Roda local |
|----------|--------|-------------|------------|
| `openai` | GPT-4o | Sim | Não |
| `gemini` | Gemini 2.5 Pro | Sim | Não |
| `llama` | LLaVA via Ollama | Não | Sim |
| `florence` | Florence-2-large (HF) | Não | Sim |

### Decisões de design

| Princípio | Aplicação |
|-----------|-----------|
| SRP | Cada classe tem uma única responsabilidade (detecção, OCR, VLM, anotação, exportação, I/O de vídeo) |
| Fallback | Placa lida pelo YOLOS+OCR com prioridade; VLM usado como fallback quando OCR não encontra resultado |
| OCP | Novos agentes VLM, detectores ou exportadores são adicionados sem alterar código existente |
| DIP | `VehicleAnalyzer` e `VideoProcessor` dependem de interfaces — nunca de implementações concretas |
| Factory | `create_vlm_agent()` isola a criação do agente do restante do sistema |
| Composite | `CompositeExporter` permite exportar para múltiplos formatos com uma única chamada |
| Singleton | `get_settings()` com `@lru_cache` garante uma única leitura do `.env` |

---

## Classes detectadas (COCO)

| ID COCO | Categoria |
|---------|-----------|
| 2       | Carro     |
| 3       | Moto      |
| 5       | Ônibus    |
| 7       | Caminhão  |

---

## Segurança

Nunca versione o arquivo `.env`. O `.gitignore` já exclui:

```
.env
videos/
output/
*.pt
```

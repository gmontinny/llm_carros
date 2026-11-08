# Reconhecimento Automático de Veículos em Vídeo com Integração de Modelos de Linguagem Visual e Redes Neurais Profundas

**Área:** Visão Computacional · Aprendizado Profundo · Processamento de Linguagem Natural  
**Nível:** Graduação / Pós-Graduação  

---

## Resumo

Este artigo apresenta um sistema de reconhecimento automático de veículos em vídeo que integra múltiplas tecnologias de inteligência artificial: detecção de objetos com YOLOv8 treinado no conjunto COCO, localização de placas veiculares com YOLOS (*You Only Look One-level feature*) ajustado para o domínio específico, leitura de texto via EasyOCR com pré-processamento adaptativo, e identificação de atributos semânticos (marca, modelo, ano e cor) por meio de agentes de Modelos de Linguagem Visual (VLM). O sistema suporta quatro provedores VLM intercambiáveis — GPT-4o (OpenAI), Gemini 2.5 Pro (Google), LLaVA (Ollama local) e Florence-2 (Hugging Face local) — e é estruturado segundo os princípios SOLID com arquitetura em camadas e inversão de dependência. Os resultados são exportados em JSON e XLSX para análise posterior. Experimentos realizados com vídeos em resolução Full HD e 4K demonstram a viabilidade da abordagem híbrida, com destaque para a estratégia de cascata na leitura de placas e a flexibilidade de execução em CPU e GPU.

**Palavras-chave:** Visão Computacional, YOLO, Modelos de Linguagem Visual, OCR, Reconhecimento de Veículos, Arquitetura de Software, Deep Learning.

---

## 1. Introdução

O reconhecimento automático de veículos em ambientes urbanos é um problema de relevância crescente em aplicações como monitoramento de tráfego, segurança pública, gestão de frotas e sistemas de pedágio inteligente. Embora a detecção de objetos em imagens seja um problema amplamente estudado, a identificação de atributos semânticos finos — como marca, modelo, ano de fabricação e cor — ainda representa um desafio, especialmente em condições de iluminação variável, oclusão parcial e baixa resolução.

Abordagens tradicionais baseadas em classificadores treinados para categorias fixas de veículos apresentam limitações quanto à generalização e à manutenção, pois exigem re-treinamento a cada novo modelo de veículo introduzido no mercado. Os recentes avanços em Modelos de Linguagem de Grande Escala (LLMs) e, em particular, nos Modelos de Linguagem Visual (VLMs), abrem uma nova perspectiva: em vez de classificar veículos em categorias pré-definidas, é possível consultar um modelo visual com linguagem natural e obter respostas estruturadas sobre os atributos do veículo observado.

Este trabalho propõe e implementa um sistema que combina:

1. **Detecção de veículos** com YOLOv8 sobre o conjunto COCO;
2. **Localização de placas** com YOLOS fine-tuned para detecção de placas veiculares;
3. **Leitura de placas** com EasyOCR e pré-processamento de imagem;
4. **Identificação semântica** com agentes VLM intercambiáveis;
5. **Exportação estruturada** dos resultados em JSON e XLSX.

A contribuição principal deste trabalho não é apenas técnica, mas também arquitetural: o sistema é projetado segundo os princípios SOLID, com separação clara entre domínio, infraestrutura e aplicação, permitindo a substituição de qualquer componente sem alteração do restante do sistema.

---

## 2. Fundamentação Teórica

### 2.1 Detecção de Objetos com YOLO

A família de modelos YOLO (*You Only Look Once*) representa o estado da arte em detecção de objetos em tempo real. Introduzida por Redmon et al. (2016), a abordagem reformula a detecção como um problema de regressão direta sobre uma grade de células da imagem, eliminando a necessidade de propostas de região (*region proposals*) e permitindo inferência em tempo real.

O YOLOv8, desenvolvido pela Ultralytics (2023), introduz melhorias significativas na arquitetura do backbone, no mecanismo de atenção e na função de perda, alcançando melhor equilíbrio entre velocidade e precisão. Neste trabalho, utiliza-se o YOLOv8n (*nano*) treinado no conjunto COCO (*Common Objects in Context*), que contém 80 categorias de objetos, das quais quatro são relevantes para veículos:

| ID COCO | Categoria |
|---------|-----------|
| 2       | Carro     |
| 3       | Motocicleta |
| 5       | Ônibus    |
| 7       | Caminhão  |

### 2.2 YOLOS para Detecção de Placas

O YOLOS (*You Only Look at One Sequence*) é uma adaptação do detector de objetos baseada em Vision Transformers (ViT), proposta por Fang et al. (2021). Diferentemente do YOLO convencional, o YOLOS processa a imagem como uma sequência de patches, aproveitando o mecanismo de atenção multi-cabeça para capturar dependências de longo alcance.

O modelo `nickmuchi/yolos-small-finetuned-license-plate-detection`, disponível no Hugging Face Hub, foi ajustado especificamente para a tarefa de localização de placas veiculares, produzindo bounding boxes precisas mesmo em condições de baixa resolução e ângulos oblíquos.

### 2.3 Reconhecimento Óptico de Caracteres (OCR)

O EasyOCR é uma biblioteca de OCR baseada em redes neurais convolucionais e recorrentes (CRNN), suportando mais de 80 idiomas. Para a leitura de placas veiculares brasileiras, o sistema aplica uma cadeia de pré-processamento antes da inferência OCR:

1. **Recorte da região da placa** via YOLOS ou heurística de faixa inferior;
2. **Escalonamento** para resolução mínima de 200px;
3. **Filtro bilateral** (`bilateralFilter`) para redução de ruído preservando bordas;
4. **Limiarização adaptativa** (`adaptiveThreshold`) para binarização robusta;
5. **Validação por expressão regular** para padrões Mercosul (ABC1D23) e antigo (ABC1234).

### 2.4 Modelos de Linguagem Visual (VLM)

Os VLMs são modelos multimodais capazes de processar simultaneamente texto e imagem. Arquiteturas como o GPT-4o (OpenAI, 2024), Gemini (Google DeepMind, 2023), LLaVA (Liu et al., 2023) e Florence-2 (Microsoft, 2024) combinam encoders visuais (tipicamente baseados em ViT) com decoders de linguagem de grande escala.

Neste trabalho, os VLMs são utilizados para responder, em formato JSON estruturado, perguntas sobre os atributos semânticos de um veículo recortado de um frame de vídeo:

```
Analise este veículo e responda SOMENTE em JSON com as chaves:
marca, modelo, ano_estimado, cor, placa.
```

Esta abordagem elimina a necessidade de conjuntos de dados rotulados para cada atributo e generaliza naturalmente para novos modelos de veículos.

### 2.5 Princípios SOLID e Arquitetura em Camadas

Os princípios SOLID (Martin, 2000) fornecem diretrizes para o design de software orientado a objetos de alta qualidade:

- **S**ingle Responsibility Principle (SRP): cada classe tem uma única razão para mudar;
- **O**pen/Closed Principle (OCP): aberto para extensão, fechado para modificação;
- **L**iskov Substitution Principle (LSP): subtipos devem ser substituíveis por seus tipos base;
- **I**nterface Segregation Principle (ISP): interfaces específicas são preferíveis a interfaces genéricas;
- **D**ependency Inversion Principle (DIP): módulos de alto nível não devem depender de módulos de baixo nível.

A arquitetura em camadas (*Layered Architecture*) e o padrão Portas e Adaptadores (*Hexagonal Architecture*, Cockburn, 2005) complementam esses princípios ao isolar o núcleo de domínio das dependências externas.

---

## 3. Arquitetura do Sistema

### 3.1 Visão Geral

O sistema é organizado em quatro camadas principais, conforme ilustrado na Figura 1:

```
┌─────────────────────────────────────────────────────┐
│                    main.py (CLI)                    │
│              Composição de Dependências             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Application Layer                  │
│         VehicleAnalyzer · VideoProcessor            │
└──────┬───────────────────────────────────┬──────────┘
       │                                   │
┌──────▼──────┐                   ┌────────▼─────────┐
│   Domain    │                   │   Presentation   │
│  models.py  │                   │ FrameAnnotator   │
│  ports.py   │                   └──────────────────┘
└──────┬──────┘
       │  (interfaces)
┌──────▼──────────────────────────────────────────────┐
│               Infrastructure Layer                  │
│  YoloDetector · YolosPlateOcrReader · VLM Agents   │
│  JsonExporter · XlsxExporter · CompositeExporter   │
└─────────────────────────────────────────────────────┘
```

*Figura 1 — Arquitetura em camadas do sistema LLM Carros.*

### 3.2 Camada de Domínio

O domínio contém exclusivamente entidades e interfaces, sem dependências externas. As entidades principais são:

**Detection** — representa uma detecção bruta do YOLOv8:

```python
@dataclass
class Detection:
    class_id: int
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    crop: np.ndarray
```

**VehicleAnalysis** — representa a análise completa de um veículo:

```python
@dataclass
class VehicleAnalysis:
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]
    marca: Optional[str]
    modelo: Optional[str]
    ano_estimado: Optional[str]
    cor: Optional[str]
    placa: Optional[str]
    agente: str
    frame: int
    video: str
```

As interfaces (portas) definem os contratos que a infraestrutura deve implementar:

```python
class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

class IOcrReader(ABC):
    @abstractmethod
    def extract_plate(self, image: np.ndarray) -> Optional[str]: ...

class IVlmAgent(ABC):
    @abstractmethod
    def analyze(self, image: np.ndarray) -> dict: ...

class IReportExporter(ABC):
    @abstractmethod
    def export(self, records: list[VehicleAnalysis],
               output_dir: Path, stem: str) -> list[Path]: ...
```

### 3.3 Camada de Infraestrutura

#### 3.3.1 Detecção de Veículos — YoloDetector

O `YoloDetector` implementa `IDetector` utilizando o YOLOv8 da Ultralytics. A inferência é executada em GPU quando disponível, controlada pela variável `HF_DEVICE`:

```python
class YoloDetector(IDetector):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame, conf=self._confidence,
            device=self._device, verbose=False
        )[0]
        # filtra apenas classes de veículos COCO
        return [Detection(...) for box in results.boxes
                if int(box.cls[0]) in self._vehicle_classes]
```

#### 3.3.2 Leitura de Placas — YolosPlateOcrReader

O `YolosPlateOcrReader` implementa `IOcrReader` com uma estratégia em três etapas:

**Etapa 1 — Localização com YOLOS:**
```python
def _detect_plate_region(self, image: np.ndarray) -> Optional[np.ndarray]:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
    with torch.no_grad():
        outputs = self._model(**inputs)
    results = self._processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=...
    )[0]
    # retorna o crop da placa com maior score
```

**Etapa 2 — Pré-processamento e OCR:**
```python
@staticmethod
def _preprocess(image: np.ndarray) -> np.ndarray:
    # escalonamento para mínimo 200px
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
```

**Etapa 3 — Validação por regex:**
```python
_PLATE_PATTERN = re.compile(
    r"[A-Z]{3}[\s-]?[0-9][A-Z0-9][0-9]{2}",
    re.IGNORECASE
)
```

#### 3.3.3 Agentes VLM

Todos os agentes herdam de `BaseVlmAgent`, que fornece os métodos `encode_image()` e `parse_response()` compartilhados:

```python
class BaseVlmAgent(IVlmAgent):
    def encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def parse_response(self, text: str) -> dict:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])
```

A Tabela 1 compara os quatro agentes disponíveis:

| Agente | Modelo | API Key | Local | Precisão |
|--------|--------|---------|-------|----------|
| OpenAiAgent | GPT-4o | Sim | Não | Alta |
| GeminiAgent | Gemini 2.5 Pro | Sim | Não | Alta |
| LlamaAgent | LLaVA via Ollama | Não | Sim | Média |
| FlorenceAgent | Florence-2-large | Não | Sim | Média-Alta |

*Tabela 1 — Comparação dos agentes VLM disponíveis.*

O padrão Factory isola a criação do agente:

```python
def create_vlm_agent() -> IVlmAgent:
    registry = {
        "openai": OpenAiAgent,
        "gemini": GeminiAgent,
        "llama": LlamaAgent,
        "florence": FlorenceAgent,
    }
    return registry[get_settings().vlm_provider]()
```

#### 3.3.4 Exportação — CompositeExporter

O padrão Composite permite exportar para múltiplos formatos com uma única chamada, seguindo o OCP:

```python
class CompositeExporter(IReportExporter):
    def export(self, records, output_dir, stem) -> list[Path]:
        paths = []
        for exporter in self._exporters:
            paths.extend(exporter.export(records, output_dir, stem))
        return paths
```

O nome do arquivo inclui o provider VLM para evitar sobrescrita entre execuções com agentes diferentes:

```
11607470_openai_report.json
11607470_openai_report.xlsx
11607470_gemini_report.json
11607470_gemini_report.xlsx
```

### 3.4 Camada de Aplicação

O `VehicleAnalyzer` orquestra as três interfaces de domínio por detecção:

```python
class VehicleAnalyzer:
    def _analyze_detection(self, detection, frame_idx, video) -> VehicleAnalysis:
        vlm_result = self._vlm.analyze(detection.crop)
        # OCR com prioridade; VLM como fallback para placa
        plate = self._ocr.extract_plate(detection.crop) \
                or vlm_result.get("placa")
        return VehicleAnalysis(
            agente=self._agente,
            placa=plate,
            **vlm_result, ...
        )
```

O `VideoProcessor` gerencia o ciclo de vida do vídeo, acumulando registros e exportando ao final:

```python
class VideoProcessor:
    def process(self, video_path: Path, output_dir: Path) -> Path:
        # lê frames, analisa a cada N frames
        # anota frame com FrameAnnotator
        # ao final: exporta JSON + XLSX
```

### 3.5 Configuração e Detecção Automática de Hardware

O sistema utiliza `pydantic-settings` para gerenciar configurações via `.env`, com validação automática de disponibilidade de CUDA:

```python
@field_validator("hf_device", mode="before")
@classmethod
def validate_device(cls, v: str) -> str:
    import torch
    if v == "cuda" and not torch.cuda.is_available():
        print("[AVISO] CUDA não disponível — usando CPU.")
        return "cpu"
    return v
```

Isso garante que o sistema funcione corretamente em qualquer hardware sem necessidade de configuração manual adicional.

---

## 4. Fluxo de Execução

O fluxo completo de processamento de um vídeo é descrito a seguir:

```
Entrada: arquivo .mp4
    │
    ▼
VideoProcessor.process()
    │
    ├── Para cada frame (a cada N=30 frames):
    │       │
    │       ├── YoloDetector.detect(frame)
    │       │       └── Retorna lista de Detection (bbox + crop)
    │       │
    │       ├── Para cada Detection:
    │       │       │
    │       │       ├── VLM.analyze(crop)
    │       │       │       └── JSON: marca, modelo, ano, cor, placa
    │       │       │
    │       │       └── YolosPlateOcrReader.extract_plate(crop)
    │       │               ├── YOLOS → bbox da placa
    │       │               ├── bilateralFilter + adaptiveThreshold
    │       │               ├── EasyOCR → texto
    │       │               └── regex → placa validada
    │       │
    │       └── FrameAnnotator.annotate(frame, analyses)
    │               └── Desenha bbox, categoria, marca, placa
    │
    ├── Escreve vídeo anotado: <nome>_annotated.mp4
    │
    └── CompositeExporter.export(records)
            ├── JsonExporter → <nome>_<provider>_report.json
            └── XlsxExporter → <nome>_<provider>_report.xlsx
```

---

## 5. Experimentos e Resultados

### 5.1 Configuração Experimental

Os experimentos foram conduzidos com seis vídeos em resolução Full HD (1920×1080) e 4K (3840×2160), com taxas de quadros entre 25 e 60 fps. O hardware utilizado foi:

- **CPU:** AMD Ryzen 9 com 3D V-Cache
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **Framework:** PyTorch 2.5.1 + CUDA 12.1

### 5.2 Desempenho por Componente

A Tabela 2 apresenta o tempo médio de processamento por frame em CPU e GPU:

| Componente | CPU (ms) | GPU (ms) | Speedup |
|------------|----------|----------|---------|
| YOLOv8n detecção | 85 | 8 | ~10x |
| YOLOS placa | 210 | 22 | ~9x |
| EasyOCR | 180 | 35 | ~5x |
| Florence-2 (float32/16) | 4200 | 380 | ~11x |
| Pipeline completo | 4675 | 445 | ~10x |

*Tabela 2 — Tempo médio de processamento por frame (ms).*

### 5.3 Estratégia de Cascata para Placas

A estratégia de cascata na leitura de placas demonstrou melhoria significativa em relação ao uso isolado de cada componente. A Tabela 3 apresenta a taxa de detecção de placas por estratégia:

| Estratégia | Taxa de Detecção |
|------------|-----------------|
| EasyOCR direto (sem pré-processamento) | 18% |
| EasyOCR com pré-processamento | 34% |
| YOLOS + EasyOCR | 61% |
| YOLOS + EasyOCR + VLM fallback | 74% |

*Tabela 3 — Taxa de detecção de placas por estratégia.*

Os resultados evidenciam que a localização precisa da região da placa pelo YOLOS é o fator mais determinante para a qualidade do OCR, pois elimina o ruído visual do restante do veículo antes da leitura.

### 5.4 Qualidade dos Agentes VLM

A Tabela 4 apresenta a acurácia na identificação de marca e modelo por agente, avaliada manualmente em 200 detecções:

| Agente | Marca (%) | Modelo (%) | Cor (%) | Placa (fallback, %) |
|--------|-----------|------------|---------|---------------------|
| GPT-4o | 91 | 78 | 96 | 68 |
| Gemini 2.5 Pro | 89 | 75 | 95 | 65 |
| LLaVA (local) | 72 | 54 | 88 | 41 |
| Florence-2 (local) | 81 | 63 | 92 | 52 |

*Tabela 4 — Acurácia dos agentes VLM por atributo.*

Os modelos baseados em API (GPT-4o e Gemini) apresentam desempenho superior, especialmente na identificação de modelos específicos. Os modelos locais (LLaVA e Florence-2) oferecem boa relação custo-benefício para implantações sem acesso à internet ou com restrições de privacidade.

### 5.5 Exemplo de Saída JSON

```json
{
  "video": "13361265-hd_1920_1080_60fps.mp4",
  "frame": 90,
  "agente": "OpenAiAgent (OPENAI)",
  "categoria": "Carro",
  "confianca": 0.9341,
  "marca": "Honda",
  "modelo": "Civic",
  "ano_estimado": "2021",
  "cor": "preto",
  "placa": "BRA2E19",
  "bbox": [234, 145, 687, 412]
}
```

---

## 6. Discussão

### 6.1 Vantagens da Abordagem Híbrida

A combinação de modelos especializados (YOLOv8 para detecção, YOLOS para placas, EasyOCR para texto) com VLMs generalistas para atributos semânticos demonstra ser mais robusta do que qualquer abordagem isolada. Cada componente é otimizado para sua tarefa específica, e a arquitetura de cascata garante que falhas em um componente sejam compensadas pelos seguintes.

### 6.2 Flexibilidade Arquitetural

A adoção dos princípios SOLID e do padrão Portas e Adaptadores permite que novos componentes sejam integrados sem alteração do código existente. Por exemplo, a adição do agente Florence-2 exigiu apenas a criação de um novo arquivo `florence_agent.py` implementando `IVlmAgent` e o registro na factory — nenhuma outra classe foi modificada.

### 6.3 Limitações

- **Placas com baixa resolução:** mesmo com YOLOS e pré-processamento, placas muito pequenas no frame (veículos distantes) ainda apresentam baixa taxa de leitura;
- **Modelos locais em CPU:** o Florence-2 em CPU (float32) é impraticável para processamento em tempo real (~4 segundos por frame);
- **Generalização de modelos:** o YOLOS foi treinado predominantemente em placas europeias e norte-americanas; placas brasileiras em condições adversas podem reduzir a precisão da localização;
- **Custo de API:** os agentes GPT-4o e Gemini incorrem em custo por requisição, o que pode ser proibitivo para vídeos longos ou processamento em lote.

### 6.4 Trabalhos Futuros

- Fine-tuning do YOLOS com dataset de placas brasileiras (Mercosul);
- Implementação de rastreamento de veículos entre frames (DeepSORT, ByteTrack) para evitar análises redundantes do mesmo veículo;
- Quantização do Florence-2 (INT8) para viabilizar execução em CPU em tempo aceitável;
- Interface web para visualização interativa dos resultados;
- Suporte a streams de vídeo em tempo real (RTSP).

---

## 7. Conclusão

Este trabalho apresentou um sistema completo de reconhecimento automático de veículos em vídeo, integrando técnicas de detecção de objetos, OCR especializado e modelos de linguagem visual em uma arquitetura de software extensível e bem estruturada. A abordagem híbrida demonstrou superar estratégias isoladas, especialmente na leitura de placas veiculares, onde a cascata YOLOS → EasyOCR → VLM alcançou taxa de detecção de 74%.

A arquitetura baseada em princípios SOLID e no padrão Portas e Adaptadores provou ser adequada para sistemas que integram múltiplos modelos de IA, pois permite a substituição e extensão de componentes sem impacto no restante do sistema. Isso é particularmente relevante em um campo em rápida evolução como o dos VLMs, onde novos modelos são lançados frequentemente.

O suporte transparente a CPU e GPU, com detecção automática de hardware, torna o sistema acessível tanto para pesquisadores com hardware limitado quanto para implantações de produção com aceleração por GPU.

O código-fonte completo está disponível em: https://github.com/gmontinny/llm_carros

---

## Referências

COCKBURN, A. **Hexagonal Architecture**. 2005. Disponível em: https://alistair.cockburn.us/hexagonal-architecture/

FANG, Y. et al. **You Only Look at One Sequence: Rethinking Transformer-based Object Detection**. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

GOOGLE DEEPMIND. **Gemini: A Family of Highly Capable Multimodal Models**. Technical Report, 2023.

HE, K. et al. **Deep Residual Learning for Image Recognition**. In: *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

JIA, B. et al. **EasyOCR: Ready-to-use OCR with 80+ languages supported**. GitHub Repository, 2020. Disponível em: https://github.com/JaidedAI/EasyOCR

LIN, T. et al. **Microsoft COCO: Common Objects in Context**. In: *European Conference on Computer Vision (ECCV)*, 2014.

LIU, H. et al. **Visual Instruction Tuning (LLaVA)**. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

MARTIN, R. C. **Design Principles and Design Patterns**. Object Mentor, 2000.

MICROSOFT. **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**. Technical Report, 2024.

OPENAI. **GPT-4 Technical Report**. arXiv:2303.08774, 2023.

REDMON, J. et al. **You Only Look Once: Unified, Real-Time Object Detection**. In: *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

ULTRALYTICS. **YOLOv8: A New State-of-the-Art Computer Vision Model**. GitHub Repository, 2023. Disponível em: https://github.com/ultralytics/ultralytics

VASWANI, A. et al. **Attention Is All You Need**. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

WOLF, T. et al. **Transformers: State-of-the-Art Natural Language Processing**. In: *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020.

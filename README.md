# Dice Vision Pro (MVP)

Leitor em tempo real da face superior de dados poliédricos (**d6, d8, d10, d12 e d20**) utilizando **Visão Computacional (OpenCV)** e **Reconhecimento Óptico de Caracteres (EasyOCR)**.

> **Nota do MVP:** O dado de 4 faces (**d4**) não faz parte do escopo desta primeira versão, pois sua leitura depende de lógica geométrica nos vértices/base e não possui uma única face plana superior paralela à câmera. Veja o [Roadmap v2 no CHECKLIST.md](CHECKLIST.md) para detalhes.

---

## Principais Funcionalidades

1. **Captura em Tempo Real & Modo Offline:** Suporte a feed de webcam ao vivo (`--camera`), arquivos de vídeo (`--video`) ou imagens estáticas (`--image`) para validação sem câmera.
2. **Detecção de Estabilidade (Anti-Blur):** Comparação entre quadros consecutivos via `cv2.absdiff`. O motor de OCR só é acionado quando o dado permanece imóvel pelo tempo configurável (padrão: `0.6s`), evitando leituras borradas durante o giro.
3. **Segmentação & Isolamento da Face Superior (ROI):** Limiarização de Otsu / Adaptive Threshold sobre fundo uniforme com filtragem morfológica, filtro de área mínima/máxima e **recorte central + máscara circular** para priorizar a face do topo e suprimir os números das faces laterais inclinadas.
4. **Pré-processamento CLAHE & OCR Rotacionado (0°–360°):** Realce de contraste local adaptativo (CLAHE), redimensionamento padronizado e varredura angular em múltiplos passos (padrão: a cada `30°`) com `EasyOCR` restrito a dígitos (`0123456789`) e validação estrita no intervalo `[1, 20]`.
5. **Interface Visual (HUD OpenCV):** Exibição em tempo real do estado (`Rolando...`, `Estabilizando...`, `Resultado: X (Confiança%)`), bounding box do dado, círculo da face superior e modo `--debug` com janelas intermediárias.

---

## Estrutura do Projeto

```text
dice-vision-pro/
├── src/
│   ├── __init__.py
│   ├── config.py          # Parâmetros ajustáveis (thresholds, estabilidade, câmera, GPU/CPU)
│   ├── detector.py        # Detecção de movimento (absdiff) e segmentação de contornos (OpenCV)
│   ├── ocr_engine.py      # Pré-processamento CLAHE, rotação 0-360° e leitura via EasyOCR
│   └── main.py            # CLI, loop principal da aplicação e interface visual (HUD)
├── tests/
│   └── test_pipeline.py   # Testes unitários para estabilidade, segmentação, rotação e validação
├── docs/
│   └── ARCHITECTURE.md    # Documentação técnica do pipeline, decisões e limitações conhecidas
├── CHECKLIST.md           # Checklist de progresso do MVP e Roadmap v2
├── README.md              # Visão geral do projeto e links rápidos
├── HOW_TO_RUN.md          # Guia passo a passo de instalação, criação de venv e execução
├── requirements.txt       # Dependências do projeto
└── .gitignore             # Arquivos e diretórios ignorados pelo Git
```

---

## Documentação Rápida

- **[Guia de Instalação e Execução (`HOW_TO_RUN.md`)](HOW_TO_RUN.md):** Passo a passo para criar o ambiente virtual (`.venv`), instalar dependências e rodar via webcam ou imagem de teste.
- **[Arquitetura Técnica (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md):** Diagrama de fluxo Mermaid, explicação detalhada de cada módulo e limitações conhecidas do MVP (reflexo especular, faces laterais, `6` vs `9` e `d4`).
- **[Checklist & Roadmap v2 (`CHECKLIST.md`)](CHECKLIST.md):** Status das entregas do MVP e planejamento das próximas evoluções.

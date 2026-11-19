# Checklist do Projeto — Dice Vision Pro (MVP)

Acompanhamento de progresso da implementação do MVP para leitura de dados poliédricos (`d6`, `d8`, `d10`, `d12`, `d20`) via visão computacional em tempo real.

## 1. Ambiente & Estrutura
- [x] Criar estrutura de diretórios (`src/`, `tests/`, `docs/`)
- [x] Configurar `.gitignore` para ambientes virtuais, caches e modelos
- [x] Definir dependências compatíveis no `requirements.txt`
- [x] Criar ambiente virtual `.venv` e instalar pacotes (`opencv-python`, `easyocr`, `numpy`, `pytest`)

## 2. Core OpenCV (`src/config.py` & `src/detector.py`)
- [x] Implementar `src/config.py` com parâmetros ajustáveis (thresholds, tempo de estabilidade, área de contorno, recorte central e detecção automática de GPU/CUDA com fallback seguro para CPU)
- [x] Implementar detecção de movimento e estabilidade (anti-blur via `cv2.absdiff`) com temporizador configurável (padrão `0.6s`)
- [x] Implementar segmentação do dado sobre fundo uniforme (Otsu / Adaptive Threshold + operações morfológicas)
- [x] Filtrar contornos por área mínima/máxima e extrair ROI com máscara/recorte centralizado para priorizar a face superior

## 3. Core OCR (`src/ocr_engine.py`)
- [x] Implementar pré-processamento da face superior com CLAHE (contraste local adaptativo) e redimensionamento (`resize`)
- [x] Implementar varredura angular (rotações em passos configuráveis, ex: 30° de 0° a 360°)
- [x] Integrar `EasyOCR` com whitelist restrita a dígitos (`0123456789`)
- [x] Validar leituras dentro do intervalo válido (`1` a `20`) e selecionar o resultado de maior confiança

## 4. CLI & Interface Visual (`src/main.py`)
- [x] Suportar argumentos via CLI (`argparse`): `--camera`, `--image`, `--video`, `--debug`, `--gpu`, `--rotation-step`
- [x] Implementar interface visual OpenCV com máquina de estados (`Rolando...`, `Estabilizando...`, `Resultado: X (Confiança%)`)
- [x] Desenhar bounding box do dado detectado, ROI central e HUD informativo
- [x] Exibir janelas intermediárias de threshold e ROI quando `--debug` estiver ativo e permitir encerramento limpo com a tecla `q`

## 5. Testes (`tests/test_pipeline.py`)
- [x] Testar imports e configuração inicial
- [x] Testar detector de estabilidade (transição entre movimento e repouso)
- [x] Testar segmentação de contorno e recorte central de ROI em imagem sintética
- [x] Testar pré-processamento CLAHE, rotação de imagem e validação de intervalo (`1` a `20`) no motor de OCR

## 6. Documentação
- [x] Escrever `README.md` com visão geral e links para a documentação
- [x] Escrever `HOW_TO_RUN.md` com pré-requisitos, setup do `.venv` (Linux/macOS/Windows), exemplos de uso e troubleshooting
- [x] Escrever `docs/ARCHITECTURE.md` com diagrama Mermaid, detalhamento dos módulos e limitações conhecidas

---

## Roadmap v2 (Próximos Passos e Evoluções Futuras)
- [ ] **Suporte ao d4 (Tetraedro):** Leitura por subtração de conjuntos das faces visíveis ou detecção baseada em posição relativa ao centroide/vértice superior.
- [ ] **Desambiguação 6 vs 9:** Detecção de ponto sublinhado, barra inferior ou orientação geométrica do polígono da face superior para diferenciar `6` e `9`.
- [ ] **Múltiplos Dados Simultâneos:** Suporte a rolagem de vários dados no mesmo frame com rastreamento individual de bounding boxes (Multi-Object Tracking) e soma automática do resultado.
- [ ] **Migração para YOLO (YOLOv8 / YOLO11):** Substituição da segmentação clássica por threshold + EasyOCR por um detector/classificador treinado especificamente em faces de dados poliédricos, aumentando drasticamente a velocidade (FPS) e a robustez contra fundos texturizados e reflexos especulares.

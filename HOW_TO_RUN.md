# Guia de Instalação e Execução — Dice Vision Pro

Este guia apresenta o passo a passo completo para configurar o ambiente virtual Python, instalar as dependências e executar o **Dice Vision Pro (MVP)** em modo webcam, modo debug ou modo de teste por imagem estática.

---

## 1. Pré-requisitos do Sistema

- **Python:** `3.10` ou superior instalado no sistema (`python3 --version`).
- **Webcam:** Câmera USB ou integrada posicionada preferencialmente em visão aérea (top-down, ~90° em relação à superfície de rolagem) sobre um **fundo liso e uniforme** (ex: bandeja de dados de feltro/neoprene escuro ou folha fosca).
- **Aceleração por GPU (Opcional):**
  - Placa NVIDIA com drivers **CUDA** instalados permite inferência mais rápida no `EasyOCR` (PyTorch).
  - Caso não haja GPU CUDA disponível, o sistema detecta automaticamente e opera em **modo CPU** (`USE_GPU = False`) de forma transparente.

---

## 2. Configuração do Ambiente Virtual (`.venv`)

Na raiz do repositório, crie e ative o ambiente virtual:

### Linux e macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

---

## 3. Instalação das Dependências

Com o ambiente virtual ativado, atualize o `pip` (opcional) e instale os pacotes do `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** Na primeira execução do `EasyOCR`, os pesos do modelo de detecção e reconhecimento serão baixados automaticamente para `~/.EasyOCR/model/`.

---

## 4. Exemplos de Execução

### 4.1. Modo Webcam Padrão
Inicia a leitura em tempo real utilizando a câmera padrão (`--camera 0`):

```bash
python -m src.main --camera 0
```

- **Controles na janela:**
  - Pressione **`q`** (ou `ESC`) para encerrar a aplicação de forma limpa.
  - Pressione **`r`** para forçar uma nova leitura de OCR do dado parado.

### 4.2. Modo Debug (Janelas Intermediárias)
Exibe janelas extras com a máscara de diferença de movimento (`Motion Diff`), a máscara binária de segmentação (`Foreground Mask`), o recorte central da face superior (`Top Face ROI`) e a imagem pré-processada com CLAHE:

```bash
python -m src.main --camera 0 --debug
```

### 4.3. Modo Imagem Estática (Validação Offline sem Webcam)
Permite testar todo o pipeline de segmentação e OCR rotacionado usando uma foto de dado:

```bash
python -m src.main --image caminho/para/teste.jpg --debug
```

Para executar em ambientes sem interface gráfica (ex: CI ou terminal remoto), adicione `--headless`:

```bash
python -m src.main --image caminho/para/teste.jpg --headless
```

### 4.4. Parâmetros Adicionais da CLI
```bash
python -m src.main \
  --camera 0 \
  --stability-time 0.6 \
  --rotation-step 30 \
  --cpu \
  --debug
```

| Argumento | Padrão | Descrição |
| :--- | :--- | :--- |
| `--camera` | `0` | Índice da webcam no OpenCV (`0`, `1`, etc.). |
| `--image` | `None` | Caminho para imagem estática de teste. |
| `--video` | `None` | Caminho para arquivo de vídeo gravado. |
| `--stability-time` | `0.6` | Tempo (s) que o dado deve ficar parado antes de acionar o OCR. |
| `--rotation-step` | `30` | Passo angular (graus) na varredura de 0° a 360°. |
| `--gpu` / `--cpu` | Auto | Força execução em GPU (CUDA) ou CPU. |
| `--debug` | `False` | Abre janelas intermediárias de segmentação, ROI e CLAHE. |
| `--headless` | `False` | Executa o modo `--image` apenas no terminal, sem `cv2.imshow`. |

---

## 5. Executando os Testes Automatizados

Para validar os módulos de configuração, estabilidade, segmentação, pré-processamento CLAHE e filtragem do OCR:

```bash
pytest -v
```

---

## 6. Troubleshooting (Solução de Problemas Comuns)

1. **Erro de Permissão de Câmera (macOS / Linux / Windows):**
   - **macOS:** Acesse *Ajustes do Sistema -> Privacidade e Segurança -> Câmera* e habilite o terminal ou IDE (VS Code / Cursor / iTerm) que está executando o script.
   - **Linux:** Verifique se o seu usuário pertence ao grupo `video` (`sudo usermod -aG video $USER`) e se `/dev/video0` está acessível.
   - **Índice incorreto:** Se o notebook possuir câmera virtual ou câmera de continuidade, teste `--camera 1` ou `--camera 2`.

2. **Lentidão na Leitura de OCR em CPU:**
   - A varredura completa em passos de `30°` executa 12 rotações por parada do dado. Graças ao **detector de estabilidade**, o OCR roda **apenas uma vez** quando o dado para (mantendo o vídeo a 30+ FPS enquanto o dado rola).
   - Se a pausa no momento da leitura estiver longa em CPUs antigas, aumente o passo angular para 45° ou 60°:
     ```bash
     python -m src.main --camera 0 --rotation-step 45
     ```

3. **Dado Não Detectado ou Bounding Box Instável:**
   - Utilize um fundo de cor uniforme e contrastante em relação ao dado (ex: dado claro sobre fundo escuro ou dado escuro sobre fundo claro).
   - Ajuste `min_contour_area` e `max_contour_area` em `src/config.py` caso a câmera esteja muito distante ou muito próxima do dado.


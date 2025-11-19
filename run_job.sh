#!/bin/bash

# ==============================================================
# Script de Ejecución Generalizado - NuestraMemorIA
# ==============================================================
# Este script es portable: funciona en Linux (CUDA), Mac (MPS) y CPU.
# Requisitos: 'uv' debe estar instalado previamente por el usuario.
# ==============================================================

SESSION_NAME="transcripcion_ia"
SCRIPT_PATH="audio_transcription/scripts/transcribe_whisperx.py"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}--- NuestraMemorIA Launcher ---${NC}"

# 1. Verificación de prerrequisitos (Sin instalar nada)
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error crítico: 'uv' no está instalado o no está en el PATH.${NC}"
    echo "Por favor, instálalo (sin sudo) ejecutando: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error crítico: 'tmux' no está instalado.${NC}"
    echo "Este script requiere tmux para dejar el proceso corriendo en segundo plano."
    exit 1
fi

# 2. Configuración del Entorno (Python 3.11 estricto)
echo -e "${YELLOW}[Entorno] Asegurando Python 3.11...${NC}"
uv python pin 3.11
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: No se pudo anclar la versión de Python 3.11.${NC}"
    exit 1
fi

echo -e "${YELLOW}[Entorno] Sincronizando dependencias...${NC}"
uv sync --quiet
echo -e "${GREEN}[Entorno] Listo.${NC}"

# 3. Detección de Hardware vía Python (Portable)
# Ejecutamos un pequeño snippet de Python para que torch nos diga qué hay realmente.
echo ""
echo "Detectando hardware disponible..."

# Este script python interno genera la lista de opciones
DETECT_SCRIPT="
import torch
import sys

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f'DETECTED:CUDA:{count}')
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        # Imprimimos: INDICE | NOMBRE
        print(f'{i}|{name}')
elif torch.backends.mps.is_available():
    print('DETECTED:MPS:1')
else:
    print('DETECTED:CPU:1')
"

# Capturamos la salida
HW_INFO=$(uv run python -c "$DETECT_SCRIPT")
MODE=$(echo "$HW_INFO" | grep "DETECTED" | cut -d':' -f2)

# 4. Lógica de Selección según Hardware
GPU_ID=0
BATCH_DEFAULT=8

if [ "$MODE" == "CUDA" ]; then
    echo -e "${GREEN}-> Sistema NVIDIA CUDA detectado.${NC}"
    echo "Selecciona la GPU para este trabajo:"
    echo "-------------------------------------"
    
    # Mostrar opciones
    echo "$HW_INFO" | grep -v "DETECTED" | while IFS='|' read -r id name; do
        echo "  [$id] $name"
    done
    echo "-------------------------------------"
    read -p "Ingresa el ID de la GPU a usar [0]: " GPU_ID
    GPU_ID=${GPU_ID:-0} # Default a 0 si está vacío
    
    # Preguntar Batch Size (útil si eliges la 3090 vs la 2080)
    read -p "Tamaño del Batch (Recomendado: 8 para 11GB VRAM, 16 para 24GB) [8]: " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-8}

elif [ "$MODE" == "MPS" ]; then
    echo -e "${GREEN}-> Sistema Apple Silicon (Mac M-Series) detectado.${NC}"
    echo "Usando aceleración Metal Performance Shaders (MPS)."
    GPU_ID=0 # Irrelevante en MPS, pero el script lo pide
    
    read -p "Tamaño del Batch (Recomendado: 4 para Mac Air) [4]: " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-4}

else
    echo -e "${YELLOW}-> No se detectó acelerador. Usando CPU.${NC}"
    echo "Advertencia: Esto será lento."
    GPU_ID=0
    BATCH_SIZE=1
fi

# 5. Solicitar Rutas
echo ""
read -p "Ruta de entrada (archivo o carpeta): " INPUT_PATH
if [ -z "$INPUT_PATH" ]; then
    echo -e "${RED}Error: Debes especificar una ruta de entrada.${NC}"
    exit 1
fi

read -p "Ruta de salida [audio_transcription/outputs/transcripciones]: " OUTPUT_PATH
OUTPUT_PATH=${OUTPUT_PATH:-audio_transcription/outputs/transcripciones}

# 6. Lanzamiento en TMUX
CMD="uv run python $SCRIPT_PATH \"$INPUT_PATH\" --output_dir \"$OUTPUT_PATH\" --device_index $GPU_ID --batch_size $BATCH_SIZE"

echo ""
echo -e "${YELLOW}[Lanzador] Iniciando sesión tmux: $SESSION_NAME...${NC}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME
else
    tmux new-window -t $SESSION_NAME
fi

# Enviamos el comando + un comando read para que la terminal no se cierre al terminar y puedas ver el log
tmux send-keys -t $SESSION_NAME "$CMD; echo ''; echo '--- PROCESO FINALIZADO (Presiona Enter para cerrar) ---'; read" C-m

echo -e "${CYAN}=========================================================${NC}"
echo -e "${GREEN} Tarea enviada exitosamente a segundo plano.${NC}"
echo -e " Modo: $MODE | Disp: $GPU_ID | Batch: $BATCH_SIZE"
echo -e "${CYAN}=========================================================${NC}"
echo " -> Para ver el progreso:   tmux attach -t $SESSION_NAME"
echo " -> Para salir (detach):    Ctrl+B, luego D"
echo ""
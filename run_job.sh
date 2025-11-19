#!/bin/bash

# ==============================================================
# Script de Ejecución 
# ==============================================================

# --- [ 1. ZONA DE CONFIGURACIÓN ] ---
# Ajusta estos valores según tu entorno y preferencias.

# GPU Preferida (Solo aplica en Linux/Nvidia)
# ID 3 = RTX 3090 | IDs 0,1,2 = RTX 2080 Ti
TARGET_DEVICE_INDEX=3

# Rutas (Pueden ser relativas o absolutas)
INPUT_PATH="audio_transcription/inputs/audios/test_audios"
OUTPUT_PATH="audio_transcription/outputs/transcripciones"

# Parámetros del Modelo
BATCH_SIZE=16           # Bajar a 8 si usas las 2080 Ti
LANGUAGE="es"
ASR_MODEL="large-v3"
COMPUTE_TYPE="int8"

SESSION_NAME="whisper_auto_job"

# --- [ FIN DE CONFIGURACIÓN ] ---

SCRIPT_PATH="audio_transcription/scripts/transcribe_whisperx.py"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}--- Launcher NuestraMemorIA (Auto-Validated) ---${NC}"

# 1. Verificaciones Básicas
if ! command -v uv &> /dev/null; then echo -e "${RED}Error: Falta 'uv'.${NC}"; exit 1; fi
if ! command -v tmux &> /dev/null; then echo -e "${RED}Error: Falta 'tmux'.${NC}"; exit 1; fi

# 2. Preparar Entorno
echo -e "${YELLOW}[System] Verificando entorno...${NC}"
uv python pin 3.11 > /dev/null 2>&1
uv sync --quiet

# 3. EL "CEREBRO": Validación de Hardware
# Usamos Python para ver qué hay realmente en la máquina y validar tu TARGET_DEVICE_INDEX
echo -e "${YELLOW}[System] Validando hardware disponible...${NC}"

VALIDATION_SCRIPT="
import torch
import sys

try:
    target = int(sys.argv[1])
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if target >= count:
            print(f'ERROR:GPU_NOT_FOUND:Solo hay {count} GPUs (0-{count-1}). Solicitaste {target}.')
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(target)
        print(f'OK:CUDA:{target}:{gpu_name}')
    elif torch.backends.mps.is_available():
        print('WARN:MPS:0:Apple Silicon Detectado (Ignorando índice Nvidia)')
    else:
        print('WARN:CPU:0:Sin aceleración detectada')
except Exception as e:
    print(f'ERROR:UNKNOWN:{e}')
    sys.exit(1)
"

# Ejecutamos la validación pasándole tu configuración
CHECK_RESULT=$(uv run python -c "$VALIDATION_SCRIPT" "$TARGET_DEVICE_INDEX")
STATUS=$(echo "$CHECK_RESULT" | cut -d':' -f1)
TYPE=$(echo "$CHECK_RESULT" | cut -d':' -f2)
FINAL_INDEX=$(echo "$CHECK_RESULT" | cut -d':' -f3)
MSG=$(echo "$CHECK_RESULT" | cut -d':' -f4)

if [ "$STATUS" == "ERROR" ]; then
    echo -e "${RED}Error de Configuración:${NC} $MSG"
    echo "Por favor edita el archivo run_job.sh y corrige el TARGET_DEVICE_INDEX."
    exit 1
elif [ "$STATUS" == "WARN" ]; then
    echo -e "${YELLOW}Aviso de Hardware:${NC} $MSG"
    echo "Se usará el dispositivo disponible automáticamente."
else
    echo -e "${GREEN}Hardware Validado:${NC} Usando GPU $FINAL_INDEX ($MSG)"
fi

# 4. Construir Comando Final
CMD="uv run python $SCRIPT_PATH \"$INPUT_PATH\" \
    --output_dir \"$OUTPUT_PATH\" \
    --device_index $FINAL_INDEX \
    --batch_size $BATCH_SIZE \
    --language $LANGUAGE \
    --asr_model $ASR_MODEL \
    --compute_type $COMPUTE_TYPE"

# 5. Ejecución en TMUX
echo -e "${YELLOW}[Job] Lanzando tarea en segundo plano ($SESSION_NAME)...${NC}"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME
else
    tmux new-window -t $SESSION_NAME
fi

# Enviamos el comando
tmux send-keys -t $SESSION_NAME "$CMD; echo ''; echo '--- FINALIZADO (Presiona Enter para cerrar) ---'; read" C-m

echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}¡Proceso corriendo exitosamente!${NC}"
echo -e " -> GPU Usada: $FINAL_INDEX ($TYPE)"
echo -e " -> Input:     $INPUT_PATH"
echo -e "------------------------------------------------"
echo -e "Monitor:  tmux attach -t $SESSION_NAME"
echo -e "Salir:    Ctrl+B, luego D"
echo -e "${CYAN}================================================${NC}"
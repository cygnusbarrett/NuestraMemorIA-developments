import os
import logging
import argparse
import sys
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

try:
    from audio_transcription.scripts.pipeline import WhisperXPipeline, PipelineConfig
except ImportError:
    sys.exit("Error importando pipeline. Verifica PYTHONPATH.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CLI")

def main():
    parser = argparse.ArgumentParser(description="WhisperX Pipeline")
    
    parser.add_argument("input_path", type=str, help="Archivo o carpeta de audio")
    parser.add_argument("-o", "--output_dir", type=str, default="audio_transcription/outputs/transcripciones")
    parser.add_argument("-l", "--language", type=str, default="es")
    parser.add_argument("--asr_model", type=str, default="large-v3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--compute_type", type=str, default="int8")
    
    # Argumento nuevo para el servidor
    parser.add_argument("--device_index", type=int, default=0, 
                        help="ID de la GPU a utilizar (0, 1, 2, 3...). Default: 0")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Detectar si el usuario pidió una GPU que no existe
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.device_index >= device_count:
            logger.error(f"Error: Solicitaste GPU {args.device_index} pero solo hay {device_count} GPUs (0-{device_count-1}).")
            return

    config = PipelineConfig(
        language_code=args.language,
        asr_model_name=args.asr_model,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        device_index=args.device_index
    )

    pipeline = WhisperXPipeline(config)
    
    try:
        pipeline.transcribe_batch(args.input_path, args.output_dir)
    finally:
        pipeline._unload_models()

if __name__ == "__main__":
    main()
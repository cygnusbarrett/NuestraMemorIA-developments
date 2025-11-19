import whisperx
import torch
import gc
import os
import json
import logging
import psutil
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import nltk

load_dotenv(override=True)

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """
    Configuración para el WhisperXPipeline.
    """
    language_code: str = "es"
    asr_model_name: str = "large-v3"
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HUGGING_FACE_TOKEN"))
    
    batch_size: int = 8
    compute_type: str = "int8"
    
    # Control de Hardware
    device: Optional[str] = None       # "cuda", "cpu", "mps"
    device_index: int = 0              # ID de la GPU (0, 1, 2, 3...)
    
    def __post_init__(self):
        """
        Autodetecta el hardware óptimo si no se especifica.
        Prioridad: CUDA (Nvidia) > MPS (Apple Silicon) > CPU.
        """
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"NVIDIA GPU detectada. Usando CUDA en dispositivo ID: {self.device_index}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Apple Silicon (MPS) detectado.")
            else:
                self.device = "cpu"
                logger.warning("No se detectó acelerador. Usando CPU (Lento).")
        
        if self.hf_token is None:
            logger.warning("Falta HUGGING_FACE_TOKEN. La diarización se omitirá.")

class WhisperXPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.asr_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        
        self._setup_dependencies()
        self._load_models()

    def _setup_dependencies(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        custom_nltk_path = os.path.join(os.path.dirname(script_dir), "utilities", "nltk_data")
        if os.path.exists(custom_nltk_path):
            nltk.data.path.append(custom_nltk_path)

    def _load_models(self):
        try:
            # --- 1. ASR ---
            # Lógica específica para MPS (Apple) vs CUDA (Linux Server)
            device_asr = self.config.device
            
            if self.config.device == "mps":
                # WhisperX/Faster-Whisper a veces falla en MPS puro para ASR, forzamos CPU si es Mac
                # pero mantenemos MPS para alineación/diarización.
                device_asr = "cpu"
                logger.info("Configuración MPS: ASR en CPU, Alineación/Diarización en GPU.")

            logger.info(f"[1/3] Cargando ASR '{self.config.asr_model_name}' en {device_asr} (ID: {self.config.device_index})...")
            
            self.asr_model = whisperx.load_model(
                self.config.asr_model_name,
                device=device_asr,
                device_index=self.config.device_index, # Importante para multi-gpu
                compute_type=self.config.compute_type,
                language=self.config.language_code
            )
            
            # --- 2. Alineación ---
            logger.info(f"[2/3] Cargando Alineación en {self.config.device}...")
            # Nota: load_align_model no siempre acepta device_index explícito en versiones viejas, 
            # pero suele inferirlo del contexto de torch si seteamos el device correcto.
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.config.language_code,
                device=self.config.device
            )
            
            # --- 3. Diarización ---
            if self.config.hf_token:
                logger.info(f"[3/3] Cargando Diarización en {self.config.device}...")
                self.diarize_model = DiarizationPipeline(
                    use_auth_token=self.config.hf_token,
                    device=self.config.device
                )
            
            logger.info("Modelos cargados.")

        except Exception as e:
            logger.critical(f"Error cargando modelos: {e}")
            raise

    def _log_resource_usage(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu_info = ""
        if self.config.device == "cuda":
            # Monitorizar memoria de la GPU específica
            vram_used = torch.cuda.memory_allocated(self.config.device_index) / 1024**3
            vram_res = torch.cuda.memory_reserved(self.config.device_index) / 1024**3
            gpu_info = f" | GPU {self.config.device_index} VRAM: {vram_used:.2f}GB (Reservada: {vram_res:.2f}GB)"
        
        logger.info(f"Recursos: CPU {cpu}% | RAM {mem}%{gpu_info}")

    def _process_file(self, audio_path: str, output_dir: str):
        try:
            logger.info(f"Cargando audio: {os.path.basename(audio_path)}")
            audio = whisperx.load_audio(audio_path)
            self._log_resource_usage()

            # 1. ASR
            result = self.asr_model.transcribe(audio, batch_size=self.config.batch_size)
            
            # 2. Alineación
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False
            )
            
            # 3. Diarización
            if self.diarize_model:
                diarize_segments = self.diarize_model(audio)
                result = assign_word_speakers(diarize_segments, result)

            self._save_results(result, audio_path, output_dir)
            
            # Limpieza forzada para batchs largos
            gc.collect()
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Fallo en {audio_path}: {e}")

    def _save_results(self, result: Dict[str, Any], audio_path: str, output_dir: str):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # JSON
        with open(os.path.join(output_dir, f"{base_name}_completo.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # TXT
        with open(os.path.join(output_dir, f"{base_name}_simple.txt"), 'w', encoding='utf-8') as f:
            if "segments" in result:
                for seg in result["segments"]:
                    spk = seg.get("speaker", "UNKNOWN")
                    f.write(f"[{spk}] {seg['text'].strip()}\n")
            else:
                f.write("Sin segmentos.")

    def transcribe_batch(self, input_path: str, output_dir: str):
        files = self._discover_files(input_path)
        if not files: return
        
        logger.info(f"Iniciando lote de {len(files)} archivos en {self.config.device}:{self.config.device_index}")
        
        for i, f in enumerate(files, 1):
            logger.info(f"--- Procesando {i}/{len(files)} ---")
            self._process_file(f, output_dir)

    def _discover_files(self, input_path):
        valid_ext = ('.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg')
        files = []
        if os.path.isfile(input_path) and input_path.lower().endswith(valid_ext):
            files.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, fs in os.walk(input_path):
                for f in fs:
                    if f.lower().endswith(valid_ext):
                        files.append(os.path.join(root, f))
        return sorted(files) # Ordenado para consistencia

    def _unload_models(self):
        # Borrado seguro
        attrs = ['asr_model', 'align_model', 'align_metadata', 'diarize_model']
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        elif self.config.device == "mps":
            torch.mps.empty_cache()
# Plan de mejora de la diarización de hablantes — Instrucciones para agentes

> **Destinatario:** agentes de Claude Code que trabajan en este repositorio (`WhisperXAD`).
> **Objetivo:** mejorar de forma medible la diarización de hablantes del pipeline sin degradar la transcripción (ASR) ni la alineación.
> **Idioma de trabajo:** español en documentación, logs y mensajes de commit; inglés aceptable en nombres de código.

Lee este documento completo antes de tocar código. Las fases tienen dependencias: no saltes la Fase 0 ni la 1.

---

## 1. Contexto del proyecto

- El repositorio transcribe **audio de archivo histórico** del proyecto *Nuestra Memoria*: entrevistas en cinta magnética digitalizadas, registros de radio y prensa de los años 70–80, y testimonios.
- Características acústicas típicas: ruido de cinta (siseo), zumbido eléctrico, ancho de banda reducido, reverberación de sala, saturación, cambios de nivel, voces lejanas del micrófono y diálogos breves ("ya", "claro", "¿y después?").
- El caso dominante es **entrevistador + entrevistado** (2 hablantes), pero hay grabaciones con 3 o más voces (paneles, noticieros, reuniones).
- El material es **sensible** (testimonios de derechos humanos). **No subas audio ni transcripciones a servicios externos** (por ejemplo, la API de pyannoteAI, OpenAI o similares) sin aprobación explícita del usuario. Todo el procesamiento debe ser local o en infraestructura propia.

## 2. Estado actual (diagnóstico)

### 2.1 Código relevante

| Archivo | Rol en la diarización |
|---|---|
| `audio_transcription/scripts/transcribe.py` | CLI principal. Método `diarize_speakers()` (aprox. líneas 119–132): crea `whisperx.diarize.DiarizationPipeline(use_auth_token=..., device=...)`, la llama como `diarize_model(audio)` **sin parámetros de número de hablantes** y luego hace `assign_word_speakers(diarize_segments, aligned_result)`. Si falla, sigue sin diarización. Flag `--no-diarize`. |
| `audio_transcription/scripts/batch_jxv_worker.py` | Worker por lotes. Método `diarize()` (aprox. líneas 361–370), con la misma llamada. Exige `HUGGING_FACE_TOKEN` (aprox. línea 287). Si falla, escribe la salida alineada con `SPEAKER_UNKNOWN` (aprox. líneas 504–511). Genera salidas `*_por_minuto_diarizado.txt`. |
| `audio_transcription/scripts/make_diarized_videos.py` | Genera videos de subtítulos coloreados por hablante (paleta `SPEAKER_PALETTE`). Útil como referencia visual, pero no tiene espectrograma. |
| `pyproject.toml` | `requires-python = ">=3.11,<3.12"`, `whisperx` desde git, `pyannote.audio>=3.1,<4.0`. Gestión con `uv` (`uv.lock`). |
| `.env` | `HUGGING_FACE_TOKEN` (y opcionalmente `DROPBOX_URL`). **Nunca lo leas en voz alta, nunca lo imprimas en logs, nunca lo commitees.** |

### 2.2 Problemas identificados

1. **No se indica el número de hablantes** (`num_speakers`, `min_speakers`, `max_speakers`), así que el clustering adivina el número.
2. **No hay preprocesamiento acústico** antes de diarizar: el audio ruidoso de cinta entra directo al modelo de segmentación y de descriptores de hablante.
3. **Modelo desactualizado:** pyannote 3.1. Ya existe `pyannote.audio` 4.0 con el pipeline `speaker-diarization-community-1`, y hay alternativas open source fuertes, como DiariZen.
4. **No existe medición:** no hay anotaciones de referencia ni cálculo de DER, así que hoy es imposible saber si un cambio mejora o empeora.
5. **Síntoma observado** en la salida `*_short_full.json` de una cinta de prueba en `audio_transcription/output/tests/` (D 394, caja 6, cinta 1, lado B; ~998 s): 313 segmentos, de los cuales SPEAKER_02 = 265, SPEAKER_01 = 42 y sin hablante = 6. Muchas palabras tienen `confidence` cercana a 0. El desbalance es compatible con que el entrevistador quede absorbido por el entrevistado, pero **no lo des por hecho sin verificarlo con anotación humana**.

## 3. Hipótesis del usuario y evaluación técnica

**Hipótesis:** representar el audio en tiempo-frecuencia (espectrograma) y extraer el **contorno melódico (F0)** de cada hablante permitiría separarlos con alta precisión, como hace la app *Music Blueprint* con voces corales apiladas.

**Evaluación (tenla presente al diseñar soluciones):**

- La app resuelve un problema más fácil: las voces corales **suenan a la vez en líneas melódicas paralelas y separadas por intervalos musicales**, y seguirlas es un problema de estimación de múltiples F0.
- En una conversación los hablantes **se turnan**, la F0 varía mucho con la entonación y la emoción, y los rangos se superponen (dos hombres adultos ≈ 85–180 Hz). La F0 o la prosodia por sí sola es un **discriminador débil** de identidad; en la literatura funciona como complemento de las características espectrales.
- Lo que identifica a un hablante es el **timbre** (envolvente espectral, formantes). pyannote **ya opera en tiempo-frecuencia**: sus descriptores de hablante (WeSpeaker ResNet) reciben filtros mel.
- **Conclusión:** no reemplaces los descriptores neuronales por características de F0. Usa la F0 (a) como **característica auxiliar** combinada con los descriptores, sobre todo para separar voces masculinas de femeninas y desempatar, y (b) como base de una **herramienta visual de diagnóstico y corrección** (espectrograma + F0 + hablantes coloreados), que sí tiene mucho valor.

## 4. Principios de trabajo

1. **Medir antes de cambiar.** Ningún cambio de modelo o parámetro se acepta sin una comparación de DER contra la línea base, en el mismo conjunto de evaluación.
2. **No romper lo que funciona.** El ASR y la alineación actuales no se tocan. La diarización nueva entra como **backend seleccionable** (por ejemplo `--diar-backend pyannote31|community1|diarizen|enrolled`), con `pyannote31` como valor por defecto hasta que otro gane en la evaluación.
3. **Entornos aislados para dependencias conflictivas.** pyannote 4.x y DiariZen pueden requerir versiones de `torch` u otras librerías incompatibles con `whisperx`. No modifiques el entorno principal (`uv.lock`) para probarlos. Crea entornos separados (por ejemplo `envs/community1/`, `envs/diarizen/` con su propio `pyproject.toml`) que lean audio y escriban RTTM. La integración al pipeline puede ser por subproceso, leyendo el RTTM resultante.
4. **Formato intermedio estándar: RTTM.** Todo backend debe producir RTTM (`SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>`). La asignación de hablantes a palabras se hace después, a partir de RTTM, con una única función compartida.
5. **Reproducibilidad.** Fija semillas cuando aplique, registra versión de modelo, parámetros y hash del audio en un `run_manifest.json` por corrida.
6. **Commits pequeños** en una rama dedicada (`feat/diarization-eval`, `feat/diarization-viz`, etc.). No hagas push ni merges a `main` sin pedirlo. Mensajes de commit en español.
7. **GPU:** el pipeline respeta `CUDA_VISIBLE_DEVICES` (commit `3f73e98`). Mantén ese comportamiento. Si no hay GPU, todo debe funcionar en CPU (más lento).

---

## 5. Fases de trabajo

### Fase 0 — Preparación del conjunto de evaluación (requiere humano)

**Objetivo:** tener anotaciones de referencia ("verdad") para medir DER.

Tareas del agente:
1. Crea `eval/` con esta estructura:
   ```
   eval/
   ├── audio/          # fragmentos de evaluación (NO commitear audio; agregar a .gitignore)
   ├── reference/      # RTTM de referencia anotados por humanos (sí commitear)
   ├── uem/            # opcional: regiones evaluables
   ├── manifest.csv    # uri, ruta_audio, duración, n_hablantes_esperado, tipo (entrevista/panel/radio), notas
   └── README.md       # cómo anotar y cómo correr la evaluación
   ```
2. Propón al usuario una selección de **3 a 6 fragmentos de 3 a 5 minutos** (en total 15–30 min) que cubran: entrevista limpia, entrevista ruidosa, voces del mismo género, más de 2 hablantes, y habla superpuesta o intervenciones cortas. Usa los audios de `audio_transcription/inputs/` y las cintas de prueba existentes.
3. Prepara la **pre-anotación**: corre la diarización actual y exporta a un formato editable (RTTM, y también un `.txt`/`.csv` de etiquetas importable en Audacity: `start\tend\tSPEAKER_xx`). El humano corrige sobre esa base, lo que es mucho más rápido que anotar desde cero.
4. Documenta en `eval/README.md` el protocolo de anotación:
   - Etiquetas de hablante consistentes por fragmento (por ejemplo `ENTREVISTADOR`, `ENTREVISTADO`, `HABLANTE_3`).
   - Marcar el habla superpuesta como dos segmentos solapados.
   - Ignorar ruidos y risas que no son habla (o marcarlos aparte).
   - Herramienta sugerida: Audacity (pistas de etiquetas) o `pyannote` prodigy recipes si están disponibles.
5. **Detente y pide al usuario** que realice o valide las anotaciones. No inventes anotaciones de referencia ni uses la salida de un modelo como verdad.

**Criterio de término:** existen al menos 3 RTTM de referencia validados por un humano y `manifest.csv` completo.

### Fase 1 — Arnés de evaluación y línea base

**Objetivo:** un comando único que corre cualquier backend sobre el conjunto de evaluación y reporta métricas.

1. Crea `diarization/` como paquete (o `audio_transcription/diarization/`) con:
   - `backends/` — un módulo por backend, con interfaz común:
     ```python
     def diarize(audio_path: Path, *, num_speakers: int | None, min_speakers: int | None,
                 max_speakers: int | None, device: str, **kwargs) -> pyannote.core.Annotation
     ```
   - `rttm.py` — leer y escribir RTTM.
   - `assign.py` — asignar hablantes a palabras de WhisperX desde un `Annotation`/RTTM (reemplazo compatible de `assign_word_speakers`; maneja solapamientos asignando por mayor intersección y deja trazabilidad de los empates).
   - `evaluate.py` — CLI:
     ```bash
     uv run python -m diarization.evaluate --backend pyannote31 --manifest eval/manifest.csv \
         --out eval/results/pyannote31/ [--oracle-num-speakers] [--preprocess none|demucs|...]
     ```
2. Métricas, con `pyannote.metrics`:
   - **DER** total y sus componentes: *missed speech*, *false alarm*, *speaker confusion*.
   - Reporta **dos configuraciones**: `collar=0.25 s, skip_overlap=False` (principal) y `collar=0, skip_overlap=False` (estricta).
   - **JER** (Jaccard Error Rate), para no premiar a los hablantes dominantes.
   - Número de hablantes estimado contra el real.
   - Tiempo de procesamiento y factor de tiempo real (RTF).
   - Salida: `eval/results/<backend>/<config>/metrics.csv` + `summary.md` + RTTM de hipótesis por archivo.
3. Corre la **línea base** exactamente como el pipeline actual: `pyannote/speaker-diarization-3.1`, sin número de hablantes y sin preprocesamiento.
4. Crea `eval/results/LEADERBOARD.md`, una tabla acumulativa con una fila por configuración: backend, preprocesamiento, número de hablantes (oráculo/auto), DER@0.25, DER@0, confusión, missed, FA, JER y RTF.

**Criterio de término:** la línea base está medida y reproducible con un solo comando.

### Fase 2 — Mejoras de bajo costo sobre pyannote 3.1

Evalúa cada variante por separado contra la línea base:

1. **Número de hablantes:**
   - `num_speakers` oráculo (desde `manifest.csv`).
   - `min_speakers=2, max_speakers=4` (caso realista sin conocer el número exacto).
   - Si mejora, agrega al pipeline los argumentos CLI `--num-speakers`, `--min-speakers` y `--max-speakers`, y el soporte para leerlos desde un CSV de metadatos por archivo (`--speakers-meta archivo.csv` con columnas `filename,num_speakers`).
2. **Hiperparámetros del pipeline**, solo si hay tiempo y sobre un subconjunto de desarrollo distinto al de evaluación, para evitar sobreajuste: umbral de clustering (`clustering.threshold`) y `segmentation.min_duration_off`. Usa `pipeline.instantiate(params)`.
3. **Preprocesamiento acústico solo para la diarización** (el ASR sigue con el audio original):
   - Remuestreo a 16 kHz mono (ya se hace en el worker; verificar).
   - Aislamiento de voz con **Demucs** (`htdemucs`, pista `vocals`), útil si hay música o ruido estructurado.
   - Reducción de ruido estacionario (`noisereduce`, espectral) y/o modelos de *speech enhancement* (por ejemplo DeepFilterNet, o los de SpeechBrain).
   - Des-reverberación (por ejemplo WPE con `nara_wpe`).
   - Normalización de loudness (EBU R128 con `ffmpeg loudnorm`).
   - Evalúa cada técnica sola y la mejor combinación. **Atención:** el realce agresivo puede borrar las pistas de timbre y empeorar la confusión de hablantes. Solo los números deciden.

**Criterio de término:** el leaderboard tiene las variantes de la Fase 2 y queda identificada la mejor configuración con pyannote 3.1.

### Fase 3 — Backends de nueva generación

En entornos aislados (ver principio 3):

1. **pyannote.audio 4.0 + `pyannote/speaker-diarization-community-1`** (pesos CC-BY-4.0; requiere aceptar las condiciones en Hugging Face con el mismo token). Revisa la API vigente en su model card antes de implementar: hay cambios de API entre 3.x y 4.x (por ejemplo, el parámetro de autenticación y la salida "exclusive").
2. **DiariZen** (BUT Speech@FIT; EEND local + WavLM + clustering de pyannote). En el benchmark independiente *Benchmarking Diarization Models* (arXiv 2509.26177) fue la mejor alternativa open source (~13,3% DER contra 11,2% del servicio comercial PyannoteAI). Revisa la licencia de los pesos antes de adoptarlo.
3. **Opcional:** NVIDIA NeMo (Sortformer o MSDD), si los anteriores no rinden en audio de cinta.
4. Prueba cada backend con y sin el mejor preprocesamiento de la Fase 2, y con y sin número de hablantes.

**Criterio de término:** el leaderboard compara al menos pyannote31 (mejor variante), community1 y diarizen. Hay una recomendación escrita en `eval/results/RECOMENDACION.md` con los números, la latencia, las dependencias y el riesgo de mantenimiento.

### Fase 4 — Herramienta de visualización y corrección (espectrograma + F0 + hablantes)

**Objetivo:** materializar la intuición del usuario como herramienta de diagnóstico y de corrección humana.

1. Script `diarization/visualize.py`:
   ```bash
   uv run python -m diarization.visualize --audio X.wav --rttm hyp.rttm [--ref ref.rttm] \
       [--words X_words.csv] --out X_viz.html [--png X_viz.png] [--start 60 --end 180]
   ```
2. Contenido de la visualización:
   - **Fondo:** espectrograma log-mel o CQT (`librosa`), eje de frecuencia logarítmico hasta ~4–8 kHz.
   - **Curva F0** (`librosa.pyin`, fmin≈60 Hz, fmax≈400 Hz), **coloreada según el hablante asignado** en cada instante. Usa la misma paleta que `make_diarized_videos.py` (`SPEAKER_PALETTE`) para mantener coherencia.
   - **Franjas de hablante** bajo el espectrograma: hipótesis y, si se entrega, referencia en una franja paralela. Marca en rojo las zonas de desacuerdo.
   - **Palabras** de WhisperX alineadas bajo las franjas, si se entrega `--words`.
3. Versión **HTML interactiva autocontenida** (un solo archivo, sin servidor), con reproductor de audio sincronizado (cursor que avanza), zoom temporal, clic para saltar y **edición de la etiqueta de hablante de un segmento** (dividir, unir, reasignar). Al guardar, exporta un RTTM corregido descargable. Usa canvas o Plotly; mantén el archivo en un solo HTML. Para clips largos, genera el espectrograma como imagen embebida en base64 en vez de datos crudos.
4. Esta herramienta sirve para: (a) acelerar la anotación de la Fase 0, (b) inspeccionar los errores de cada backend y (c) producir las muestras de enrolamiento de la Fase 5.

**Criterio de término:** con un audio de evaluación se genera el HTML, se reproduce sincronizado, se corrige un segmento y se exporta un RTTM válido que `evaluate.py` acepta.

### Fase 5 — Diarización con hablantes enrolados (supervisada)

**Objetivo:** pasar de agrupar sin información previa a **clasificar contra voces conocidas**, lo que es mucho más preciso cuando los hablantes se repiten entre cintas (por ejemplo, el mismo entrevistador en todo un fondo documental).

1. **Registro de hablantes** `speakers/registry.json`:
   ```json
   {
     "spk_entrevistador_A": {
       "nombre": "…",
       "muestras": [{"audio": "…", "start": 12.3, "end": 31.0, "fuente": "anotación humana"}],
       "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
       "centroide": "speakers/embeddings/spk_entrevistador_A.npy",
       "f0_stats": {"mediana_hz": 118.0, "p10": 95.0, "p90": 160.0}
     }
   }
   ```
   Los descriptores (`.npy`) no se commitean si contienen información sensible. Discútelo con el usuario.
2. **Enrolamiento:** con 20–60 s de habla limpia por hablante (obtenidos con la herramienta de la Fase 4), calcula los descriptores por ventanas de 1,5–3 s y guarda el centroide y la dispersión.
3. **Inferencia:**
   - Segmentación y detección de habla con el mejor backend.
   - Descriptor por segmento.
   - Similitud coseno contra los centroides del registro.
   - Asignación con umbral. Si ningún hablante supera el umbral, el segmento queda como `DESCONOCIDO_n` y esos segmentos se agrupan entre sí con clustering estándar.
   - **Característica auxiliar de F0** (aquí entra la hipótesis del usuario): combina la similitud de timbre con la compatibilidad de las estadísticas de F0 del segmento contra `f0_stats` del hablante. Por ejemplo: `score = sim_cos + λ·compat_f0`, con λ calibrado en el conjunto de desarrollo. Reporta el DER con λ=0 y con el λ óptimo para cuantificar el aporte real de la F0.
4. Suavizado temporal: evita cambios de hablante en segmentos menores a ~0,3 s salvo evidencia fuerte (filtro de mediana o HMM simple sobre las etiquetas).
5. Backend `enrolled` integrado al CLI: `--diar-backend enrolled --speaker-registry speakers/registry.json --speakers spk_entrevistador_A,spk_entrevistado_B`.

**Criterio de término:** DER del backend `enrolled` en el conjunto de evaluación (usando muestras de enrolamiento **disjuntas** de los fragmentos evaluados), comparado en el leaderboard con el mejor backend no supervisado.

### Fase 6 — Integración al pipeline

1. Refactoriza `transcribe.py` y `batch_jxv_worker.py` para que llamen a `diarization.backends` en vez de `whisperx.diarize.DiarizationPipeline` directamente, con flags `--diar-backend`, `--num-speakers`, `--min-speakers`, `--max-speakers`, `--speakers-meta`, `--diar-preprocess` y `--speaker-registry`.
2. Mantén el comportamiento de respaldo (fallback) actual: si la diarización falla, escribe las salidas alineadas con `SPEAKER_UNKNOWN` y registra el error.
3. Guarda el RTTM junto a las salidas existentes (`<base>.rttm`) y añade al JSON los metadatos del backend usado.
4. Cambia el backend por defecto solo si el leaderboard lo justifica, y documéntalo.
5. Actualiza `README.md` (sección de diarización) y `run_transcribe.sh`.

---

## 6. Entregables esperados

- `eval/` con manifiesto, RTTM de referencia (humanos), resultados y `LEADERBOARD.md`.
- Paquete `diarization/` con backends, asignación, evaluación, visualización y enrolamiento.
- `eval/results/RECOMENDACION.md` con la decisión argumentada con números.
- Herramienta HTML de visualización y corrección.
- Pipeline integrado con backend seleccionable y documentación actualizada.

## 7. Cuándo detenerse y consultar al usuario

- Para crear o validar anotaciones de referencia (Fase 0). **Nunca las generes tú.**
- Antes de usar cualquier servicio externo o API en la nube con el audio.
- Antes de cambiar el backend por defecto del pipeline o modificar `uv.lock` del entorno principal.
- Si una dependencia exige aceptar licencias o condiciones nuevas en Hugging Face.
- Si los resultados contradicen este plan (por ejemplo, la F0 aporta más de lo esperado o el preprocesamiento empeora todo): reporta los números y propone el ajuste en vez de seguir a ciegas.

## 8. Referencias

- pyannote.audio: https://github.com/pyannote/pyannote-audio
- pyannote speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- pyannote speaker-diarization-community-1 (pyannote.audio 4.0): https://huggingface.co/pyannote/speaker-diarization-community-1
- pyannote.metrics: https://pyannote.github.io/pyannote-metrics/
- DiariZen: https://github.com/BUTSpeechFIT/DiariZen
- *Benchmarking Diarization Models* (2025): https://arxiv.org/abs/2509.26177. Su hallazgo principal es que los errores vienen sobre todo de habla no detectada y, en segundo lugar, de confusión entre hablantes.
- WhisperX: https://github.com/m-bain/whisperX
- librosa (pyin, CQT, specshow): https://librosa.org
- Demucs: https://github.com/adefossez/demucs
- nara_wpe (des-reverberación): https://github.com/fgnt/nara_wpe

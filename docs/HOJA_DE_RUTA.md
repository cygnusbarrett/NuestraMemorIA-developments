# Hoja de ruta: mejora medible de la diarización

> Plan de trabajo, estado, decisiones y preguntas abiertas. La referencia
> técnica (diagnóstico, hipótesis F0, detalle de cada fase y bibliografía)
> está en [`DIARIZATION_PLAN.md`](DIARIZATION_PLAN.md); las tareas la citan en
> **Ref.** Reglas generales en [`AGENTS.md`](../AGENTS.md).

**Última actualización:** 2026-11-18

## 1. Estado

| Etapa | Objetivo | Estado |
|---|---|---|
| 1 | v0 de punta a punta sin anotación: línea base corrida y reportada con un comando | cerrada (2026-11-18) |
| 2 | Conjunto de evaluación completo y mejoras baratas sobre pyannote 3.1 | por planificar (DEC-11 obliga a replantearla) |
| 3 | Backends de nueva generación en entornos aislados | sin iniciar |
| 4 | Visualización y corrección (espectrograma + F0 + hablantes) | sin iniciar |
| 5 | Hablantes enrolados | sin iniciar |
| 6 | Integración al pipeline | sin iniciar |

**Siguiente paso:** planificar la etapa 2 con medidas sin referencia (DEC-12) cuando el responsable entregue P-09.

Casillas: `[ ]` pendiente, `[~]` implementada y en revisión, `[x]` cerrada.
Todas las tareas de agente se implementan con el prompt A y se revisan con el
prompt B antes de cerrarse. Las marcadas **[Responsable]** las hace una
persona.

## 2. Etapas y tareas

### Etapa 1: v0 de punta a punta

**Objetivo verificable:** con un solo comando se corre la diarización actual
(`pyannote/speaker-diarization-3.1`, sin número de hablantes ni
preprocesamiento) sobre `ev01` y el resultado queda en
`eval/results/LEADERBOARD.md` marcado como *v0, provisional, cobertura 1
fragmento, sin referencia*: hablantes estimados, tiempo de habla y turnos por
hablante, y RTF. El cálculo de DER queda implementado y probado con datos
sintéticos, listo para cuando haya referencias.

Es la estructura completa en su versión más barata: entrada, recorte,
backend, métricas y reporte. Por DEC-11 no hay anotación humana por ahora: las
tareas 1.4 y 1.5 quedan diferidas y ningún resultado se presenta como
precisión. Las etapas siguientes agregan fragmentos, variantes, backends y,
cuando se retome, la anotación, en los mismos lugares y con los mismos
comandos.

- [x] **1.1 Estructura de `eval/` y protocolo de anotación**
  - **Depende de:** — (DEC-10 se aplica en esta misma tarea)
  - **Ref.:** DIARIZATION_PLAN §1, §5 Fase 0 (pasos 1 y 4)
  - **Rol:** dev-datos
  - Agregar a `pyproject.toml` la sección
    `[tool.setuptools.packages.find]` de DEC-10, sin modificar `uv.lock`.
  - Crear `eval/` con `audio/`, `reference/`, `uem/`, `manifest.csv`
    (cabecera: `uri,ruta_audio,duracion_s,n_hablantes_esperado,tipo,notas`) y
    `README.md` con el protocolo de anotación y el uso de Audacity.
  - Agregar a `.gitignore`: `eval/audio/`, `eval/preannotation/` y
    `eval/fuentes.local.csv` (correspondencia código → archivo original y
    tramo, con columnas `uri,archivo_original,inicio_s,fin_s`).
  - **Terminado cuando:** la estructura existe; `git check-ignore` confirma
    que las tres rutas quedan ignoradas; `eval/README.md` describe etiquetas
    (`ENTREVISTADOR`, `ENTREVISTADO`, `HABLANTE_3`), habla superpuesta,
    ruidos y el uso de códigos `evNN` en vez de nombres.

- [x] **1.2 Lectura y escritura de RTTM y backend `pyannote31`**
  - **Depende de:** 1.1
  - **Ref.:** DIARIZATION_PLAN §4 (principios 3, 4 y 7), §5 Fase 1 (paso 1)
  - **Rol:** dev-backend
  - `diarization/rttm.py`: leer y escribir RTTM a `pyannote.core.Annotation`
    y convertir desde y hacia etiquetas de Audacity (`inicio\tfin\tetiqueta`).
  - `diarization/backends/pyannote31.py` con la interfaz común
    `diarize(audio_path, *, num_speakers, min_speakers, max_speakers, device,
    **kwargs) -> Annotation`, usando el mismo modelo y la misma llamada que
    hoy hace `transcribe.py` (sin número de hablantes por defecto).
  - Tests en `tests/` con anotaciones sintéticas: ida y vuelta RTTM ↔
    `Annotation` ↔ Audacity sin pérdida (tolerancia 1 ms).
  - **Terminado cuando:** la verificación estándar sale con código 0; el
    backend corre sobre un WAV sintético o de prueba y devuelve un
    `Annotation` (comprobado a mano, sin mostrar datos reales).

- [x] **1.3 Recorte de fragmentos de evaluación**
  - **Depende de:** 1.1
  - **Ref.:** DIARIZATION_PLAN §5 Fase 0 (paso 2)
  - **Rol:** dev-datos
  - `python -m diarization.fragments`: lee `eval/fuentes.local.csv`, recorta
    cada tramo con ffmpeg a WAV 16 kHz mono en `eval/audio/<uri>.wav` (el
    original puede ser un video; se extrae su pista de audio) y
    completa `ruta_audio` y `duracion_s` en `eval/manifest.csv`. Idempotente:
    no recorta de nuevo si el WAV ya existe con la duración esperada.
  - Crea `eval/audio/` si no existe (está ignorado y no viene en un clon).
  - Registrar `ev01` (P-04): en `eval/fuentes.local.csv`, el video de archivo
    con inicio 1620 s y fin 1860 s; en `eval/manifest.csv`, la fila de `ev01`
    con `n_hablantes_esperado` vacío y `tipo` = `documental`.
  - **Terminado cuando:** un test con un audio sintético genera el recorte con
    la duración correcta (±10 ms) y la segunda corrida no lo rehace; existe
    `eval/audio/ev01.wav` de 240 s (±10 ms); la verificación estándar sale con
    código 0.

- [ ] **1.4 Pre-anotación** · *diferida (DEC-11)*
  - **Depende de:** 1.2, 1.3
  - **Ref.:** DIARIZATION_PLAN §5 Fase 0 (paso 3)
  - **Rol:** dev-datos
  - `python -m diarization.preannotate --manifest eval/manifest.csv`: corre
    `pyannote31` sobre cada fragmento y escribe
    `eval/preannotation/<uri>.rttm` y `<uri>.txt` (etiquetas de Audacity).
  - **Terminado cuando:** con el fragmento elegido en 1.5 se generan ambos
    archivos y el `.txt` abre en Audacity como pista de etiquetas; la
    verificación estándar sale con código 0.

- [ ] **1.5 [Responsable] Anotar el fragmento v0** · *diferida (DEC-11)*
  - **Depende de:** 1.1 (para elegir), 1.4 (para anotar)
  - **Ref.:** `eval/README.md`; P-04
  - Corregir la pre-anotación en Audacity, exportar las etiquetas y
    convertirlas a `eval/reference/ev01.rttm` con `diarization.rttm`.
  - **Terminado cuando:** `eval/reference/ev01.rttm` existe y el responsable
    lo da por validado.

- [x] **1.6 Arnés de evaluación**
  - **Depende de:** 1.2
  - **Ref.:** DIARIZATION_PLAN §4 (principio 5), §5 Fase 1 (pasos 1, 2 y 4)
  - **Rol:** dev-backend
  - `python -m diarization.evaluate --backend pyannote31 --manifest
    eval/manifest.csv --out eval/results/pyannote31/<config>/` con
    `pyannote.metrics`: DER con `collar=0.25` (principal) y `collar=0`
    (estricta), ambos con `skip_overlap=False`; missed speech, false alarm y
    confusión; JER; hablantes estimados contra esperados; tiempo y RTF.
  - Salidas: `metrics.csv`, `summary.md`, RTTM de hipótesis por URI y
    `run_manifest.json` (versión de modelo, parámetros, semilla, hash del
    audio). Agrega o reemplaza su fila en `eval/results/LEADERBOARD.md` con
    versión y cobertura.
  - Sin referencia para un URI (DEC-11): no calcula DER ni JER, deja esas
    columnas como `sin referencia` y reporta solo lo que no necesita verdad:
    hablantes estimados, tiempo de habla y turnos por hablante, duración
    mínima, mediana y máxima de los turnos, y RTF.
  - Carga `.env` con `python-dotenv` al arrancar, como `transcribe.py`. El
    backend lee `HUGGING_FACE_TOKEN` del entorno; sin token, pyannote 3.x
    devuelve un pipeline `None` y WhisperX falla con `AttributeError` en
    `.to()`.
  - Carga el pipeline una sola vez por corrida y no por fragmento: cargarlo
    tarda unos 12 s en CPU.
  - **Terminado cuando:** un test con referencia e hipótesis sintéticas de DER
    conocido reproduce el valor esperado (±0,001); otro test sin referencia
    produce las métricas descriptivas correctas; la verificación estándar
    sale con código 0.

- [x] **1.7 Línea base v0**
  - **Depende de:** 1.3, 1.6
  - **Ref.:** DIARIZATION_PLAN §5 Fase 1 (paso 3)
  - Correr la línea base sobre `ev01` (local o en `jumbita`, según P-02).
  - **Terminado cuando:** el LEADERBOARD tiene la fila `pyannote31 · sin
    preprocesamiento · auto` marcada *v0, provisional, cobertura 1 fragmento,
    sin referencia*; una segunda corrida da el mismo RTTM; costo (tiempo, GPU
    o CPU) anotado en la bitácora.

- [x] **1.8 Visualización v0 (espectrograma + F0 + hablantes)**
  - **Depende de:** 1.2, 1.3 (la imagen de `ev01` se genera después de 1.7)
  - **Ref.:** DIARIZATION_PLAN §3, §5 Fase 4 (pasos 1 y 2); P-05
  - **Rol:** dev-datos
  - `python -m diarization.visualize --audio <wav> --rttm <hyp.rttm>
    [--ref <ref.rttm>] --out <png> [--start S --end S]`: imagen PNG con
    espectrograma log-mel (eje de frecuencia logarítmico hasta 8 kHz), curva
    de F0 aproximada coloreada según el hablante de cada instante con la
    paleta `SPEAKER_PALETTE` de `make_diarized_videos.py`, y una franja por
    hablante debajo. Con `--ref`, una franja paralela y los desacuerdos en
    rojo.
  - Solo con dependencias que ya están en `uv.lock` (`torchaudio`,
    `matplotlib`, `numpy`): espectrograma con `torchaudio.transforms` y F0 con
    `torchaudio.functional.detect_pitch_frequency`, limitada a 60–400 Hz y
    sin valor en los tramos sin habla. `pyin` (`librosa`) queda para la v1
    (4.1), si se aprueba la dependencia.
  - La salida de audio real va a `eval/results/` y no contiene texto de la
    transcripción.
  - **Terminado cuando:** un test con un tono sintético de frecuencia conocida
    y un RTTM sintético genera el PNG y la F0 estimada queda a ±5 % del valor
    real; existe la imagen de `ev01` con la hipótesis de 1.7 y el responsable
    la revisa; la verificación estándar sale con código 0.

- [x] **1.10 Video desplazable con audio**
  - **Depende de:** 1.8
  - **Ref.:** DIARIZATION_PLAN §3 y §5 Fase 4 (pasos 2 y 3); R-05
  - **Rol:** dev-datos
  - Motivo: el responsable revisó la imagen de la 1.8, pero no puede juzgar la
    diarización sin oír cada tramo mientras lo ve, como en la app Music
    Blueprint.
  - Mismo comando, opción nueva: `python -m diarization.visualize ...
    --video <mp4> [--window 20] [--px-per-s 100] [--fps 30]`.
  - Se renderiza una imagen ancha (`px-per-s` píxeles por segundo) con el
    mismo contenido de la 1.8: espectrograma, F0 por hablante, franjas de
    hablante y marcas de tiempo `mm:ss` cada 5 s. Con ffmpeg se recorta una
    ventana de `window` segundos que avanza con el audio (`crop` con
    `x = t·px-per-s`), y se rellena el inicio y el final para que el cursor
    quede fijo en el centro: lo que pasa bajo el cursor es lo que suena.
  - Quedan siempre visibles, fuera de la zona que se desplaza: los nombres de
    hablante con su color, el eje de Hz y el cursor (`drawbox`). El
    `drawtext` de ffmpeg no está disponible, así que todo texto va dibujado
    en las imágenes.
  - Audio: el mismo tramo que la imagen, en AAC, sincronizado. Video H.264
    `yuv420p` de 1280×720 o más, que se reproduzca en QuickTime.
  - Solo ffmpeg (sistema o `imageio_ffmpeg`, como `fragments.py`) y
    dependencias que ya están en `uv.lock`. El MP4 contiene audio real:
    `eval/results/**/*.mp4` va a `.gitignore` y el video nunca sale del
    equipo.
  - **Terminado cuando:** un test unitario comprueba la correspondencia
    tiempo → píxel del recorte (el inicio de un tono sintético cae bajo el
    cursor en su instante, ±1 cuadro); un test con audio sintético genera un
    MP4 con pistas de audio y video cuya duración coincide con la del audio
    (±0,1 s); existe el video de `ev01` (240 s) y el responsable lo revisa;
    la verificación estándar sale con código 0; costo (tiempo y tamaño)
    anotado en la bitácora.

- [x] **1.9 Cierre de la etapa 1** (conductor, metodo.md §9). Depende de
  la 1.10.

### Etapa 2: conjunto completo y mejoras baratas sobre pyannote 3.1

**Objetivo verificable:** al menos 3 referencias humanas (15 a 30 minutos en
total) que cubren entrevista limpia, ruidosa, voces del mismo género, más de 2
hablantes y habla superpuesta; el LEADERBOARD identifica la mejor
configuración de pyannote 3.1. Ref.: DIARIZATION_PLAN §5 Fases 0 y 2.

Las tareas se detallan (Depende de, Rol, Terminado cuando completos) al
planificar la etapa, con las fichas `manager-etapa` y `adversario-plan`.
Mientras siga vigente DEC-11, el objetivo de esta etapa se replantea al
planificarla: sin referencias, solo se puede comparar el acuerdo entre
configuraciones y escuchar los desacuerdos, no medir el error.

- [ ] **2.1 [Responsable]** Elegir y anotar de 2 a 5 fragmentos más.
- [ ] **2.2** Número de hablantes: oráculo desde `manifest.csv` y
  `min_speakers=2, max_speakers=4`. Terminado cuando ambas filas están en el
  LEADERBOARD con la cobertura completa.
- [ ] **2.3** Preprocesamiento barato solo para diarizar (`--preprocess`):
  remuestreo y normalización de loudness (EBU R128). Terminado cuando cada
  técnica tiene su fila.
- [ ] **2.4** Reducción de ruido (`noisereduce`) y des-reverberación (WPE),
  cada una por separado. Terminado cuando cada técnica tiene su fila.
- [ ] **2.5** Aislamiento de voz con Demucs (`htdemucs`, GPU en `jumbita`),
  primero sobre un fragmento. Terminado cuando tiene su fila y su costo.
- [ ] **2.6** Mejor combinación de 2.2 a 2.5. Terminado cuando está en el
  LEADERBOARD y la mejor configuración queda nombrada en el estado.
- [ ] **2.7** (Opcional) Hiperparámetros (`clustering.threshold`,
  `segmentation.min_duration_off`) sobre un subconjunto de desarrollo
  distinto del de evaluación.
- [ ] **2.8** Cierre de la etapa 2.

### Etapa 3: backends de nueva generación

**Objetivo verificable:** el LEADERBOARD compara pyannote31 (mejor variante),
community1 y diarizen, con y sin el mejor preprocesamiento y con y sin número
de hablantes; `eval/results/RECOMENDACION.md` argumenta con números,
latencia, dependencias y riesgo de mantenimiento. Ref.: DIARIZATION_PLAN §4
(principio 3), §5 Fase 3.

- [ ] **3.1** Entorno aislado `envs/community1/` (pyannote.audio 4.x) y
  backend por subproceso que escribe RTTM. Requiere P-06.
- [ ] **3.2** Entorno aislado `envs/diarizen/` y backend. Requiere P-06.
- [ ] **3.3** Corridas comparativas y `RECOMENDACION.md`.
- [ ] **3.4** (Opcional) NeMo, si los anteriores no rinden en audio de cinta.
- [ ] **3.5** Cierre de la etapa 3.

### Etapa 4: visualización y corrección

**Objetivo verificable:** con un fragmento de evaluación se genera un HTML
autocontenido, se reproduce sincronizado, se corrige un segmento y se exporta
un RTTM que `diarization.evaluate` acepta. Ref.: DIARIZATION_PLAN §3, §5
Fase 4.

- [ ] **4.1** v1 estática: la v0 se hizo en la 1.8 (P-05). Aquí se pasa la F0 a
  `pyin` (requiere aprobar `librosa`) y se ajusta lo que haya salido de la
  revisión de la imagen de `ev01`.
- [ ] **4.2** v1 interactiva: HTML con reproductor sincronizado, zoom, salto
  por clic y edición de segmentos (dividir, unir, reasignar) con exportación
  de RTTM.
- [ ] **4.3** Cierre de la etapa 4.

### Etapa 5: hablantes enrolados

**Objetivo verificable:** DER del backend `enrolled` con muestras de
enrolamiento disjuntas de los fragmentos evaluados, comparado en el
LEADERBOARD con el mejor backend no supervisado. Ref.: DIARIZATION_PLAN §5
Fase 5.

- [ ] **5.1** Registro `speakers/registry.json` y cálculo de centroides
  (sin nombres reales en el repo; ver AGENTS.md §2).
- [ ] **5.2** Backend `enrolled` y su evaluación.
- [ ] **5.3** Cierre de la etapa 5.

### Etapa 6: integración al pipeline

**Objetivo verificable:** `transcribe.py` y `batch_jxv_worker.py` diarizan a
través de `diarization.backends` con backend seleccionable, conservan el
respaldo `SPEAKER_UNKNOWN`, guardan `<base>.rttm` y documentan el cambio.
Ref.: DIARIZATION_PLAN §5 Fase 6.

- [ ] **6.1** `diarization/assign.py`: asignación de hablantes a palabras
  desde RTTM, compatible con `assign_word_speakers`, con trazabilidad de
  empates.
- [ ] **6.2** Flags `--diar-backend`, `--num-speakers`, `--min-speakers`,
  `--max-speakers`, `--speakers-meta`, `--diar-preprocess`,
  `--speaker-registry` en ambos scripts, sin tocar ASR ni alineación.
- [ ] **6.3** Cambio de backend por defecto, solo si el LEADERBOARD lo
  justifica y el responsable lo aprueba (DEC nueva).
- [ ] **6.4** `README.md` y `run_transcribe.sh` al día.
- [ ] **6.5** Cierre de la etapa 6.

## 3. Decisiones vigentes

- **DEC-01 · Backend seleccionable.** La diarización nueva entra como backend
  seleccionable; `pyannote31` es el valor por defecto hasta que el
  LEADERBOARD justifique otro y el responsable lo apruebe. ASR y alineación no
  se tocan.
- **DEC-02 · RTTM como formato intermedio.** Todo backend produce RTTM o
  `Annotation`; la asignación a palabras se hace después con una única función
  compartida.
- **DEC-03 · Entornos aislados.** Las dependencias incompatibles con
  `whisperx` van en `envs/<backend>/` con su propio `pyproject.toml` y se
  integran por subproceso. `uv.lock` y `pyproject.toml` principales no se
  modifican sin aprobación.
- **DEC-04 · Rol de la F0.** La F0 es característica auxiliar y base de la
  herramienta de diagnóstico; no reemplaza los descriptores neuronales de
  hablante.
- **DEC-05 · Procesamiento solo en infraestructura propia.** Este equipo y
  `jumbita`. Ningún servicio externo recibe audio ni transcripciones sin
  aprobación explícita.
- **DEC-06 · Referencias humanas con códigos.** Las anotaciones de referencia
  las hace o valida una persona; los URI son `evNN` y la correspondencia con
  los archivos originales es local y no se versiona.
- **DEC-07 · Métricas.** DER con `collar=0.25` (principal) y `collar=0`
  (estricta), ambas con `skip_overlap=False`, más JER y RTF.
- **DEC-08 · Rama propia.** Todo el trabajo de diarización va en la rama
  `feature/diarizacion`, y el push se hace a esa rama en el remoto
  `upstream`; push y merge a `main`
  solo a pedido del responsable.
- **DEC-09 · v0 primero.** La etapa 1 recorre todo el camino con un solo
  fragmento anotado; los resultados se marcan con versión y cobertura hasta
  completar el conjunto en la etapa 2.
- **DEC-10 · Paquete `diarization/` en la raíz.** Aprobada el 2026-11-18
  (P-01). `pyproject.toml` declara los paquetes con
  `[tool.setuptools.packages.find]`
  `include = ["audio_transcription*", "diarization*"]`, para que `eval/`,
  `envs/`, `speakers/` y `tests/` no se descubran como paquetes. `uv.lock` no
  cambia.
- **DEC-11 · Sin anotación humana por ahora.** Decidida por el responsable el
  2026-11-18. Se trabaja sin referencias: las tareas 1.4 y 1.5 se difieren,
  el v0 reporta solo métricas que no necesitan referencia y el DER queda
  implementado, pero sin valores reales. DEC-06 sigue vigente: la salida de un
  modelo nunca se usa como verdad, así que ninguna cifra se presenta como
  precisión. Se revisa al planificar la etapa 2.
- **DEC-12 · Medición sin referencia.** Decidida por el responsable el
  2026-11-18. El material es demasiado diverso (épocas, países, dialectos y
  tecnologías de grabación) para que una anotación pequeña generalice. La
  etapa 2 compara configuraciones con medidas que no necesitan referencia:
  acierto en el número de hablantes conocido por archivo, fragmentación
  (turnos y hablantes mínimos), estabilidad ante perturbaciones leves,
  acuerdo entre configuraciones con escucha dirigida de los desacuerdos, y un
  conjunto de fragmentos más diverso. Ningún resultado se presenta como
  precisión (DEC-06).


## 4. Preguntas abiertas

- **P-01 · Ubicación del paquete.** Resuelta en DEC-10. Propuesta original:
  `diarization/` en la raíz, que coincide con los comandos
  `python -m diarization.…` de DIARIZATION_PLAN. Hallazgo: `pyproject.toml`
  no declara paquetes y setuptools los descubre solo. Cualquier carpeta nueva
  en la raíz (`eval/`, `diarization/`, `envs/`, `speakers/`) hace fallar la
  instalación del proyecto: "Multiple top-level packages discovered in a
  flat-layout". Solución propuesta: agregar a `pyproject.toml`
  `[tool.setuptools.packages.find]` con
  `include = ["audio_transcription*", "diarization*"]`. Probado en una copia:
  `uv.lock` no cambia. Toca `pyproject.toml`, así que necesita aprobación
  (DEC-03). Aprobada.
- **P-02 · Servidor `jumbita`** (bloquea 1.7 si se corre allí y 2.5). Ruta del
  repo en el servidor, GPU que se pueden usar y si tiene `uv` y el entorno
  instalados.
- **P-03 · Remoto de push.** Resuelta (2026-11-18): rama
  `feature/diarizacion` en `upstream` (DEC-08).
- **P-04 · Fragmento v0.** Resuelta (2026-11-18): `ev01` es el audio de un
  video de archivo, del minuto 27:00 al 31:00 (240 s). La ruta del original
  va solo en `eval/fuentes.local.csv`. El número de hablantes se fija al
  anotarlo.
- **P-05 · Orden de la visualización.** Resuelta (2026-11-18): la v0 de la
  visualización se adelanta a la etapa 1 como tarea 1.8.
- **P-06 · Licencias en Hugging Face** (bloquea 3.1 y 3.2). Aprobación para
  aceptar las condiciones de `speaker-diarization-community-1` y revisar la
  licencia de los pesos de DiariZen.
- **P-07 · Errores de ruff previos.** `audio_transcription/` tiene 21 errores
  (14 E402, 4 F401, 3 F541) anteriores a este plan, por eso la verificación
  estándar revisa solo `diarization` y `tests`. ¿Se limpian en una tarea
  aparte antes de la etapa 6?
- **P-08 · Cómo medir una mejora sin anotación.** Resuelta (2026-11-18): sin
  anotación (DEC-12).
- **P-09 · Insumos para la etapa 2** (bloquea su planificación). (a) Número
  de hablantes conocido por archivo, para algunas cintas. (b) 2 o 3
  fragmentos más, de otra naturaleza (idealmente entrevistas en cinta de
  otras épocas o lugares), con ruta y tramo.

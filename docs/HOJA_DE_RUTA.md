# Hoja de ruta: mejora medible de la diarización

> Plan de trabajo, estado, decisiones y preguntas abiertas. La referencia
> técnica (diagnóstico, hipótesis F0, detalle de cada fase y bibliografía)
> está en [`DIARIZATION_PLAN.md`](DIARIZATION_PLAN.md); las tareas la citan en
> **Ref.** Desde la etapa 4, la guía de mejora es
> [`GUIA_DIARIZACION.md`](GUIA_DIARIZACION.md), citada como `GUIA §n`.
> Reglas generales en [`AGENTS.md`](../AGENTS.md).

**Última actualización:** 2026-11-24

## 1. Estado

| Etapa | Objetivo | Estado |
|---|---|---|
| 1 | v0 de punta a punta sin anotación: línea base corrida y reportada con un comando | cerrada (2026-11-18) |
| 2 | Variantes de pyannote 3.1 comparadas sin referencia | cerrada (2026-11-20; escucha parcial, DEC-17) |
| 3 | Backends de nueva generación en entornos aislados | cerrada (2026-11-20; sin escucha, DEC-17) |
| 4 | Diagnóstico, conjunto diverso, corrección en HTML y primera DER real | en curso (replanificada 2026-11-24, DEC-18 a DEC-21) |
| 5 | Preprocesamiento adaptativo según el diagnóstico | sin iniciar |
| 6 | Integración al pipeline | sin iniciar |
| 7 | Hablantes enrolados (diferida) | sin iniciar |
| 8 | Corrección con LLM local (Qwen; diferida) | sin iniciar |

**Recomendación vigente (v1 provisional):** `pyannote31/auto__loudnorm`, en `eval/results/RECOMENDACION.md`. El backend por defecto no cambia (DEC-01).

**Siguiente paso:** 4.6 (HTML de revisión y corrección; Cursor) y, en paralelo, 4.5 (conjunto diverso; conductor).

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

### Etapa 2: variantes de pyannote 3.1 comparadas sin referencia

**Objetivo verificable:** cada variante de pyannote 3.1 (número de
hablantes, preprocesamiento y sus combinaciones) corre con un solo comando
en `jumbita` sobre `ev01` a `ev05`. Su resultado queda guardado aparte, con
sus métricas sin referencia (DEC-12), su imagen y su video. El LEADERBOARD
ordena las variantes con los criterios de DEC-13, y la mejor queda nombrada
en el estado tras una ronda de escucha del responsable sobre los tramos
donde las variantes discrepan. Ref.: DIARIZATION_PLAN §5 Fase 2; DEC-12 a
DEC-15.

Riesgos que la etapa controla (revisión con `adversario-plan`):
- Una variante degenerada (por ejemplo, un solo hablante) puede salir
  estable y sin fragmentación: por eso el número de hablantes es el primer
  criterio y el resumen marca toda variante que estime 1 hablante.
- El preprocesamiento no puede mover el tiempo: el audio procesado debe
  durar lo mismo que el original (±1 cuadro de 10 ms) o la variante falla.
- CPU y GPU pueden dar resultados distintos: todas las variantes, incluida la
  línea base, corren en el mismo dispositivo (`jumbita`, GPU 1).
- Los 5 fragmentos son del mismo documental: las conclusiones valen para ese
  material y no se generalizan al archivo sin otra comprobación.

- [x] **2.1 Conjunto de la etapa** (conductor, 2026-11-18)
  - `ev02` a `ev05` (min 10–13, 13–16, 16–21 y 21–24 del mismo documental)
    recortados y registrados; `ev01` con 5 hablantes aproximados según el
    responsable, con multitudes coreando. WAV copiados a `jumbita` (hash
    verificado).

- [x] **2.2 Arnés v1: variantes, criterios y resultados por iteración**
  - **Depende de:** 2.1
  - **Ref.:** DIARIZATION_PLAN §4 (principio 5), §5 Fase 2; R-04; DEC-12 a
    DEC-14
  - **Rol:** dev-backend
  - `diarization.evaluate` acepta `--num-speakers N`, `--oracle-num-speakers`
    (desde `n_hablantes_esperado`; donde está vacío corre en automático),
    `--min-speakers`, `--max-speakers`, `--preprocess <nombre>` (de momento
    solo `none`) y `--version`. El identificador de la variante sale de esos
    parámetros (por ejemplo `auto__none`, `oraculo__none`, `min2-max8__none`)
    y define la carpeta `eval/results/pyannote31/<variante>/` (DEC-14).
    La carpeta v0 `eval/results/pyannote31/auto/` (CPU, solo `ev01`) se
    conserva como histórico y su fila del LEADERBOARD se rotula así.
  - Métricas nuevas por fragmento y agregadas: acierto del número de
    hablantes donde se conoce; fragmentación (proporción de turnos de menos
    de 0,5 s y número de hablantes con menos del 2 % del habla); marca de
    "degenerada" si estima 1 hablante.
  - **Estabilidad:** `--estabilidad` vuelve a correr cada fragmento con dos
    perturbaciones leves (−6 dB de ganancia; 50 ms de silencio al inicio,
    descontados después) y reporta la discrepancia entre la salida original y
    cada perturbada con la métrica de DER, rotulada como `inestabilidad` y
    nunca como error (DEC-06).
  - **Comparación:** `python -m diarization.compare --a <variante> --b
    <variante>` calcula la discrepancia por fragmento y lista los 5 tramos
    más largos en los que no coinciden (inicio, fin, etiquetas de cada
    lado), para la escucha dirigida.
  - El LEADERBOARD tiene una fila por variante, ordenada según DEC-13, con
    hablantes por fragmento (no la suma; R-04) y el dispositivo.
  - **Terminado cuando:** tests sintéticos cubren el identificador de
    variante, la fragmentación, la marca de degenerada, el descuento del
    desfase de 50 ms en la estabilidad y `compare` con un desacuerdo conocido;
    la verificación estándar sale con código 0.

- [x] **2.3 Visualización v1 con `librosa` (adelanta 4.1; R-05)**
  - **Depende de:** 2.1
  - **Ref.:** DIARIZATION_PLAN §5 Fase 4; R-05, R-06; DEC-15
  - **Rol:** dev-datos
  - Agregar `librosa` al entorno principal (`pyproject.toml` y `uv.lock`;
    aprobado). Comprobar antes que el paquete es el oficial y que el lock solo
    agrega `librosa` y sus dependencias, sin cambiar `torch`, `whisperx` ni
    `pyannote`.
  - F0 con `librosa.pyin`, dibujando solo los cuadros sonoros; espectrograma
    CQT o eje desde 50 Hz, sin bloques en los graves; marcas de tiempo con
    línea y texto centrado; `--force` para regenerar el video (R-06).
  - `--ref` acepta otra hipótesis con un rótulo propio (`--ref-label`), para
    comparar dos variantes en la misma imagen o video con los desacuerdos en
    rojo.
  - **Terminado cuando:** el test del tono sintético pasa con `pyin`; un tono
    con silencios no dibuja F0 en los silencios; la verificación estándar
    sale con código 0; la diferencia del `uv.lock` queda anotada en la
    bitácora.

- [x] **2.4 Preprocesamiento con ffmpeg**
  - **Depende de:** 2.2
  - **Ref.:** DIARIZATION_PLAN §5 Fase 2 (paso 3)
  - **Rol:** dev-datos
  - `diarization/preprocess.py` con un registro de técnicas por nombre:
    `loudnorm` (EBU R128), `highpass80` y `afftdn` (reducción de ruido de
    ffmpeg). Escribe el audio procesado en `eval/cache/<técnica>/<uri>.wav`
    (ignorado por git), lo reutiliza si ya existe y comprueba que dura lo
    mismo. Solo el audio que va a diarizar se procesa: el ASR no se toca.
  - **Terminado cuando:** tests sintéticos de duración igual y de
    reutilización; `--preprocess loudnorm` funciona en `evaluate`; la
    verificación estándar sale con código 0.

- [x] **2.5 Preprocesamiento en entornos aislados**
  - **Depende de:** 2.4
  - **Ref.:** DIARIZATION_PLAN §4 (principio 3), §5 Fase 2 (paso 3); DEC-03,
    DEC-15
  - **Rol:** dev-infra
  - `envs/preproc/` (con su `pyproject.toml` y su lock) con `noisereduce`,
    `nara_wpe` y `demucs`, y su script `wav entrada → wav salida`. El
    registro de 2.4 agrega `noisereduce`, `wpe` y `demucs` (pista `vocals`
    de `htdemucs`), que se llaman por subproceso. Demucs usa la GPU indicada
    por `CUDA_VISIBLE_DEVICES`.
  - **Terminado cuando:** cada técnica procesa un WAV sintético y conserva la
    duración; el entorno principal y su `uv.lock` no cambian; la
    verificación estándar sale con código 0.

- [x] **2.6 Corridas en `jumbita`** (conductor)
  - **Depende de:** 2.2, 2.3 y 2.4 para la primera ronda; 2.5 para la
    segunda
  - En `jumbita` con `CUDA_VISIBLE_DEVICES=1`, sobre `ev01` a `ev05`, con
    `--estabilidad`. Ronda 1: `auto__none` (línea base en GPU),
    `oraculo__none`, `min2-max8__none`, `auto__loudnorm`, `auto__highpass80`
    y `auto__afftdn`. Ronda 2: `auto__noisereduce`, `auto__wpe` y
    `auto__demucs`. Ronda 3: la combinación de las dos mejores técnicas con
    la mejor opción de número de hablantes.
  - Imagen y video por variante y fragmento, más videos de comparación de
    cada variante contra la línea base. Los resultados se traen a este
    equipo; las imágenes, los videos y el audio procesado quedan fuera de
    git.
  - **Terminado cuando:** las 3 rondas están en el LEADERBOARD con su costo
    (tiempo de GPU y de pared) anotado en la bitácora.

- [x] **2.7 [Responsable] Ronda de escucha** · *ronda 1 hecha; rondas 2 y 3 omitidas (DEC-17)*
  - **Depende de:** 2.6
  - Unos 10 minutos por ronda: escuchar en los videos de comparación los
    tramos de desacuerdo que lista `compare`, y decir en cada uno qué variante
    suena más correcta.
  - **Terminado cuando:** el juicio del responsable queda en la bitácora.

- [x] **2.8 Recomendación y cierre de la etapa 2** (conductor)
  - `eval/results/RECOMENDACION.md` con la mejor variante de pyannote 3.1,
    los números según DEC-13, la escucha y sus límites (un solo documental,
    sin referencia). Cierre según metodo.md §9.

- [x] **2.10 Técnicas encadenadas** (antes de la ronda 3)
  - **Depende de:** 2.4
  - **Rol:** dev-datos
  - `--preprocess highpass80+loudnorm` aplica las técnicas en ese orden, con
    caché propia (`eval/cache/highpass80+loudnorm/`), control de duración y
    compensación de desfase del conjunto (R-10). El nombre de la variante es
    `auto__highpass80+loudnorm`.
  - **Terminado cuando:** test sintético de una cadena de dos técnicas con
    duración igual y desfase compensado; la verificación estándar sale con
    código 0.

- [ ] **2.9** (Opcional, no se hizo) Hiperparámetros (`clustering.threshold`,
  `segmentation.min_duration_off`), solo si la ronda 3 deja dudas.

### Etapa 3: backends de nueva generación

**Objetivo verificable:** el LEADERBOARD compara pyannote31 (mejor variante),
community1 y diarizen, con y sin el mejor preprocesamiento y con y sin número
de hablantes; `eval/results/RECOMENDACION.md` argumenta con números,
latencia, dependencias y riesgo de mantenimiento. Ref.: DIARIZATION_PLAN §4
(principio 3), §5 Fase 3.

- [x] **3.1** Entorno aislado `envs/community1/` (pyannote.audio 4.x) y
  backend por subproceso que escribe RTTM. Requiere P-06.
- [x] **3.2** Entorno aislado `envs/diarizen/` y backend. Requiere P-06.
- [x] **3.3** Corridas comparativas y `RECOMENDACION.md` (v1, sin escucha; DEC-17).
- [ ] **3.4** (Opcional, no se hizo) NeMo, si los anteriores no rinden en audio de cinta.
- [x] **3.5** Cierre de la etapa 3.

### Etapa 4: diagnóstico, conjunto diverso y primera DER real

**Objetivo verificable:** cada fragmento tiene un reporte de degradaciones;
el conjunto incluye grabaciones distintas del documental de `ev01` a `ev05`;
el responsable corrige en un HTML la hipótesis de al menos 3 fragmentos de
grabaciones distintas y el LEADERBOARD muestra la primera DER real, con su
desglose, de la línea base y de la recomendación v1. Ref.: GUIA §11.3, §12,
§13 Fase 0 y Fase 2.1; DEC-18 a DEC-20.

- [x] **4.1** v1 estática (cubierta por la 2.3: F0 con `pyin`, espectrograma logarítmico).
- [x] **4.2 F0 en solapamientos** (conductor; pedido del responsable)
  - **Ref.:** GUIA §10; DEC-04
  - `visualize.py` dibuja la F0 también donde hay más de un hablante activo,
    en un color neutro (`pyin` da un solo tono por instante, no se sabe de
    quién es), con puntos más gruesos. Regenerar con `--force` los 10
    videos principales de `pyannote31/auto__loudnorm` (CPU local, ~10 min).
  - **Terminado cuando:** un test sintético con dos hablantes solapados
    dibuja F0 en el solapamiento con el color neutro; los 10 videos están
    regenerados y el responsable los revisa; la verificación estándar sale
    con código 0.
- [x] **4.3 Guía en `docs/`** (conductor): `docs/GUIA_DIARIZACION.md`,
  citada en AGENTS.md §1 y en los Ref. Commit a pedido.
- [x] **4.4 Diagnóstico de degradaciones por archivo**
  - **Depende de:** —
  - **Ref.:** GUIA §10, §13 Fase 2.1 (tabla y boceto)
  - **Rol:** dev-datos
  - `python -m diarization.diagnose --manifest eval/manifest.csv [--rttm-dir
    <dir>] --out eval/results/diagnostico/`: por fragmento, espectro
    promedio de largo plazo (Welch) total y solo con voz, zumbido en 50 Hz y
    armónicos (dB sobre la vecindad), frecuencia de corte, pendiente
    300–3400 Hz (dB/octava), proporción de muestras saturadas, conteo de
    clics, LUFS y LRA (filtro `ebur128` de ffmpeg) y SNR con la voz del RTTM.
    Salidas: `diagnostico.csv` (versionable, sin datos sensibles) y una
    imagen del espectro por fragmento (local).
  - Modo archivo: `--inputs <lista.txt> --out <dir> [--max-minutes N]`
    (ffmpeg → 16 kHz mono; CSV con rutas; carpeta local no versionada,
    p. ej. `eval/diagnostico.local/`).
  - Sin dependencias nuevas: `numpy`, `scipy` (ya en el lock por `librosa`),
    `soundfile` o `torchaudio`, y ffmpeg.
  - **Terminado cuando:** tests sintéticos detectan un zumbido de 50 Hz
    agregado (±1 dB del valor puesto), la saturación de un seno recortado,
    el corte de un ruido filtrado a 4 kHz (±10 %) y no marcan nada en un
    ruido limpio; corre sobre `ev01` a `ev05`; la verificación estándar sale
    con código 0.
- [ ] **4.5 Conjunto diverso** (conductor + [Responsable]; DEC-19)
  - **Depende de:** 4.4; P-11
  - El conductor corre el diagnóstico sobre el archivo y preselecciona unos
    15 tramos de grabaciones distintas que cubran el rango de cada medida.
    El responsable descarta o confirma cada uno (~10 min) e indica formato y
    hablantes aproximados. Recorte con `diarization.fragments` como `ev06`
    en adelante, sin nombres en el repo.
  - **Terminado cuando:** de 5 a 10 fragmentos nuevos en el manifiesto con
    su diagnóstico; línea base y `auto__loudnorm` corridos sobre ellos.
- [ ] **4.6 HTML de revisión y corrección**
  - **Depende de:** 4.2
  - **Ref.:** DIARIZATION_PLAN §3, §5 Fase 4; GUIA §13 Fase 0; DEC-18, DEC-20
  - **Rol:** dev-frontend
  - `python -m diarization.review --audio <wav> --rttm <hyp.rttm> --out
    <html>`: HTML autocontenido que abre sin internet, con espectrograma, F0
    y franjas, reproductor sincronizado, zoom y salto por clic. Edición:
    dividir, unir, reasignar, crear hablante, marcar `CROWD`, mover bordes.
    Exporta RTTM que `diarization.rttm` lee. El HTML contiene audio real: va
    a `.gitignore`.
  - **Terminado cuando:** un test genera el HTML con audio sintético; un RTTM
    exportado de prueba se lee sin pérdida (1 ms); el responsable corrige un
    tramo y `evaluate` acepta el RTTM; la verificación estándar sale con
    código 0.
- [ ] **4.7 [Responsable] Referencias por corrección** (DEC-18)
  - **Depende de:** 4.5, 4.6
  - Corregir en el HTML la hipótesis de `pyannote31/auto__loudnorm` de al
    menos 3 fragmentos de grabaciones distintas (10–15 min de audio),
    marcando la multitud como `CROWD`. Se guardan en `eval/reference/`.
  - **Terminado cuando:** las referencias existen y el responsable las da
    por validadas.
- [ ] **4.8 Primera DER real** (conductor)
  - **Depende de:** 4.7
  - **Ref.:** GUIA §11.3, §16; DEC-07, DEC-20
  - `evaluate` puntúa con y sin `CROWD` (si falta la opción, tarea corta del
    ejecutor). DER desglosada por fragmento de la línea base, la v1 y
    `community1/auto__loudnorm`. Se anota que la referencia parte de una
    hipótesis (sesgo a favor de `auto__loudnorm`).
  - **Terminado cuando:** el LEADERBOARD muestra la DER con desglose y la
    recomendación v2 se escribe en `RECOMENDACION.md`.
- [ ] **4.9** Cierre de la etapa 4.

### Etapa 5: preprocesamiento adaptativo

**Objetivo verificable:** una cadena fija con parámetros por archivo, sacados
del diagnóstico, comparada paso a paso (ablation) contra la v2 con DER donde
hay referencia y con DEC-13 donde no. Ref.: GUIA §8.4, §13 Fase 2.2, §16.

- [ ] **5.1 Técnicas nuevas** (dev-datos): notch en 50 Hz y armónicos
  (`bandreject` de ffmpeg, Q alto), `loudnorm23` (−23 LUFS) y EQ de canal
  (espectro con voz contra una curva estándar de voz, ±12 dB, tercios de
  octava, sin subir bandas sobre el corte; `firequalizer`). Tests sintéticos
  de duración y de efecto medido con `diagnose`.
- [ ] **5.2 `--preprocess adaptativo`** (dev-datos): reglas fijas a mano del
  diagnóstico a los parámetros (notch solo si el zumbido supera 10 dB, EQ
  acotado, ruido según SNR). Guarda en el registro de la corrida qué aplicó.
- [ ] **5.3 Corridas y ablation** (conductor, `jumbita`): línea base →
  +highpass → +notch → +EQ → +loudnorm, y cada paso quitado del completo.
- [ ] **5.4 [Responsable]** Escucha de los desacuerdos (~10 min).
- [ ] **5.5** (Opcional) Hiperparámetros (`clustering.threshold`,
  `segmentation.min_duration_off`) contra DER, solo si hay referencia
  suficiente para separar ajuste y prueba (GUIA §16).
- [ ] **5.6** Cierre de la etapa 5 con la recomendación v3.

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
- [ ] **6.3** Cambio de backend o preprocesamiento por defecto, solo si el
  LEADERBOARD lo justifica y el responsable lo aprueba (DEC nueva).
- [ ] **6.4** `README.md` y `run_transcribe.sh` al día.
- [ ] **6.5** Cierre de la etapa 6.

### Etapa 7: hablantes enrolados (diferida)

**Objetivo verificable:** DER del backend `enrolled` con muestras de
enrolamiento disjuntas de los fragmentos evaluados, comparado con el mejor
backend no supervisado. Ref.: DIARIZATION_PLAN §5 Fase 5; GUIA §13 Fase 3.

- [ ] **7.1** Registro `speakers/registry.json` y centroides (sin nombres
  reales en el repo; AGENTS.md §2). Muestras de 10–20 s marcadas en el HTML.
- [ ] **7.2** Backend `enrolled` con umbral de desconocido y su evaluación.
- [ ] **7.3** Confianza por tramo y lista de tramos dudosos para revisar en
  el HTML (aprendizaje activo).
- [ ] **7.4** Cierre de la etapa 7.

### Etapa 8: corrección con LLM local (diferida; DEC-21)

**Objetivo verificable:** un Qwen local en `jumbita` propone cambios de
hablante solo en tramos de baja confianza acústica, y la DER con y sin la
corrección queda en el LEADERBOARD. Ref.: GUIA §13 Fase 4, §14, §15.

- [ ] **8.1** Modelo, licencia y servidor de inferencia local (aprobación del
  responsable; GPU libres verificadas con `nvidia-smi`).
- [ ] **8.2** Transcripción etiquetada con confianza, prompt y salida JSON a
  RTTM corregido; umbral de confianza ajustado en dev.
- [ ] **8.3** Cierre de la etapa 8.

Más adelante, sin etapa asignada (requieren horas anotadas, GUIA §13 Fases 5
y 6): fine-tuning con degradación simulada y banco de voces entre
grabaciones.

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

- **DEC-13 · Orden de criterios sin referencia.** Aprobado por el
  responsable el 2026-11-18: (1) acierto del número de hablantes conocido;
  (2) menos fragmentación; (3) más estabilidad ante perturbaciones leves;
  (4) juicio del responsable al escuchar los desacuerdos.
- **DEC-14 · Resultados por iteración.** Cada variante guarda su resultado en
  su propia carpeta (`eval/results/<backend>/<variante>/`) con métricas,
  RTTM, registro de la corrida, imagen y video, sin pisar los de otras
  variantes, para comparar todas al final. Las imágenes, los videos y el
  audio procesado se guardan en local y no se versionan.
- **DEC-15 · Dependencias aprobadas para la etapa 2.** `librosa` en el
  entorno principal (cambia `uv.lock` con aprobación del responsable);
  `noisereduce`, `nara_wpe` y `demucs` en `envs/preproc/`, aislado. Cómputo
  en `jumbita` (GPU 1, libre) y, si no está disponible, en el servidor
  `kuntun`, ambos infraestructura propia (amplía DEC-05).
- **DEC-16 · La escucha prevalece sobre el número de hablantes.** Decidida
  por el responsable el 2026-11-18, tras la ronda 1 de escucha (2.7). Ajusta
  DEC-13: si el juicio del responsable en los tramos de desacuerdo
  contradice el criterio (1), manda la escucha, porque el número de
  hablantes se conoce en un solo fragmento (`ev01`) y es aproximado. El
  LEADERBOARD conserva el orden por cifras; la recomendación (2.8) ordena
  primero por escucha y lo explica. En la ronda 1, `auto__loudnorm` ganó o
  empató en 8 de 9 tramos y la línea base no ganó ninguno.
- **DEC-17 · Cierre de las etapas 2 y 3 sin más escucha.** Decidida por el
  responsable el 2026-11-20. Las rondas 2 y 3 y la comparación de backends
  no se escuchan; la recomendación v1 se apoya en la ronda 1 y en las cifras,
  y sin escucha no cambia la variante elegida. Queda `pyannote31/auto__loudnorm`
  como provisional; los tramos propuestos siguen en la bitácora por si se
  retoma.
- **DEC-18 · Referencias por corrección.** Decidida por el responsable el
  2026-11-24; ajusta DEC-11 y DEC-12. El responsable corrige en el HTML de
  la 4.6 la hipótesis de al menos 3 fragmentos de grabaciones distintas; esas
  correcciones son referencia humana (DEC-06) y habilitan la DER real. Las
  medidas sin referencia de DEC-12 y DEC-13 siguen para los fragmentos sin
  anotar. Toda cifra de DER indica cuántos fragmentos cubre.
- **DEC-19 · Conjunto diverso preseleccionado por diagnóstico.** Decidida por
  el responsable el 2026-11-24. El conductor preselecciona tramos de
  grabaciones distintas según las medidas de `diagnose`; el responsable
  confirma, descarta e indica formato y hablantes aproximados.
- **DEC-20 · Multitud como `CROWD`.** Decidida por el responsable el
  2026-11-24. En las referencias, cánticos y bullicio llevan la etiqueta
  `CROWD`; la DER se informa con y sin esos tramos. P-10 sigue valiendo para
  el número de hablantes en las medidas sin referencia.
- **DEC-21 · LLM local Qwen, en las últimas etapas.** Decidida por el
  responsable el 2026-11-24. La corrección con LLM (GUIA §13 Fase 4) usa un
  modelo Qwen corriendo en `jumbita`, sin servicios externos (DEC-05), y
  queda para la etapa 8. Modelo y licencia exactos se aprueban en la 8.1.


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
- **P-02 · Servidor `jumbita`.** Resuelta (2026-11-18): repo en
  `~/Repos/WhisperXAD` en `feature/diarizacion`, `uv` y `.env` presentes, GPU 1
  libre (la GPU 0 la usa otro proceso y no se toca).
- **P-03 · Remoto de push.** Resuelta (2026-11-18): rama
  `feature/diarizacion` en `upstream` (DEC-08).
- **P-04 · Fragmento v0.** Resuelta (2026-11-18): `ev01` es el audio de un
  video de archivo, del minuto 27:00 al 31:00 (240 s). La ruta del original
  va solo en `eval/fuentes.local.csv`. El número de hablantes se fija al
  anotarlo.
- **P-05 · Orden de la visualización.** Resuelta (2026-11-18): la v0 de la
  visualización se adelanta a la etapa 1 como tarea 1.8.
- **P-06 · Licencias en Hugging Face.** Aprobada por el responsable
  (2026-11-20). `speaker-diarization-community-1`: pesos CC-BY-4.0, con
  aprobación automática al aceptar las condiciones; la acepta el responsable
  en Hugging Face con la cuenta del token. DiariZen: código MIT, pesos
  `BUT-FIT/diarizen-wavlm-*` CC-BY-NC-4.0 (solo uso no comercial); el
  responsable confirma que *Nuestra Memoria* los usa sin fines comerciales
  (2026-11-20).
- **P-07 · Errores de ruff previos.** `audio_transcription/` tiene 21 errores
  (14 E402, 4 F401, 3 F541) anteriores a este plan, por eso la verificación
  estándar revisa solo `diarization` y `tests`. ¿Se limpian en una tarea
  aparte antes de la etapa 6?
- **P-08 · Cómo medir una mejora sin anotación.** Resuelta (2026-11-18): sin
  anotación (DEC-12).
- **P-09 · Insumos para la etapa 2.** Resuelta (2026-11-18): ver 2.1 y
  DEC-13 a DEC-15.
- **P-10 · Número de hablantes de `ev01`.** Resuelta (2026-11-20): son 5
  contando la multitud (cántico o bullicio) como un hablante. Coincide con
  `n_hablantes_esperado` del manifiesto, así que el acierto de las variantes
  no cambia.
- **P-11 · Ubicación del archivo completo.** ¿En qué equipo está el archivo
  de audio completo sobre el que corre el diagnóstico de la 4.5 (`jumbita`,
  `kintun` o este equipo)? Necesaria para la 4.5. Respuesta del responsable
  (2026-11-24): probablemente en este equipo; si no, en `jumbita`. El
  conductor lo comprueba al empezar la 4.5 (en `jumbita`, solo lectura).

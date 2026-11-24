# Reglas del proyecto para agentes

Este archivo se lee primero, completo, en cada sesión. Si existe la carpeta
local `.metodologia/`, después se sigue su `LEEME.md`.

## 1. Qué es

Pipeline local de transcripción (Whisper large-v3), alineación (WhisperX) y
diarización (pyannote 3.1) de audio de archivo histórico del proyecto *Nuestra
Memoria*: entrevistas en cinta, registros de radio y testimonios de derechos
humanos. Objetivo actual: mejorar de forma medible la diarización sin degradar
el ASR ni la alineación.

- Plan, estado, decisiones y preguntas abiertas: `docs/HOJA_DE_RUTA.md`.
- Referencia técnica de la diarización (diagnóstico, fases, referencias):
  `docs/DIARIZATION_PLAN.md`. Se lee solo la sección que cite el **Ref.** de
  cada tarea.
- Guía para mejorar la diarización desde la etapa 4 (fases, diagnóstico de
  degradaciones, protocolo experimental): `docs/GUIA_DIARIZACION.md`. Las
  tareas la citan como `GUIA §n`; se lee solo esa sección.

## 2. Datos y privacidad

El material es sensible. Se consideran datos sensibles:

- el audio real y sus transcripciones (`*_full.json`, `*_words.csv`,
  `*.txt`);
- los nombres de personas, que aparecen en los nombres de archivo de las
  cintas;
- los valores de `.env` (`HUGGING_FACE_TOKEN`, `DROPBOX_URL`).

Reglas:

1. **Nada sale de la infraestructura propia** (este equipo y el servidor
   `jumbita`). No se suben audio ni transcripciones a servicios externos ni a
   API en la nube sin aprobación explícita del responsable.
2. Los datos sensibles **nunca** se muestran en logs, respuestas, mensajes de
   commit, issues ni PR. `.env` no se imprime, no se lee en voz alta y no se
   versiona.
3. No se versionan audio, transcripciones ni salidas de audio real.
   `audio_transcription/inputs/`, `audio_transcription/output/`,
   `audio_transcription/outputs/`, `eval/audio/` y `eval/fuentes.local.csv`
   deben estar en `.gitignore`. Los RTTM (solo tiempos y etiquetas, con URI
   en código) sí se versionan.
4. En el conjunto de evaluación, los URI son **códigos** (`ev01`, `ev02`…),
   nunca nombres de personas. La correspondencia código → archivo original
   vive en un archivo local no versionado (`eval/fuentes.local.csv`).
5. Los tests usan audio sintético o fixtures sin datos reales.
6. Las anotaciones de referencia las hace o valida una persona. Un agente
   nunca las genera ni usa la salida de un modelo como verdad.

## 3. Método

- Se trabaja por etapas y tareas de `docs/HOJA_DE_RUTA.md`. Cada tarea tiene
  **Depende de**, **Ref.** y **Terminado cuando**; se implementa solo la tarea
  pedida.
- Casillas: `[ ]` pendiente, `[~]` implementada y en revisión, `[x]` cerrada.
- Las decisiones `DEC-nn` no se reabren en una tarea: si hay desacuerdo, se
  anota y se sigue la decisión. Una decisión nueva la aprueba el responsable.
- De lo general a lo particular: primero una versión mínima completa de punta
  a punta (v0); después se mejora un aspecto a la vez, en el mismo lugar y con
  el mismo comando.
- Medir antes de cambiar: ningún cambio de modelo, parámetro o
  preprocesamiento se acepta sin comparar DER contra la línea base en el mismo
  conjunto de evaluación.
- El ASR y la alineación actuales no se tocan. La diarización nueva entra como
  backend seleccionable; `pyannote31` sigue por defecto hasta que el
  leaderboard justifique cambiarlo y el responsable lo apruebe.
- No se modifica `uv.lock` ni `pyproject.toml` del entorno principal sin
  aprobación. Las dependencias conflictivas van en entornos aislados
  (`envs/<backend>/`). Toda dependencia nueva se verifica (existe, es la
  oficial) antes de instalarla.
- Aceptar licencias o condiciones nuevas en Hugging Face requiere aprobación
  del responsable.
- El pipeline respeta `CUDA_VISIBLE_DEVICES` y debe funcionar en CPU.
- En `jumbita`, cualquier acción que no sea de lectura requiere confirmación
  explícita del responsable. Antes de usar GPU se revisa cuáles están libres
  (`nvidia-smi`).
- Lo pesado se prueba en pequeño: un fragmento, luego una muestra, luego el
  total, con tiempo estimado antes de lanzarlo.

## 4. Git

- Sin commits ni push salvo pedido explícito del responsable.
- El trabajo de diarización va en su propia rama (ver `DEC-08`); nada se
  mergea a `main` sin pedido.
- Stage archivo por archivo o con `git add -p`; nunca `git add -A`,
  `git add .` ni `git commit -a`. Revisar `git diff --cached` antes de
  commitear, en especial los archivos nuevos.
- Los hooks corren siempre; nunca `--no-verify`.
- Mensajes de commit en español, cortos, en modo imperativo
  (`feat(eval): …`, `fix(diarization): …`).
- Ramas con prefijo `feature/`, `fix/` o `docs/`.

## 5. Cero marcas de IA

- En commits, PR, issues y notas: sin trailers `Co-Authored-By:`, sin
  "Generated with…", sin correos `noreply@` de herramientas ni emojis de
  robot. El autor es el usuario de `git config user.name`.
- En código, tests y configuración: sin comentarios "generado por IA",
  "versión mejorada", "como pediste", "nuevo:", ni comentarios que repiten el
  código. Se comenta el porqué no evidente, con la densidad del código
  vecino.
- Sin archivos de notas de agentes, sin nombres `*_v2` o `*_mejorado`, sin
  configuraciones locales de agentes versionadas.

## 6. Verificación estándar

Desde la raíz del repo; cada comando debe salir con código 0:

```bash
uv run --extra dev ruff check diarization tests
uv run --with pytest pytest -q
```

`--with pytest` usa pytest sin modificar `uv.lock`. Mientras no exista
`diarization/` o `tests/`, se omite la ruta o el comando que no aplique y se
indica en el resumen de la tarea. `audio_transcription/` tiene errores de
ruff anteriores a este plan; una tarea que edite uno de esos archivos no
agrega errores nuevos en él (`ruff check <archivo>` no empeora).

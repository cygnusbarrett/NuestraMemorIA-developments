# Recomendación: backend y preprocesamiento de diarización

**Versión:** v1 provisional (2026-11-20). Cierra las etapas 2 y 3 sin la
escucha de las rondas 2 y 3 ni la de los backends nuevos (DEC-17): solo la
ronda 1 tiene juicio de escucha. Si más adelante se escucha, esta versión se
reemplaza en este mismo archivo.

**Cobertura:** `ev01` a `ev05`, cinco fragmentos de un mismo documental, en
GPU (`jumbita` para pyannote31; `kintun` para community1 y diarizen, con
`pyannote31` repetido allí y RTTM idénticos byte a byte). Sin anotación de
referencia (DEC-12): ninguna cifra de este documento es precisión. Las
conclusiones valen para ese material y no se generalizan al archivo.

## Recomendación provisional

**`pyannote31/auto__loudnorm`**: pyannote 3.1 sin fijar el número de
hablantes, con normalización de volumen EBU R128 (`--preprocess loudnorm`)
solo sobre el audio que se diariza.

- Es la única variante con juicio de escucha favorable: en la ronda 1 ganó o
  empató en 8 de los 9 tramos de desacuerdo con la línea base, y la línea
  base no ganó ninguno (DEC-16).
- Ninguna variante nueva tiene evidencia de escucha que la ponga por encima,
  y sin escucha no se cambia la recomendación por cifras (DEC-17).
- Cuesta poco: un filtro de ffmpeg, sin dependencias nuevas, en el entorno
  principal.

`pyannote31` sin preprocesamiento sigue por defecto en el pipeline hasta que
el responsable apruebe el cambio (AGENTS.md §3, tarea 6.3).

**Alternativa a escuchar primero si se retoma:** `community1/auto__loudnorm`.
Es la más estable de todas y se parece a la recomendada en tres de los cinco
fragmentos; difiere sobre todo en `ev04` (3 hablantes frente a 6).

## Números según DEC-13

Del LEADERBOARD (v1 provisional, GPU). Hablantes por fragmento `ev01` a
`ev05`; el acierto solo se conoce en `ev01` (5 hablantes aproximados,
contando la multitud como un hablante; P-10).

| variante | hablantes | acierto | turnos cortos | inestabilidad |
|---|---|---|---|---|
| `pyannote31/auto__none` (línea base) | 5, 5, 4, 6, 6 | 1/1 | 0,1882 | 0,0328 |
| `pyannote31/auto__loudnorm` | 4, 5, 3, 6, 6 | 0/1 | 0,1992 | 0,0239 |
| `pyannote31/auto__highpass80` | 5, 5, 4, 6, 6 | 1/1 | 0,2093 | 0,0359 |
| `pyannote31/auto__highpass80+loudnorm` | 4, 5, 3, 7, 6 | 0/1 | 0,2065 | 0,0370 |
| `pyannote31/auto__wpe` | 5, 5, 4, 7, 6 | 1/1 | 0,2099 | 0,0404 |
| `pyannote31/auto__afftdn` | 4, 5, 3, 6, 6 | 0/1 | 0,2348 | 0,0709 |
| `pyannote31/auto__demucs` | 5, 4, 3, 5, 6 | 1/1 | 0,2718 | 0,0851 |
| `pyannote31/auto__noisereduce` | 4, 5, 2, 5, 6 | 0/1 | 0,3147 | 0,0498 |
| `community1/auto__none` | 5, 4, 4, 3, 6 | 1/1 | 0,2838 | 0,0167 |
| `community1/auto__loudnorm` | 5, 4, 4, 3, 6 | 1/1 | 0,2603 | 0,0153 |
| `diarizen/auto__none` | 5, 5, 8, 4, 6 | 1/1 | 0,2347 | 0,0574 |
| `diarizen/auto__loudnorm` | 5, 5, 8, 4, 6 | 1/1 | 0,2348 | 0,0548 |

Las variantes con número de hablantes fijado (`oraculo__*`, `min2-max8__*`)
dan el mismo resultado que las automáticas en los tres backends: fijar o
acotar el número de hablantes no cambia nada en este material.

Discrepancia con `pyannote31/auto__loudnorm` (`ev01` a `ev05`):

| variante | ev01 | ev02 | ev03 | ev04 | ev05 |
|---|---|---|---|---|---|
| `community1/auto__loudnorm` | 0,050 | 0,162 | 0,052 | 0,334 | 0,021 |
| `diarizen/auto__loudnorm` | 0,157 | 0,295 | 0,286 | 0,498 | 0,279 |

Lectura, solo por cifras:

- `loudnorm` es el mejor preprocesamiento en los tres backends: baja o
  iguala la fragmentación y la inestabilidad.
- `wpe` casi no cambia la línea base; `noisereduce` fragmenta más que todas
  y estima 2 hablantes en `ev03`; `demucs` es la más inestable;
  `highpass80+loudnorm` no muestra ventaja sobre `loudnorm` sola.
- `community1` es el backend más estable, pero fragmenta más que pyannote31
  y estima menos hablantes en `ev04`.
- `diarizen` es el que más se aparta del resto: 8 hablantes en `ev03`,
  donde los demás ven 2 a 4, y en tramos de cántico o habla no marca voz.

## Costo, dependencias y mantenimiento

| backend | costo por variante (5 fragmentos, con `--estabilidad`) | dependencias | riesgo de mantenimiento |
|---|---|---|---|
| `pyannote31` | 30–84 s (`jumbita`, GPU compartida) | entorno principal, sin nada nuevo | bajo: es el que ya usa el pipeline |
| `community1` | 204–233 s (`kintun`, RTX 2080 Ti) | `envs/community1/`: pyannote.audio 4.1.1, torch 2.8 fijado por el controlador CUDA 12.3 de `kintun` | medio: entorno aparte, condiciones aceptadas en Hugging Face |
| `diarizen` | 225–273 s (`kintun`, RTX 3090, unos 13,5 GB de GPU) | `envs/diarizen/`: dependencias de git fijadas por commit, pesos fijados por revisión | alto: fork de pyannote 3.1, dependencias de git y licencia CC-BY-NC (uso no comercial confirmado, P-06) |

Los tiempos son de equipos distintos y no se comparan como latencia exacta;
sí muestran el orden de magnitud.

## Límites

- Un solo documental, cinco fragmentos.
- Sin escucha de las rondas 2 y 3 ni de los backends nuevos (DEC-17): la
  recomendación se apoya en la ronda 1 y en las cifras.
- La escucha de la ronda 1 cubre solo los tramos de desacuerdo.
- El número de hablantes de `ev01` es aproximado (P-10).
- Fallas comunes a todos: voces superpuestas y multitudes (cánticos,
  bullicio).

## Pendiente para la versión siguiente

- Si se retoma la escucha: los tramos propuestos están en la bitácora
  (entradas de la 2.6 y la 3.3); empezar por `community1/auto__loudnorm` en
  `ev04` y `ev02`.
- Con anotación de referencia (DEC-11, diferida), medir DER y cambiar esta
  recomendación por precisión.

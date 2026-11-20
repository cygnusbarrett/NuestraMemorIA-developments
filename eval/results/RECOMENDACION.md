# Recomendación: diarización con pyannote 3.1

**Versión:** v0 provisional (2026-11-20). Falta la escucha de las rondas 2 y
3; cuando se haga, esta versión se reemplaza en este mismo archivo.

**Cobertura:** `ev01` a `ev05`, cinco fragmentos de un mismo documental, en
`jumbita` (GPU 1). Sin anotación de referencia (DEC-12): ninguna cifra de
este documento es precisión. Las conclusiones valen para ese material y no
se generalizan al archivo.

## Recomendación provisional

**`auto__loudnorm`**: pyannote 3.1 sin fijar el número de hablantes, con
normalización de volumen EBU R128 (`--preprocess loudnorm`) solo sobre el
audio que se diariza.

- Es la única variante con juicio de escucha favorable: en la ronda 1 ganó o
  empató en 8 de los 9 tramos de desacuerdo con la línea base, y la línea
  base no ganó ninguno. Por DEC-16, la escucha prevalece sobre el acierto en
  el número de hablantes.
- Es la más estable ante perturbaciones leves (inestabilidad 0,0239, frente
  a 0,0328 de la línea base).
- Cuesta poco: un filtro de ffmpeg, sin dependencias nuevas.

`pyannote31` sin preprocesamiento sigue por defecto en el pipeline hasta que
el responsable apruebe el cambio (AGENTS.md §3).

## Números según DEC-13

Del LEADERBOARD (v1 provisional, GPU). Hablantes por fragmento `ev01` a
`ev05`; el acierto solo se conoce en `ev01` (5 hablantes aproximados,
contando la multitud; P-10).

| variante | hablantes | acierto | turnos cortos | inestabilidad |
|---|---|---|---|---|
| `auto__none` (línea base) | 5, 5, 4, 6, 6 | 1/1 | 0,1882 | 0,0328 |
| `auto__loudnorm` | 4, 5, 3, 6, 6 | 0/1 | 0,1992 | 0,0239 |
| `auto__highpass80` | 5, 5, 4, 6, 6 | 1/1 | 0,2093 | 0,0359 |
| `auto__highpass80+loudnorm` | 4, 5, 3, 7, 6 | 0/1 | 0,2065 | 0,0370 |
| `auto__wpe` | 5, 5, 4, 7, 6 | 1/1 | 0,2099 | 0,0404 |
| `auto__afftdn` | 4, 5, 3, 6, 6 | 0/1 | 0,2348 | 0,0709 |
| `auto__demucs` | 5, 4, 3, 5, 6 | 1/1 | 0,2718 | 0,0851 |
| `auto__noisereduce` | 4, 5, 2, 5, 6 | 0/1 | 0,3147 | 0,0498 |

`oraculo__none` y `min2-max8__none` dan el mismo resultado que la línea
base: fijar o acotar el número de hablantes no cambia nada en este material.

Lectura de las rondas 2 y 3, solo por cifras:

- `wpe` (reducción de reverberación) casi no cambia la salida de la línea
  base.
- `noisereduce` fragmenta más que todas y estima 2 hablantes en `ev03`.
- `demucs` (separación de voz) es la más inestable.
- `highpass80+loudnorm` resulta casi igual a `loudnorm` sola: discrepancia de
  0,005 a 0,028 en cuatro fragmentos y 0,101 en `ev04`. No muestra una
  ventaja sobre `loudnorm`.

## Límites

- Un solo documental, cinco fragmentos.
- La escucha cubre solo los tramos donde las variantes discrepan; no mide
  los tramos en los que coinciden.
- El número de hablantes de `ev01` es aproximado: 5 contando la multitud
  como un hablante (P-10).
- Fallas comunes a todas las variantes: voces superpuestas y multitudes
  (cánticos, bullicio).

## Pendiente para la versión siguiente

- Escucha de 12 tramos de las rondas 2 y 3 (2.7).

# Leaderboard

| configuración | versión | cobertura | referencia | hablantes | tiempo_habla_s | turnos | rtf | DER@0.25 | DER@0 | missed_s | false_alarm_s | confusion_s | JER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyannote31 · sin preprocesamiento · auto | v0 histórico (CPU) | 1 fragmento(s) | sin referencia | 5 | 143.387 | 56 | 0.6265 | sin referencia | sin referencia | sin referencia | sin referencia | sin referencia | sin referencia |
## Variantes v1

| variante | versión | dispositivo | cobertura | hablantes por fragmento | acierto | turnos cortos | hablantes mínimos | inestabilidad | degenerados | referencia | DER@0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyannote31/auto__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| pyannote31/min2-max8__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| pyannote31/auto__highpass80 | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.2093 | 0 | 0.0359 | 0 | sin referencia | sin referencia |
| pyannote31/auto__wpe | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 7, ev05: 6 | 1/1 | 0.2099 | 0 | 0.0404 | 0 | sin referencia | sin referencia |
| diarizen/auto__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 8, ev04: 4, ev05: 6 | 1/1 | 0.2347 | 1 | 0.0574 | 0 | sin referencia | sin referencia |
| diarizen/auto__loudnorm | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 8, ev04: 4, ev05: 6 | 1/1 | 0.2348 | 2 | 0.0548 | 0 | sin referencia | sin referencia |
| community1/auto__loudnorm | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 4, ev04: 3, ev05: 6 | 1/1 | 0.2603 | 0 | 0.0153 | 0 | sin referencia | sin referencia |
| pyannote31/auto__demucs | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 3, ev04: 5, ev05: 6 | 1/1 | 0.2718 | 0 | 0.0851 | 0 | sin referencia | sin referencia |
| community1/auto__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 4, ev04: 3, ev05: 6 | 1/1 | 0.2838 | 0 | 0.0167 | 0 | sin referencia | sin referencia |
| pyannote31/oraculo__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | forzado | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| pyannote31/auto__loudnorm | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 6, ev05: 6 | 0/1 | 0.1992 | 0 | 0.0239 | 0 | sin referencia | sin referencia |
| pyannote31/auto__highpass80+loudnorm | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 7, ev05: 6 | 0/1 | 0.2065 | 0 | 0.0370 | 0 | sin referencia | sin referencia |
| diarizen/oraculo__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 8, ev04: 4, ev05: 6 | forzado | 0.2347 | 1 | 0.0574 | 0 | sin referencia | sin referencia |
| pyannote31/auto__afftdn | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 6, ev05: 6 | 0/1 | 0.2348 | 0 | 0.0709 | 0 | sin referencia | sin referencia |
| diarizen/oraculo__loudnorm | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 8, ev04: 4, ev05: 6 | forzado | 0.2348 | 2 | 0.0548 | 0 | sin referencia | sin referencia |
| community1/oraculo__loudnorm | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 4, ev04: 3, ev05: 6 | forzado | 0.2603 | 0 | 0.0153 | 0 | sin referencia | sin referencia |
| community1/oraculo__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 4, ev04: 3, ev05: 6 | forzado | 0.2838 | 0 | 0.0167 | 0 | sin referencia | sin referencia |
| pyannote31/auto__noisereduce | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 2, ev04: 5, ev05: 6 | 0/1 | 0.3147 | 0 | 0.0498 | 0 | sin referencia | sin referencia |

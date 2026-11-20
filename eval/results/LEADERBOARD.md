# Leaderboard

| configuración | versión | cobertura | referencia | hablantes | tiempo_habla_s | turnos | rtf | DER@0.25 | DER@0 | missed_s | false_alarm_s | confusion_s | JER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyannote31 · sin preprocesamiento · auto | v0 histórico (CPU) | 1 fragmento(s) | sin referencia | 5 | 143.387 | 56 | 0.6265 | sin referencia | sin referencia | sin referencia | sin referencia | sin referencia | sin referencia |
## Variantes v1

| variante | versión | dispositivo | cobertura | hablantes por fragmento | acierto | turnos cortos | hablantes mínimos | inestabilidad | degenerados | referencia | DER@0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auto__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| min2-max8__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| auto__highpass80 | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | 1/1 | 0.2093 | 0 | 0.0359 | 0 | sin referencia | sin referencia |
| auto__wpe | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 7, ev05: 6 | 1/1 | 0.2099 | 0 | 0.0404 | 0 | sin referencia | sin referencia |
| auto__demucs | v1 provisional | cuda | 5 | ev01: 5, ev02: 4, ev03: 3, ev04: 5, ev05: 6 | 1/1 | 0.2718 | 0 | 0.0851 | 0 | sin referencia | sin referencia |
| oraculo__none | v1 provisional | cuda | 5 | ev01: 5, ev02: 5, ev03: 4, ev04: 6, ev05: 6 | forzado | 0.1882 | 0 | 0.0328 | 0 | sin referencia | sin referencia |
| auto__loudnorm | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 6, ev05: 6 | 0/1 | 0.1992 | 0 | 0.0239 | 0 | sin referencia | sin referencia |
| auto__highpass80+loudnorm | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 7, ev05: 6 | 0/1 | 0.2065 | 0 | 0.0370 | 0 | sin referencia | sin referencia |
| auto__afftdn | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 3, ev04: 6, ev05: 6 | 0/1 | 0.2348 | 0 | 0.0709 | 0 | sin referencia | sin referencia |
| auto__noisereduce | v1 provisional | cuda | 5 | ev01: 4, ev02: 5, ev03: 2, ev04: 5, ev05: 6 | 0/1 | 0.3147 | 0 | 0.0498 | 0 | sin referencia | sin referencia |

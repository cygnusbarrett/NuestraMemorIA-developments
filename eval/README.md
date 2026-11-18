# Conjunto de evaluación

Este directorio organiza los fragmentos y las referencias humanas para medir
la diarización. Cada fragmento usa un código como `ev01`, `ev02`, etc. Usa
ese código en el manifiesto y como nombre de archivo; no incluyas nombres de
personas ni transcripciones en archivos versionados o notas.

- `manifest.csv`: una fila por fragmento, con `uri`, ruta de audio relativa a
  la raíz del repositorio, duración en segundos, número esperado de hablantes,
  tipo y notas sin datos sensibles.
- `audio/`: fragmentos locales; no se versionan.
- `reference/`: referencias RTTM revisadas y validadas por una persona.
- `uem/`: regiones evaluables opcionales.
- `preannotation/`: propuestas locales para corregir; no son referencias.
- `fuentes.local.csv`: correspondencia local, no versionada, con cabecera
  `uri,archivo_original,inicio_s,fin_s` y el tramo original en segundos.

## Protocolo de anotación

Una persona anota o valida cada referencia tras escuchar el fragmento. Asigna
una etiqueta constante a cada voz dentro del fragmento: `ENTREVISTADOR`,
`ENTREVISTADO` y, para otra voz, `HABLANTE_3` (y así sucesivamente). Ajusta
el inicio y el fin de cada intervención al habla audible; no incluyas el
contenido de lo dicho en las etiquetas. Cuando dos personas hablan a la vez,
registra un segmento por voz con los intervalos superpuestos. Excluye ruidos,
silencios y risas que no sean habla; si hace falta registrarlos, hazlo aparte
de los segmentos de hablante.

## Anotar con Audacity

1. Abre localmente el fragmento y, si existe, importa su archivo de etiquetas
   de `preannotation/` desde **Archivo > Importar > Etiquetas**. La propuesta
   automática solo sirve como punto de partida.
2. Escucha el audio y crea o corrige las regiones en una pista de etiquetas.
   Usa intervalos separados y superpuestos para las voces simultáneas; añade
   otra pista de etiquetas si facilita la edición.
3. Revisa las etiquetas y sus límites escuchando el fragmento completo.
   Exporta las etiquetas desde **Archivo > Exportar > Exportar etiquetas**:
   cada fila contiene inicio, fin y etiqueta, separados por tabulaciones.
4. Convierte las etiquetas revisadas a `reference/<uri>.rttm` con la
   herramienta prevista en la tarea 1.2. Solo una referencia validada por una
   persona se usa para evaluar; conserva el audio y las etiquetas locales sin
   subirlos a servicios externos.

## Comparar variantes

Ejecuta una variante con `python -m diarization.evaluate --backend pyannote31
--manifest eval/manifest.csv --device cpu`. El directorio de resultados se
crea según los parámetros en `eval/results/pyannote31/`. Usa
`--num-speakers`, `--oracle-num-speakers` o `--min-speakers` y
`--max-speakers` para configurar los hablantes; `--estabilidad` añade dos
corridas perturbadas por fragmento. `--version` identifica la versión de la
corrida y `--preprocess none` indica que no se modifica el audio.

Para comparar dos resultados, ejecuta `python -m diarization.compare --a
<variante> --b <variante>`. La discrepancia DER y los tramos de desacuerdo
comparan salidas automáticas; no son una medida de precisión. La carpeta
`auto/` conserva el resultado histórico de la etapa 1.

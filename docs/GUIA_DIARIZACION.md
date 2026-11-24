# Guía técnica: de la teoría de señales y la atención a una diarización robusta para audios chilenos de los 80

> **Para el agente que lee esto.** Este documento resume una conversación larga entre el usuario y Claude. Partió de un paper sobre fine-tuning de agentes, pasó por fundamentos de ML/DL y procesamiento de señales, y terminó en un problema real: **diarizar audios chilenos de ~1980, degradados, ruidosos, mal ecualizados y mal grabados**. La sección 1 es el contexto, las secciones 2–10 son la base conceptual (sirven para entender el porqué de cada propuesta), y las secciones 11–17 son el plan accionable. Los fragmentos de código son **bocetos sin probar**: hay que adaptarlos al repositorio real y validarlos.

---

## Índice

1. Contexto y estado actual del proyecto
2. El paper de partida: Agentic ESOpt
3. Formas de hacer fine-tuning
4. Arquitecturas de redes neuronales
5. Convolución y sistemas LTI
6. Atención: matemática y por qué no es LTI
7. Un patrón común: el operador depende del estado (Maxwell, Schrödinger, atención)
8. Filtros adaptativos frente a filtros LTI
9. Incertidumbre, ruido y sistemas estocásticos
10. Análisis espectral
11. Diarización de hablantes: cómo funciona
12. Diagnóstico del problema actual
13. Plan de mejora por fases
14. Casos especiales
15. Generalización: qué aprende realmente el modelo
16. Protocolo experimental
17. Preguntas abiertas
18. Glosario
19. Referencias

**Anexo A** — Identidad audiovisual con el video original en blanco y negro (opcional)

---

## 1. Contexto y estado actual del proyecto

**Objetivo.** Diarizar ("quién habló cuándo") grabaciones chilenas de alrededor de 1980.

**Características del audio:**
- Degradación de cinta: siseo, wow/flutter, dropouts, saturación.
- Ruido de fondo y posible zumbido de red. **En Chile la red eléctrica es de 50 Hz.**
- Mala ecualización y ancho de banda limitado.
- Conversaciones con interrupciones, muletillas y backchannels ("ya", "mm", "cachai").
- El número de hablantes varía: entrevistas secuenciales, conversaciones, posibles multitudes.

**Estado actual:**
- **Whisper large-v3** para transcribir. Ojo: Whisper **no diariza**; el paso de hablantes lo hace pyannote u otro sistema.
- Se probaron: **pyannote 3.1**, **DiariZen** y **pyannote community-1**.
- Mejor resultado hasta ahora: **pyannote 3.1 + normalización de loudness**, con **~70%** de rendimiento. En audio moderno y limpio el sistema funciona muy bien.
- Pendiente: definir con precisión qué mide ese "70%" (¿1 − DER? ¿exactitud por segmento? ¿juicio a oído?).

**Observación clave del usuario.** Una persona sí distingue las voces en estos audios. Por lo tanto, la información necesaria está presente en la señal (o en su contexto). Eso indica que buena parte del error es **epistémico**: el modelo no conoce este tipo de audio. No es un límite impuesto por el ruido, así que en principio se puede reducir (ver sección 9).

---

## 2. El paper de partida: Agentic ESOpt

- **Título:** *Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements*
- **Autores:** Zhi Zheng, Rongsheng Chen, Yunpeng Ba, Zhenkun Wang, Yee Whye Teh, Wee Sun Lee (NUS, SUSTech, Oxford).
- **arXiv:** https://arxiv.org/abs/2608.17310 — PDF: https://arxiv.org/pdf/2608.17310
- **Página del proyecto:** https://zz1358m.github.io/Project-Agentic-ESOpt/

**Idea central.** Hacer fine-tuning de **todos los parámetros** de agentes LLM que trabajan en tareas largas usando **estrategias evolutivas (ES)** en lugar de RL con gradientes. Cada paso hace lo siguiente:
1. Muestrea perturbaciones aleatorias de los parámetros alrededor del modelo actual.
2. Evalúa cada agente perturbado con la recompensa escalar del entorno, a nivel de trayectoria completa.
3. Actualiza los parámetros con un promedio de las perturbaciones ponderado por su recompensa.
4. Reduce la escala de perturbación σ con un cosine decay, para equilibrar exploración y adaptación.

**Por qué importa.** Como no hay backpropagation, basta la memoria de GPU de inferencia: se llega a adaptar un modelo de 27B. Además funciona con retroalimentación de "caja negra" (solo una recompensa), sin necesidad de gradientes a través de toda la trayectoria.

**Resultados reportados:**
- Sudoku de horizonte largo: supera a RL en ~12,5%.
- Diseño automático de heurísticas en tiempo de prueba, con coevolución de prompt y parámetros: mejora a su baseline comparable en 28 de 36 configuraciones.
- También se valida en uso de herramientas estilo ReAct y en agentes web.

**Conexión con el proyecto.** Es más inspiración que herramienta directa. La idea de optimizar con una **métrica de caja negra** (por ejemplo, la DER) sin gradientes es la misma que se usa para **ajustar hiperparámetros del pipeline de diarización y del preprocesamiento** (sección 13, fase 1). Ahí la "recompensa" es −DER sobre un conjunto anotado.

---

## 3. Formas de hacer fine-tuning

**Aclaración importante.** El backpropagation **no es un tipo de fine-tuning**. Es el algoritmo que calcula hacia dónde mover los pesos (el gradiente). Lo que distingue a cada familia de métodos es **de dónde sale la señal de aprendizaje**.

| Familia | Señal | Ejemplos |
|---|---|---|
| Supervisado (SFT) | Pares entrada → respuesta correcta | Fine-tuning clásico, instruction tuning |
| Por preferencias | "A es mejor que B" | DPO, IPO, KTO |
| Aprendizaje por refuerzo | Una recompensa escalar | RLHF (PPO), GRPO |
| Sin gradientes | Recompensa de caja negra; se perturban y evalúan los parámetros | Estrategias evolutivas (Agentic ESOpt), CMA-ES |

**Eje independiente: cuánto del modelo se toca.**
- Full fine-tuning: todos los pesos.
- PEFT: solo una parte pequeña, como LoRA (adaptadores de bajo rango), adapters o prefix tuning.

**Para diarización:**
- Afinar el **modelo de segmentación** de pyannote es supervisado: se necesitan anotaciones RTTM.
- Afinar el **modelo de embeddings** (huellas de voz) también es supervisado, normalmente con pérdidas de clasificación de hablantes o métricas como AAM-softmax.
- Ajustar los **hiperparámetros** del pipeline es optimización de caja negra.
- Asignar hablantes por **clustering** es aprendizaje no supervisado.
- El RL **no** encaja de forma natural en este problema.

---

## 4. Arquitecturas de redes neuronales

Todas son redes neuronales, es decir, conjuntos de pesos que se ajustan. Lo que cambia es **cómo se conectan las piezas**:

- **MLP (perceptrón multicapa):** capas densas. Está dentro de casi todas las demás.
- **CNN (convolucionales):** filtros locales con pesos compartidos. Muy usadas en imagen y también en audio (espectrogramas, formas de onda). Muchos modelos de embeddings de hablante son CNN o TDNN, como ResNet en WeSpeaker o ECAPA-TDNN.
- **RNN / LSTM / GRU:** procesan la secuencia paso a paso con un estado interno. El modelo de segmentación de pyannote usa LSTM bidireccionales sobre características de SincNet.
- **Transformers:** se basan en atención y son la base de los LLM y de Whisper. WavLM, que usa DiariZen, también es un transformer.
- **Modelos de espacio de estados (S4, Mamba):** vienen de la teoría de control y procesan secuencias largas de forma eficiente.
- **GNN:** operan sobre grafos.
- **Autoencoders y modelos de difusión:** comprimen y reconstruyen; la difusión genera datos quitando ruido paso a paso.
- **GANs:** un generador y un discriminador que compiten.

**Detalles vistos en la conversación:**
- **Espacio de estados:** viene directo de control automático: x' = Ax + Bu, y = Cx + Du, discretizado. Mamba hace que A, B y C dependan de la entrada ("selectivo"). "Espacio" aquí es el espacio vectorial de las variables de estado, **no** el dominio de Fourier. Hay una conexión curiosa: un SSM lineal e invariante en el tiempo equivale a una convolución, así que se puede entrenar como convolución (incluso con FFT) y ejecutar como recurrencia.
- **GNN y grafos de conocimiento:** son el mismo objeto matemático, un grafo G = (V, E). Un grafo de conocimiento es un caso particular: dirigido, con aristas etiquetadas y a veces con varias aristas entre los mismos nodos. Las R-GCN son GNN diseñadas para él.
- **Difusión:** se agrega ruido gaussiano a los datos paso a paso hasta que solo queda ruido, y se entrena una red para predecir y quitar ese ruido. Para generar, se parte de ruido y se limpia iterativamente. La base matemática son cadenas de Markov, ecuaciones diferenciales estocásticas y *score matching*.
- **MLP dentro del transformer:** la atención mezcla información **entre** tokens, y el MLP procesa **cada** token por separado. Aporta la no linealidad, contiene cerca de 2/3 de los parámetros y se interpreta como una memoria clave-valor del "conocimiento".

---

## 5. Convolución y sistemas LTI

**Convolución discreta:** y[n] = Σₖ h[n−k]·x[k]

- Es la misma operación que en análisis de señales. En la práctica, las librerías de DL suelen implementar correlación cruzada, pero eso no cambia nada en el fondo.
- **Diferencia con señales:** en procesamiento de señales el filtro h se diseña; en una CNN el kernel se **aprende**.
- **Por qué sirve en DL:**
  - *Localidad*: detecta patrones locales.
  - *Pesos compartidos*: usa pocos parámetros.
  - *Equivariancia a traslación*: detecta el patrón esté donde esté.
- **Fourier:** el teorema de convolución sigue valiendo (convolución en el tiempo equivale a producto en frecuencia), y a veces se implementa con FFT. Aun así, las CNN suelen trabajar en el dominio espacial o temporal.

**Propiedades de los sistemas LTI que importan para el proyecto:**
- La familia LTI es **cerrada** bajo conexión en serie, en paralelo, combinación lineal y convolución: **combinar sistemas LTI da otro sistema LTI**. Esto corrige una intuición inicial de la conversación.
- Un sistema LTI puede tener realimentación y seguir siendo LTI. La realimentación por sí sola no rompe la linealidad; lo que la rompe son los **productos entre señales** o la dependencia de los parámetros respecto de la señal.
- **Cadena de grabación como sistema LTI:** micrófono → preamplificador → cinta → reproductor → digitalización se puede aproximar, para cada grabación, como un filtro fijo h (el "canal"), más ruido y más algunas no linealidades (saturación) y variaciones temporales (wow/flutter). **Esta es la base de la compensación de canal de la sección 13.**

---

## 6. Atención: matemática y por qué no es LTI

### 6.1 Ecuaciones

Cada token i genera tres vectores por proyección lineal: consulta qᵢ = W_Q xᵢ, clave keyᵢ = W_K xᵢ y valor vᵢ = W_V xᵢ.

```
a[n,j] = softmax_j( q[n] · key[j] / √d )
y[n]   = Σ_j a[n,j] · v[j]
```

En forma matricial: **Attention(Q, K, V) = softmax(QKᵀ/√d)·V**.

> Nota de notación: en la conversación se usó una vez la letra *k* tanto para el índice como para la clave, y eso causó confusión. Los pesos se calculan comparando **consulta con clave** (q·key), no consulta con valor. Los valores son lo que se promedia.

- **Consulta:** lo que busca la posición n.
- **Clave:** cómo se "anuncia" la posición j para que la encuentren.
- **Valor:** la información que entrega la posición j si la eligen.
- Separar clave y valor permite que lo que sirve para *encontrar* sea distinto de lo que se *aporta*, igual que en una base de datos.
- **Fundamento matemático:** es una búsqueda suave en una memoria clave-valor. Es pariente de la **regresión por kernel de Nadaraya-Watson**. El softmax tiene la forma de la **distribución de Boltzmann** (e^(−E/kT)/Z), y la atención equivale a la regla de actualización de las **redes de Hopfield modernas**, que son modelos basados en energía. **No hay azar**: el softmax solo reparte pesos.
- **Origen:** estadística y ML (Bahdanau et al. 2014 para traducción; Vaswani et al. 2017 para el transformer). No viene de la ingeniería eléctrica, aunque el producto punto se parece a una correlación o a un filtro adaptado.

### 6.2 Por qué no es LTI

- **No es lineal.** En la convolución, el kernel h es fijo. En atención, los pesos a[n,j] **se calculan con la propia entrada**: x aparece en los pesos (vía q y key) y en lo que se pondera (vía v), y además pasa por un softmax. Si duplicas la entrada, la salida no se duplica.
- **La invariancia en el tiempo depende de la codificación de posición:**
  - Sin codificación de posición, la atención es **equivariante a permutaciones**, más fuerte que la invariancia temporal: no conoce el orden.
  - Con **codificaciones absolutas**, la invariancia temporal se rompe.
  - Con **codificaciones relativas** (RoPE), los pesos dependen del contenido **y** del retardo n−j.
- **Analogía en ingeniería eléctrica:** la atención es como un **filtro adaptativo global**. Para cada posición se arma un filtro nuevo a partir de la misma señal, que mira toda la secuencia y no solo una ventana local. Su costo es O(N²).
- Los SSM selectivos como Mamba buscan un punto intermedio: un sistema casi LTI cuyos parámetros dependen de la entrada.

---

## 7. Un patrón común: el operador depende del estado

Las ecuaciones lineales "de fondo" siguen siendo lineales al superponer soluciones: la suma de soluciones es solución. **La no linealidad aparece cuando se acoplan con otra cosa y se forman productos.**

| Sistema | Estructura |
|---|---|
| Atención | y = A(x)·V(x): los pesos dependen de x y se aplican a algo que también depende de x |
| Maxwell + Lorentz | El campo mueve las cargas (F = q·v×B) y las cargas generan el campo; el producto v×B acopla estado con estado (plasmas). En materiales, la óptica no lineal aparece cuando la respuesta del medio no es proporcional al campo |
| Schrödinger | Lineal en ψ, pero las probabilidades son \|ψ\|² (cuadrático). En campo medio (Hartree, Gross-Pitaevskii): iħ∂ψ/∂t = (H₀ + g\|ψ\|²)ψ, donde el operador depende del estado. Es el análogo más cercano a la atención |
| Filtro adaptativo (LMS) | Los coeficientes dependen del error, que depende de la salida, que depende de los coeficientes; aparecen productos x·xᵀ·w |

**Lo que se repite no es el ruido ni la incertidumbre.** Es que **el operador se construye a partir del estado sobre el que actúa**.

---

## 8. Filtros adaptativos frente a filtros LTI

### 8.1 Definición

Un filtro, normalmente FIR, cuyos coeficientes w[n] se actualizan para minimizar un error:

```
y[n]   = w[n]ᵀ x[n]                  (x[n] = vector de las últimas muestras)
e[n]   = d[n] − y[n]                 (d = señal deseada o de referencia)
w[n+1] = w[n] + μ · e[n] · x[n]      (LMS: descenso de gradiente estocástico)
```

Otros algoritmos: NLMS (LMS normalizado) y RLS (mínimos cuadrados recursivos, que converge más rápido). Usos: cancelación de eco, audífonos con cancelación de ruido, ecualización de canal y beamforming.

### 8.2 Por qué el sistema completo no es LTI

Al reemplazar el error en la regla de actualización:

```
w[n+1] = w[n] + μ · x[n] · (d[n] − x[n]ᵀ w[n])
```

- Aparece **x·xᵀ·w**: la entrada multiplicada por sí misma y por el estado. Si duplicas x, ese término se multiplica por 4.
- La salida y = wᵀx es el **producto de dos cantidades que dependen de la señal**.
- Lo que rompe la linealidad son estos **productos**, no el lazo de realimentación por sí solo.
- **Aproximación útil:** si μ es pequeño, los coeficientes cambian lentamente, y el filtro se puede analizar como una **sucesión de filtros LTI casi congelados**. Es un análisis aproximado; el sistema real sigue siendo no lineal.

### 8.3 Dónde se estudian

Señales y sistemas → procesamiento digital de señales → probabilidad y procesos estocásticos → **filtrado adaptativo / procesamiento estadístico de señales** (electivo de fin de carrera o de posgrado en ingeniería eléctrica). Referencia: Haykin, *Adaptive Filter Theory*; Widrow & Stearns. El **filtro de Wiener** es el filtro LTI óptimo para señal más ruido estacionarios; un filtro adaptativo lo "persigue" cuando las estadísticas cambian. El **filtro de Kalman** aparece en estimación y control.

El filtrado adaptativo es **una herramienta dentro** del procesamiento estadístico de señales, que también incluye estimación, detección y análisis espectral.

### 8.4 Relevancia para el proyecto

- El preprocesamiento propuesto en la sección 13 es **adaptativo por archivo**: misma cadena, parámetros estimados de cada grabación.
- El banco de voces con centroides que se actualizan online es otra forma de adaptación.
- Con varios micrófonos, el beamforming es filtrado adaptativo espacial. Probablemente no aplica a cintas mono.

---

## 9. Incertidumbre, ruido y sistemas estocásticos

### 9.1 Dos tipos de incertidumbre

- **Epistémica (falta de conocimiento):** se reduce con datos o con mejores modelos. Ejemplos en el proyecto:
  - El filtro de canal h de cada cinta: es fijo pero desconocido. **No es ruido**; se puede estimar.
  - El modelo de embeddings nunca vio este tipo de canal (*domain shift*).
  - El modelo no usa el contexto conversacional que sí usa un humano.
- **Aleatoria (variabilidad irreducible):** no se elimina con más datos.
  - El siseo y el ruido térmico.
  - La información que la cinta ya destruyó: bandas cortadas, dropouts, saturación severa.
  - La mezcla de voces en mono dentro de una multitud.

"Aleatoriedad" e "incertidumbre" **no son sinónimos exactos**. Un dado sigue una física determinista, pero lo modelamos como aleatorio por falta de conocimiento de sus condiciones iniciales. La probabilidad sirve para modelar ambos tipos.

**Implicancia práctica.** El desempeño humano sobre los mismos minutos sirve como estimación del techo. Si el humano acierta casi siempre, la mayor parte del error del sistema es epistémica y se puede atacar.

### 9.2 LTI con incertidumbre

La aleatoriedad está en la **señal**, no en el **sistema**. Para cada realización, y = h * x sigue siendo una convolución. La estadística de la salida se propaga así:
- Densidad espectral: **S_y(f) = |H(f)|² · S_x(f)**
- Media: μ_y = H(0)·μ_x

Esa es la base del filtro de Wiener.

### 9.3 No todo sistema estocástico es LTI más ruido

Existen sistemas estocásticos que no son LTI con ruido aditivo:
- **Dinámica no lineal con ruido** (ecuaciones diferenciales estocásticas):
  - Poblaciones: dN = rN(1−N/K)dt + σN·dW.
  - Finanzas: dS = μS·dt + σS·dW (ruido multiplicativo).
  - Neuronas: integración hasta un umbral y disparo, con ruido de los canales iónicos.
- **Ruido multiplicativo** y **parámetros aleatorios**.
- Fenómenos que no existen en LTI con ruido: **resonancia estocástica** (agregar ruido mejora la detección de una señal débil), transiciones inducidas por ruido, extinciones.

---

## 10. Análisis espectral

Estima cómo se reparte la potencia de una señal entre las frecuencias, sobre todo cuando la señal es aleatoria. Con una señal ruidosa, una sola FFT da un espectro muy irregular.

- **Periodograma:** |FFT|² normalizado, con mucha varianza.
- **Método de Welch:** promedia periodogramas de tramos solapados con ventana. Es la herramienta estándar para el **espectro promedio de largo plazo (LTAS)** de una grabación.
- **Métodos paramétricos:** ajustan un modelo AR y obtienen el espectro de forma analítica (relacionado con LPC y formantes).
- **Wiener-Khinchin:** la densidad espectral es la transformada de Fourier de la autocorrelación.
- **Espectrograma (STFT):** el espectro en el tiempo, para ver eventos (clics, zumbidos, cortes).

**Usos en el proyecto:** diagnosticar degradaciones, estimar el canal (LTAS de la voz del archivo contra un LTAS de referencia), detectar zumbido, medir el ancho de banda útil y estimar el piso de ruido a partir de los silencios.

**Advertencia sobre "timbre = huella espectral".** El espectro promedio **no** es una huella de voz confiable. Mezcla hablante, fonemas, canal y emociones. La envolvente espectral (formantes, MFCC) es mejor, y los embeddings neuronales entrenados para ignorar el contenido (x-vectors, ECAPA, ResNet de WeSpeaker) son aún mejores.

---

## 11. Diarización de hablantes: cómo funciona

### 11.1 Pipeline genérico

1. **VAD:** detectar dónde hay voz.
2. **Segmentación:** detectar cambios de hablante y solapamientos.
3. **Embeddings:** una huella de voz por segmento o ventana.
4. **Clustering:** agrupar huellas por hablante, con un número conocido o estimado.
5. **Resegmentación y postproceso:** refinar los bordes y asignar los solapamientos.

**Variante online/streaming** (la intuición original del usuario): por cada ventana se calcula la huella, se compara con los centroides conocidos (similitud coseno) y se decide: si se parece lo suficiente, se asigna a ese hablante y se actualiza el centroide (adaptativo); si no, se crea un hablante nuevo. **Este paso es prácticamente atención**: la ventana es la consulta, los centroides son las claves y el canal elegido es el valor.

### 11.2 Sistemas probados

- **pyannote 3.1:** modelo de segmentación `segmentation-3.0` (SincNet + LSTM, *powerset* multiclase en ventanas de ~10 s, con detección de solapamiento), embeddings de WeSpeaker (ResNet34) y clustering aglomerativo. Hiperparámetros relevantes: `segmentation.min_duration_off`, `clustering.threshold`, `clustering.min_cluster_size`. Acepta `num_speakers`, `min_speakers` y `max_speakers`.
- **pyannote community-1:** versión más nueva, con clustering **VBx**, entre otras mejoras.
- **DiariZen:** segmentación con WavLM, un transformer auto-supervisado, más clustering tipo VBx.
- *(Verificar los detalles exactos de cada versión en sus repositorios y tarjetas de modelo).*

**Hallazgo importante.** community-1 y DiariZen ya incorporan mejoras de clustering temporal y modelos más grandes, y **aun así no superaron a la 3.1 con normalización de loudness**. Esto sugiere que el cuello de botella **no es la sofisticación del pipeline**, sino el **desajuste de dominio**: canal, nivel y degradación. Que la normalización de loudness haya ayudado tanto apunta en la misma dirección. **Advertencia:** hay que confirmar que DiariZen y community-1 también se probaron con la misma normalización; si no, la comparación no es pareja.

### 11.3 Métrica: DER

**DER = (falsa alarma + voz no detectada + confusión de hablante) / duración total de voz de referencia.**

Hay que reportar siempre el **desglose**, porque cada componente apunta a una etapa distinta:
- Voz no detectada o falsa alarma alta → problema de VAD o segmentación.
- Confusión alta → problema de embeddings o clustering.
- Error concentrado en solapamientos → problema de detección de solapamiento.

Usar `pyannote.metrics`. Definir y documentar el *collar* (por ejemplo 0 s o 0,25 s) y si se puntúan los solapamientos. Mantener siempre la misma configuración para que los resultados sean comparables.

---

## 12. Diagnóstico del problema actual

**Hipótesis ordenadas por impacto esperado:**

1. **Desajuste de canal/dominio.** Los modelos se entrenaron con audio moderno; la cinta de los 80 cambia el espectro, el nivel, el ancho de banda y el ruido. Afecta a todas las etapas. Es la causa más probable del grueso del error.
2. **Segmentos cortos y backchannels.** Los "ya", "mm" o "cachai" que alguien mete mientras otro habla son breves y solapados, y quedan por debajo de lo que el modelo detecta bien o de la duración necesaria para una huella confiable. Un humano los asigna **por contexto conversacional**, no por la acústica de esos 300 ms.
3. **Umbral y número de hablantes mal calibrados** para este dominio (el umbral de clustering está ajustado para audio limpio).
4. **Solapamientos.**
5. **Cambio de idioma de un mismo hablante**, si ocurre: puede partirlo en dos clusters. Distintas personas en distintos idiomas, en cambio, no complican.

**Qué no es el problema:** el idioma o las palabras en sí, para la parte acústica. El modelo no entiende "cachai". Las palabras **sí** son útiles para razonar sobre los turnos (fase 4).

**Experimento rápido para separar lo acústico de lo contextual.** Recortar backchannels sin contexto y pedirle a una persona que diga de quién son. Si duda, la información está en el contexto y no en el fragmento. Eso justifica la fusión con un LLM.

---

## 13. Plan de mejora por fases

Se ordena **de más barato a más caro**. Cada fase se mide contra el mismo conjunto de evaluación (fase 0). **Regla de oro: un cambio a la vez, medido con la DER desglosada.**

### Fase 0 — Conjunto de evaluación y línea base (obligatorio antes de todo)

**Anotación:**
- Elegir entre 30 y 60 minutos representativos de **varias** grabaciones distintas: distintos tipos de degradación, formatos (entrevista, conversación, multitud) y cantidad de hablantes.
- Dividirlos en **dev** (para ajustar) y **test** (que no se toca hasta el final), **separando por grabación y por hablante**, no por fragmentos de un mismo audio.
- Anotar en formato **RTTM** (y opcionalmente UEM para las regiones evaluadas). Herramientas: Audacity con pistas de etiquetas exportadas a RTTM, ELAN, Praat, o prodi.gy/Label Studio si están disponibles.
- **Guía de anotación** (escribirla y fijarla):
  - Precisión de los bordes (por ejemplo ±100 ms).
  - Cómo se anotan los backchannels: anotarlos siempre, aunque sean cortos.
  - Cómo se anotan los solapamientos: ambas etiquetas.
  - Qué hacer con la multitud: una etiqueta especial `CROWD`, excluida de la DER o puntuada aparte.
  - Qué hacer con la música, las risas y el ruido.
  - Hablantes no identificables: `UNK_x`.
- **Medir el desempeño humano:** que una segunda persona anote un subconjunto y se calcule la DER entre anotadores. Ese es el techo práctico.

**Línea base:** pyannote 3.1 + normalización de loudness con la configuración actual. Registrar DER total y desglosada por archivo y por tipo de degradación.

```python
# Boceto: evaluación DER desglosada
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
refs = load_rttm("dev_reference.rttm")      # {uri: Annotation}
hyps = load_rttm("dev_hypothesis.rttm")

for uri, ref in refs.items():
    comp = metric(ref, hyps[uri], detailed=True)
    total = comp["total"]
    print(uri,
          f"DER={comp['diarization error rate']:.3f}",
          f"miss={comp['missed detection']/total:.3f}",
          f"fa={comp['false alarm']/total:.3f}",
          f"conf={comp['confusion']/total:.3f}")
print("DER global:", abs(metric))
```

### Fase 1 — Ajustes sin reentrenar

1. Pasar **`num_speakers`** cuando se conoce, o **`min/max_speakers`** cuando se puede acotar. Suele dar una ganancia grande en audio difícil.
2. **Optimizar los hiperparámetros** sobre dev: `clustering.threshold`, `segmentation.min_duration_off`, `clustering.min_cluster_size`. Se puede usar el `Optimizer` de `pyannote.pipeline` (búsqueda bayesiana con Optuna) con −DER como objetivo. Es optimización de caja negra, en el espíritu de ESOpt.
3. Reevaluar **DiariZen y community-1 con el mismo preprocesamiento** que ganó con la 3.1, y también optimizar sus hiperparámetros, para comparar en igualdad de condiciones.
4. Probar si el **pipeline con VBx** (community-1 o DiariZen) gana una vez que el canal esté compensado (fase 2).

```python
# Boceto: instanciar pyannote 3.1 con hiperparámetros propios
from pyannote.audio import Pipeline
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=TOKEN)
pipe.instantiate({
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.70},
})
diar = pipe("audio.wav", min_speakers=2, max_speakers=5)
```
*(Verificar los nombres exactos de los parámetros con `pipe.parameters(instantiated=True)` en la versión instalada).*

### Fase 2 — Diagnóstico de degradaciones y preprocesamiento adaptativo

**Principio:** **una cadena fija con parámetros por archivo**, estimados a partir de medidas objetivas de su degradación. Es un sistema adaptativo, en el sentido de la sección 8.

#### 2.1 Cómo se ve cada degradación

| Degradación | Espectrograma | Otra vista | Medida automática |
|---|---|---|---|
| Zumbido de red | Líneas horizontales en 50, 100, 150 Hz… | Picos estrechos en el LTAS | Energía en ±2 Hz de k·50 Hz frente a la vecindad (dB) |
| Siseo de cinta | Neblina pareja en frecuencias altas, presente también en silencios | LTAS plano y alto arriba de ~3 kHz | Piso de ruido por banda, medido en los silencios (VAD) |
| Ancho de banda limitado | Todo oscuro sobre cierta frecuencia | Caída abrupta en el LTAS | Frecuencia de corte (por ejemplo, donde el LTAS cae 30 dB bajo su máximo en voz) |
| Mala ecualización | Coloración global | **Inclinación o jorobas en el LTAS (Welch)** frente a una voz de referencia | Pendiente espectral en dB/octava; diferencia por bandas contra la referencia |
| Clics y crepitaciones | Líneas verticales finas que cruzan todas las frecuencias | Impulsos en la forma de onda | Conteo de impulsos (curtosis local, detector de outliers en la derivada) |
| Saturación | Manchas que cubren todo el espectro en las partes fuertes | **Picos aplanados** en la forma de onda | Porcentaje de muestras cerca de ±máx; histograma con acumulación en los extremos |
| Wow/flutter | Los armónicos ondulan periódicamente | Pitch con modulación lenta (0,5–10 Hz) | Desviación del tono de una referencia o del pitch en tramos sostenidos |
| Reverberación | Colas borrosas tras las sílabas | — | Estimación de T60 o C50 (por ejemplo, modelos de estimación ciega) |
| Dropouts | Huecos verticales | Caídas bruscas de nivel | Tramos con caída de energía mayor a X dB y duración menor a Y ms |
| Nivel bajo o variable | — | Envolvente RMS | Loudness integrado (LUFS) y rango de loudness (LRA) |
| SNR global | — | — | Energía de voz frente a energía de silencios (con VAD) |

Para inspeccionar a mano: **Sonic Visualiser**, Audacity (vista de espectrograma) o iZotope RX si está disponible. En Python: `librosa`, `scipy.signal.welch`, `pyloudnorm`.

```python
# Boceto: reporte de degradaciones por archivo
import numpy as np, soundfile as sf, scipy.signal as ss, pyloudnorm as pyln

def degradation_report(path, vad_mask=None):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    rep = {}
    rep["lufs"] = pyln.Meter(sr).integrated_loudness(x)
    rep["clip_ratio"] = float(np.mean(np.abs(x) > 0.99 * np.max(np.abs(x))))
    f, P = ss.welch(x, sr, nperseg=8192)              # LTAS
    PdB = 10 * np.log10(P + 1e-20)
    # Zumbido 50 Hz y armónicos
    def band(fc, bw): return PdB[(f > fc - bw) & (f < fc + bw)]
    rep["hum_db"] = {k*50: float(band(k*50, 2).max() - np.median(band(k*50, 15)))
                     for k in range(1, 7)}
    # Frecuencia de corte aproximada
    peak = PdB[(f > 200) & (f < 4000)].max()
    above = f[(PdB > peak - 30) & (f > 1000)]
    rep["cutoff_hz"] = float(above.max()) if above.size else None
    # Pendiente espectral 300–3400 Hz (dB/octava)
    m = (f > 300) & (f < 3400)
    rep["tilt_db_per_oct"] = float(np.polyfit(np.log2(f[m]), PdB[m], 1)[0])
    # SNR aproximado si hay máscara VAD (booleana a la tasa de muestreo)
    if vad_mask is not None:
        ps, pn = np.mean(x[vad_mask]**2), np.mean(x[~vad_mask]**2) + 1e-20
        rep["snr_db"] = float(10 * np.log10(ps / pn))
    return rep
```

#### 2.2 Cadena de preprocesamiento recomendada (en orden)

Cada paso se activa o se gradúa según el reporte. **Cada paso se valida con la DER**, porque mejorar cómo suena el audio no garantiza mejorar la diarización.

1. **Formato:** mono, remuestreo a 16 kHz (lo que espera pyannote), quitar el offset DC.
2. **Pasa-altos suave** a 60–80 Hz (retumbe y ruido de manipulación). Cuidado de no cortar voces graves.
3. **Notch** en 50 Hz y sus armónicos, **solo si `hum_db` supera un umbral** (por ejemplo 10 dB). Usar un Q alto para no dañar la voz.
4. **De-click**, solo si el conteo de impulsos es significativo.
5. **Compensación de canal (ecualización):**
   - Calcular el **LTAS solo sobre los tramos con voz** del archivo (Welch + VAD).
   - Compararlo con un **LTAS de referencia** de voz limpia, por ejemplo promediado sobre un corpus moderno en español.
   - Diseñar un EQ correctivo suave (FIR de fase mínima o lineal) igual a la diferencia, **acotado** (por ejemplo ±12 dB) y **suavizado** por bandas de tercio de octava.
   - **No subir bandas sin señal**: arriba de la frecuencia de corte, o donde el SNR por banda es bajo. Ahí solo se amplifica el siseo.
   - *Por qué funciona:* el canal es aproximadamente un filtro LTI h. En el dominio log-espectral o cepstral, la convolución con h se vuelve una **suma constante**. Es la idea de la **CMN** (*cepstral mean normalization*), aplicada aquí sobre la forma de onda porque pyannote recibe audio y no características.
   - **Alternativa:** si el entrenamiento llega a usar características propias, aplicar CMN o CMVN por archivo directamente sobre ellas.
6. **Reducción de ruido suave:** resta espectral o filtro de Wiener, con el perfil de ruido estimado en los silencios (por ejemplo, `noisereduce` en modo estacionario). La intensidad se gradúa con el SNR. **Evitar los eliminadores neuronales agresivos** o compararlos explícitamente: pueden borrar los rasgos que distinguen a los hablantes.
7. **Corrección de wow/flutter**, solo si es severo. Es un proceso especializado (por ejemplo, la herramienta Capstan); dejarlo para el final.
8. **Normalización de loudness** (por ejemplo, −23 LUFS), que ya demostró ayudar. Opcionalmente, una compresión o AGC suave por tramos para hablantes lejanos o débiles, **validada con la DER**, porque puede aplanar diferencias útiles.

**Cómo graduar ("mismo set de filtros con distinto peso"):** se define una función que lleva de las métricas del reporte a los parámetros (umbral del notch, ganancia máxima del EQ, intensidad del denoise). Esas funciones se calibran sobre dev, a mano o con búsqueda de caja negra (Optuna), usando −DER como objetivo.

**Ablation obligatorio:** línea base → +HPF → +notch → +EQ → +denoise → +loudness, y también cada paso quitado del pipeline completo. Reportar la DER desglosada en cada caso.

### Fase 3 — Clustering con memoria temporal, enrolamiento y aprendizaje activo

1. **VBx o un HMM sobre los embeddings:** agrega el supuesto de que un hablante tiende a seguir hablando. Estabiliza las decisiones en tramos ruidosos. Ya viene en community-1 y DiariZen, así que conviene reevaluarlo tras la fase 2.
2. **Enrolamiento (pasar de clustering a verificación):** una persona marca entre 10 y 20 s limpios de cada hablante. Se calculan sus centroides y cada ventana se asigna por similitud coseno, con un umbral de "desconocido". Explota que el humano **sí** distingue las voces y convierte un problema difícil (clustering a ciegas) en uno más fácil (verificación).
3. **Aprendizaje activo:** el sistema solo pide confirmación humana en los tramos de **baja confianza** (margen pequeño entre los dos centroides más cercanos, segmentos cortos, solapamientos). Cada respuesta actualiza los centroides.
4. **Normalización de puntajes** (por ejemplo, S-norm con una cohorte del mismo dominio) para calibrar los umbrales en audio degradado.
5. **Promediar embeddings** de varias ventanas del mismo segmento y descartar las ventanas muy cortas al calcular los centroides.

### Fase 4 — Fusión con un LLM sobre la transcripción

**Orden correcto: primero la acústica, después el LLM corrige.** Solo con el texto, el LLM no tiene información de la voz y estimaría hablantes por plausibilidad (alucinaría).

Flujo:
1. pyannote entrega segmentos con etiquetas de hablante **y su confianza** (margen de similitud, probabilidades de segmentación).
2. Alinear la transcripción de Whisper palabra por palabra con los segmentos (alineamiento forzado tipo WhisperX, a nivel de palabra).
3. Construir una transcripción etiquetada, por ejemplo `[SPK1 0.92] ¿y usted dónde estaba ese día? [SPK2 0.55] ya [SPK1 0.48] mm, en la casa…`.
4. Pedirle al LLM que proponga **correcciones de etiqueta por palabra o segmento** con justificación, usando la estructura del diálogo:
   - Las preguntas suelen recibir respuesta de otra persona.
   - Los backchannels ("ya", "mm", "claro", "cachai") vienen del que **escucha**.
   - Los nombres y vocativos ("gracias, Jorge") dan pistas de identidad.
   - La continuidad temática y la gramatical.
5. **Aceptar el cambio solo donde la confianza acústica es baja** (por ejemplo, < τ). Donde pyannote está segura, manda la voz. Ajustar τ sobre dev.
6. Salida estructurada (JSON) para reconstruir el RTTM corregido y medir la DER.

**Referencia:** *DiarizationLM* (Google, 2024) usa un LLM para corregir la diarización a partir de la transcripción. **Riesgos:** los errores de ASR en audio degradado se propagan; el costo es mayor; el LLM puede "normalizar" turnos atípicos. Medir siempre contra la DER.

### Fase 5 — Fine-tuning con datos del dominio y degradación simulada

1. **Fine-tuning del modelo de segmentación** de pyannote con RTTM del archivo propio. pyannote lo soporta de forma nativa, y unas pocas horas anotadas suelen dar la mayor ganancia cuando el dominio es distinto. Ataca la voz no detectada, las falsas alarmas, los solapamientos y los segmentos cortos.
2. **Fine-tuning del modelo de embeddings** (WeSpeaker/ECAPA), **si en la fase 0 domina la confusión**.
3. **Aumentación con degradación simulada** sobre voz limpia en español, más el material real:
   - Filtros pasabanda y paso bajo con corte aleatorio entre 3 y 7 kHz.
   - Curvas de EQ aleatorias (inclinación de ±6 dB/octava, jorobas).
   - Saturación suave (tanh o curvas de cinta).
   - Siseo y ruido de color con SNR entre 0 y 20 dB.
   - Zumbido de 50 Hz con armónicos.
   - Clics y crepitaciones.
   - Wow/flutter (modulación de tasa de muestreo de 0,5 a 10 Hz y 0,1 a 1%).
   - Reverberación con respuestas al impulso reales o simuladas.
   - Códecs y remuestreos.

   Herramientas: `audiomentations`, `torch-audiomentations`, librerías de respuestas al impulso. **Objetivo:** que el modelo aprenda a **ignorar el canal en general**, no solo el de este archivo.
4. **Evitar sobreajuste a voces concretas:** con pocos hablantes anotados, el modelo puede memorizar esas voces en lugar de aprender el canal. Validar siempre con **grabaciones y hablantes no vistos**.

### Fase 6 — Memoria entre grabaciones (banco de voces)

Si los mismos hablantes aparecen en varios archivos (locutores, entrevistadores, figuras públicas de la época):
- Mantener un **banco de voces persistente**: centroides por identidad, con metadatos y número de muestras.
- Al procesar un archivo nuevo, primero diarizar y después **emparejar los clusters con el banco** por similitud. Si hay coincidencia, se usa la identidad conocida; si no, se crea una nueva.
- Actualizar los centroides con un promedio ponderado (adaptativo) y **guardar las huellas por tipo de canal**, porque la misma voz en dos cintas distintas puede quedar lejos en el espacio de embeddings.
- Revisión humana de las fusiones dudosas: unir dos identidades por error es costoso.

### Fase 7 (opcional) — Identidad audiovisual

Cuando hay video, usar el video original en blanco y negro para reconocer rostros y detectar al hablante activo, y fusionarlo con la diarización acústica. Detalle completo en el **Anexo A**. No requiere colorización.

---

## 14. Casos especiales

- **Backchannels y segmentos muy cortos:** la información está sobre todo en el **contexto** (fase 4), no en el fragmento. Anotarlos en la referencia, considerar un postproceso que los asigne al hablante que **no** tiene el turno principal, y evaluar la DER con y sin ellos.
- **Solapamientos:** pyannote 3.x detecta solapamientos en la segmentación. El fine-tuning de la fase 5 ayuda. La separación de fuentes con un solo micrófono (Conv-TasNet, SepFormer) o la diarización de punta a punta (EEND) son opciones caras; considerarlas solo si los solapamientos dominan el error.
- **Multitud (cánticos, coros, público):** **no es diarización**. Acústicamente es una sola fuente, y un humano tampoco separa a cada persona. Se etiqueta como clase `CROWD`, con un detector simple (clasificador de eventos de audio o heurística de energía y armonicidad), y se excluye del clustering. Aquí sí hay un límite aleatorio real: en mono, las voces individuales ya están mezcladas.
- **Entrevistas con muchas personas en secuencia (tipo vox populi):** si cada persona aparece una sola vez, el problema se parece más a **detectar cambios de hablante** que a agrupar voces. VBx o un modelo temporal ayudan; conviene no fijar `max_speakers` bajo.
- **Muchos hablantes en general:** hay más pares confundibles y el umbral se vuelve más delicado. Usar `min/max_speakers` cuando se pueda estimar y, si es posible, enrolamiento.
- **Una misma persona que cambia de idioma:** riesgo de que se parta en dos clusters. Revisar si ocurre en el archivo.
- **Distintos hablantes en distintos idiomas:** no es problema; incluso ayuda.

---

## 15. Generalización: qué aprende realmente el modelo

- La diarización **acústica** casi no depende del idioma ni de las palabras. Las huellas se entrenan para ignorar *qué* se dice.
- Al afinar con "Chile 1980", el modelo aprende sobre todo:
  1. **El canal** (equipos, cinta, ecualización), que se transfiere a otros archivos **si los equipos de grabación eran parecidos**, sin importar el país.
  2. **El ritmo de la conversación** (duración de los turnos, frecuencia de interrupciones y backchannels, solapamientos), que cambia según la cultura y el formato (radio, entrevista, debate) y afecta a la segmentación **por su duración y timing, no por su significado**.
  3. **El riesgo de memorizar voces**, si se anotan pocos hablantes.
- Por orden de impacto en este caso: degradación del audio, que es lo que más pesa, luego el ritmo conversacional y el número de hablantes, y en último lugar el idioma y las palabras.
- **Para que generalice** a otros archivos (por ejemplo, Argentina o Colombia en los 70): entrenar con **muchas degradaciones distintas** (fase 5.3), mezclar formatos y validar con colecciones no vistas.
- La **capa de LLM** (fase 4) sí depende del idioma y del estilo, y los LLM actuales manejan bien las variantes del español. Conviene incluir en el prompt pistas del registro local (muletillas chilenas como backchannels típicos).

---

## 16. Protocolo experimental

- **Un cambio a la vez.** Cada experimento con un ID, su configuración (YAML/JSON), el commit, la DER total y desglosada por archivo, y notas.
- **Conjuntos fijos:** dev para ajustar y test intocable hasta el final. Partición por grabación y por hablante.
- **Desglosar los resultados** por tipo de degradación (según el reporte de la fase 2), por formato (entrevista, conversación) y por número de hablantes, para saber *dónde* ayuda cada cambio.
- **Comparar con el techo humano** (la DER entre anotadores).
- **Escuchar los errores:** revisar a oído y en espectrograma los 20 peores segmentos de cada experimento.
- **Ojo con optimizar demasiado sobre dev**: si hay muchos hiperparámetros y poco dev, se sobreajusta. Confirmar en test.
- **Reproducibilidad:** fijar semillas y las versiones de pyannote, torch y de los modelos.

**Tabla de seguimiento sugerida:**

| ID | Cambio | DER | Miss | FA | Conf | Δ frente a la línea base | Notas |
|---|---|---|---|---|---|---|---|
| B0 | 3.1 + loudness (actual) | | | | | — | |
| F1a | + num_speakers | | | | | | |
| F1b | + umbral optimizado | | | | | | |
| F2a | + EQ de canal por LTAS | | | | | | |
| … | | | | | | | |

---

## 17. Preguntas abiertas

1. ¿Qué mide exactamente el "~70%" actual? ¿Hay RTTM de referencia?
2. ¿Se aplicó la misma normalización de loudness a DiariZen y community-1?
3. ¿Se conoce o se puede acotar el número de hablantes por archivo?
4. ¿Se repiten hablantes entre grabaciones (locutores, entrevistadores)?
5. ¿Cuántas horas totales hay, y cuántas se pueden anotar?
6. ¿Qué formatos predominan: entrevista, conversación, radio, actos públicos con multitud?
7. ¿Hay cambios de idioma, o solo español?
8. ¿Cuál es el uso final? (¿Transcripción por hablante para archivo histórico? ¿Búsqueda por persona?) Eso define si conviene priorizar la DER o la identificación entre archivos.
9. ¿Qué recursos de cómputo hay disponibles para el fine-tuning?
10. ¿Qué proporción de los audios tiene video asociado? Y de esos, ¿cuántos tienen al hablante en cámara, entrevistador fuera de cámara o narrador en off?
11. ¿Está validada la sincronía audio-video en las transferencias?

---

## 18. Glosario

- **AAM-softmax:** pérdida angular con margen aditivo, usada para entrenar embeddings de hablante.
- **Backchannel:** intervención breve del oyente ("ya", "mm") que no toma el turno.
- **CMN/CMVN:** normalización de la media (y varianza) cepstral por archivo; elimina el efecto de un canal LTI fijo.
- **Collar:** margen de tolerancia alrededor de los bordes al calcular la DER.
- **DER:** *Diarization Error Rate* = (miss + FA + confusión) / voz total.
- **ECAPA-TDNN / x-vector / WeSpeaker ResNet:** modelos de embeddings de hablante.
- **EEND:** diarización neuronal de punta a punta.
- **Embedding (huella de voz):** vector que representa la identidad de un hablante en un tramo.
- **Epistémica / aleatoria:** incertidumbre por falta de conocimiento (reducible) frente a variabilidad intrínseca (irreducible).
- **LMS / NLMS / RLS:** algoritmos de filtrado adaptativo.
- **LTAS:** espectro promedio de largo plazo (Welch sobre todo el archivo o sobre los tramos con voz).
- **LTI:** sistema lineal e invariante en el tiempo.
- **Powerset (segmentación):** codifica como clases todas las combinaciones de hablantes activos, incluidos los solapamientos.
- **RTTM:** formato estándar de anotación de diarización.
- **S-norm:** normalización simétrica de puntajes con una cohorte.
- **VAD:** detección de actividad de voz.
- **ASD (Active Speaker Detection):** detección de qué rostro visible está hablando.
- **SyncNet:** modelo que estima la sincronía entre los labios y el audio.
- **VBx:** clustering bayesiano con un HMM sobre x-vectors.
- **Wow/flutter:** variaciones lentas o rápidas de la velocidad de la cinta que modulan el tono.

---

## 19. Referencias

- Zheng et al. *Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements*. arXiv:2608.17310.
- Salimans et al. (2017). *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.
- Vaswani et al. (2017). *Attention Is All You Need*.
- Bahdanau, Cho, Bengio (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Ramsauer et al. (2020). *Hopfield Networks is All You Need*.
- Gu et al. (2021). *S4*; Gu & Dao (2023). *Mamba*.
- Haykin. *Adaptive Filter Theory*. Widrow & Stearns. *Adaptive Signal Processing*.
- Bredin et al. *pyannote.audio* (documentación y tarjetas de los modelos 3.1 y community-1).
- Plchot, Landini et al. *VBx: Bayesian HMM clustering of x-vector sequences*.
- DiariZen (BUT Speech@FIT), repositorio y paper asociado.
- Wang et al. (2024). *DiarizationLM: Speaker Diarization Post-Processing with Large Language Models*.
- Desplanques et al. (2020). *ECAPA-TDNN*.
- Bain et al. (2023). *WhisperX*.
- Tao et al. (2021). *TalkNet: Is Someone Speaking?* (ASD); Light-ASD; LoCoNet.
- Chung & Zisserman (2016). *Out of Time: Automated Lip Sync in the Wild* (SyncNet).
- Deng et al. *ArcFace*; InsightFace (detección y embeddings de rostro).

---

## Anexo A — Identidad audiovisual con el video original en blanco y negro (opcional)

> Este anexo pertenece al **repositorio de diarización**. Usa el **video original en blanco y negro**; **no necesita colorización** ni depende del repositorio de colorización. Es un módulo **opcional**: cuando no hay video, o el video no aporta (narrador en off, entrevistador fuera de cámara), el pipeline de solo audio de las secciones 11–14 sigue funcionando sin cambios.

### A.1 Idea

Cuando hay video, el rostro de quien habla es una segunda evidencia de identidad, independiente de la voz. Dos piezas:
- **Reconocimiento de identidad visual:** detectar rostros, seguirlos dentro de cada plano, calcular sus embeddings y agruparlos entre planos y clips.
- **Detección del hablante activo (ASD):** determinar qué rostro visible está hablando en cada momento, correlacionando el movimiento de los labios con el audio. Modelos de referencia: TalkNet-ASD, Light-ASD, LoCoNet (verificar las versiones actuales). La combinación de ambas cosas se conoce como **diarización audiovisual**.

**Qué aporta a la diarización:**
- Resuelve tramos donde el audio duda (ruido, backchannels cortos): gana el rostro que mueve los labios.
- Vuelve a unir un hablante partido en dos clusters por un cambio de canal o micrófono: es el mismo rostro.
- Separa a dos personas con voces parecidas: son rostros distintos.
- Los nombres de la transcripción ("gracias, Jorge") o de rótulos en pantalla (con OCR) se asocian a la voz y al rostro a la vez.

### A.2 El blanco y negro no es un obstáculo

- La detección de rostros y sus embeddings (por ejemplo, RetinaFace/SCRFD con ArcFace en InsightFace) funcionan bien en escala de grises: basta replicar el canal de luminancia en los tres canales de entrada. La forma del rostro y la geometría importan más que el color.
- La ASD trabaja con recortes de la boca y el audio. Muchos modelos ya usan los recortes de rostro en gris.
- Para robustez, se puede afinar o calibrar con algunos ejemplos anotados del propio archivo (rostros de la época, con bigote o barba, en baja resolución).

### A.3 Preprocesamiento mínimo de video (propio de este repositorio)

Solo lo necesario para que funcionen la detección de rostros y la ASD. Es **independiente** de la restauración más completa que haga el repositorio de colorización.
1. **Desentrelazar**, si el material es entrelazado (`bwdif`/`yadif` en ffmpeg, o QTGMC). El entrelazado daña la detección de rostros y, sobre todo, el movimiento de la boca.
2. **Deflicker** suave de luminancia (`deflicker` en ffmpeg), para estabilizar la detección.
3. **Detección de planos** (PySceneDetect o TransNetV2), para segmentar los tracks de rostro.
4. Opcional: denoise ligero y corrección de contraste, solo si mejora la tasa de detección de rostros (medirlo).

### A.4 Prerrequisito: sincronía audio-video

En transferencias antiguas es común que el audio esté **desfasado** o que **derive** a lo largo del clip. La ASD necesita una sincronía fina, de decenas de milisegundos.
1. **Estimar el desfase** por clip con un modelo tipo **SyncNet**, o con la correlación entre la apertura de la boca y la energía del audio en tramos con un solo rostro hablando.
2. **Detectar la deriva:** estimar el desfase en varias ventanas a lo largo del clip. Si cambia, corregirlo.
3. **Si la sincronía no se puede establecer con confianza, no usar la ASD** en ese clip.

### A.5 Casos donde la ASD no aplica

| Caso | Qué pasa | Qué hacer |
|---|---|---|
| **Hablante en cámara** | Caso ideal | Fusión completa: la ASD confirma o corrige la diarización y vincula voz con rostro |
| **Entrevistador fuera de cámara** (muy frecuente) | Su voz nunca tiene un rostro que hable | Etiqueta `OFFSCREEN`: si la ASD da "ningún rostro visible habla" de forma consistente en los turnos de una misma voz. Sigue siendo un hablante válido |
| **Narrador en off** | Una voz sostenida sobre muchos planos, sin rostro que hable, a menudo con otro canal (grabada en estudio) | Clase `NARRADOR`: presencia en muchos planos y escenas, ASD negativa y canal distinto. Se excluye de la vinculación voz-rostro |
| **Solo audio** | No hay video | Pipeline de solo audio sin cambios. La tabla de identidades guarda la voz; si esa persona aparece con video en otro clip, se vincula ahí |
| **Audio doblado o postsincronizado** | Los labios no coinciden | Una ASD sistemáticamente baja con rostros visibles indica falta de sincronía: no usar la ASD en ese clip |
| **Multitud o coro** | No hay un rostro individual hablando | `CROWD` en el audio; el video no se usa para identidad |
| **Varios rostros, uno habla** | La ASD tiene que elegir | Puntuación de la ASD por rostro, más la restricción de que normalmente habla uno a la vez (salvo solapamiento detectado en el audio) |
| **Sincronía no establecida** | Desfase desconocido | No usar la ASD |

**Regla de fusión conservadora:** la evidencia visual **solo** cambia una decisión del audio cuando (a) la sincronía está validada, (b) la confianza de la ASD es alta y (c) la confianza acústica es baja. Es la misma lógica que la fusión con el LLM (fase 4).

### A.6 Tabla de identidades

```json
{
  "entity_id": "P0007",
  "type": "person",                       // person | narrator | offscreen | crowd
  "names": [{"value": "Jorge ...", "source": "transcript_llm", "confidence": 0.7,
             "human_verified": false}],
  "voice": {
    "centroids_by_channel": {"cinta_A": [/* embedding */], "estudio": [/* ... */]},
    "n_segments": 143,
    "offscreen_ratio": 0.05
  },
  "face": {"centroid": [/* embedding */], "n_tracks": 37, "best_frame_ref": "clip12_f0453.png"},
  "appearances": [{"clip": "clip12", "t0": 12.4, "t1": 47.9,
                   "evidence": ["voice", "face", "asd"]}],
  "provenance": {"created": "2026-11-24", "last_update": "...", "human_verified": true}
}
```

Es la evolución del banco de voces de la fase 6, con la modalidad visual agregada. Si en el futuro el repositorio de colorización quiere usar estas identidades, puede **leer** esta tabla y guardar sus colores en un archivo propio indexado por `entity_id`. Este repositorio no necesita saber nada del color.

### A.7 Lógica de fusión (boceto)

1. **Por clip**, de forma independiente:
   - Audio: diarización con clusters de voz y su confianza.
   - Video: tracks de rostro con embeddings, y la ASD por track y ventana si la sincronía es válida.
2. **Co-ocurrencia voz-rostro:** para cada par (cluster de voz, track de rostro), sumar el tiempo en que la ASD dice que ese rostro habla mientras suena esa voz. Con esa matriz de afinidad se hace la **asignación** (algoritmo húngaro), dejando la opción "sin rostro" para `OFFSCREEN` y `NARRADOR`.
3. **Correcciones cruzadas** (solo con alta confianza de la modalidad que corrige y baja de la corregida):
   - Dos clusters de voz asignados con fuerza al **mismo rostro** → candidatos a unirse.
   - Un cluster de voz asignado a **dos rostros distintos** de forma sostenida → candidato a dividirse.
   - Un track de rostro de mala calidad durante un turno de voz confiable → hereda la identidad de esa voz.
   - Restricción de no-unión visual: dos rostros en el mismo fotograma no son la misma persona.
4. **Entre clips:** emparejar con la tabla global usando la voz (por canal), el rostro y los nombres. Las uniones dudosas pasan a revisión humana.

### A.8 Cómo medir el aporte

- DER de solo audio frente a DER audiovisual, **solo** sobre los segmentos con el hablante en cámara y la sincronía validada.
- La misma comparación sobre **todo** el set, para verificar que no empeora donde la ASD no aplica.
- Pureza y cobertura de los clusters de identidad frente a una anotación manual de quién es quién en una muestra.
- Tasa de detección de rostros con y sin el preprocesamiento mínimo (A.3).

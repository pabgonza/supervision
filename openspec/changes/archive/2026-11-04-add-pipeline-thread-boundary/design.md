## Context

`Pipeline.__iter__` lee la fuente y pasa cada item por los pasos en el mismo hilo. Las fuentes ya leen en su propio
hilo (cola de `*Capture`), así que lo que hoy no se solapa es el trabajo de los pasos entre sí. Motivación en
`proposal.md`.

## Goals / Non-Goals

**Goals:** solapar la etapa de inferencia de un frame con la de tracking del anterior, con la misma salida.

**Non-Goals:** más de dos etapas, varios workers por etapa (eso reordena y es lo que hace `PoolDetectorStep`), procesos
en vez de hilos.

## Decisions

- **No es un paso.** Un paso transforma un item y devuelve otro; el límite tiene que adueñarse de la iteración de lo
  anterior. `add_step` lo reconoce con `isinstance`, como al `PipelineSink`, y guarda su posición
  (`len(self.steps)`). Alternativa descartada: un `ThreadedStage(steps)` que envuelva pasos; obliga a reescribir cómo
  se arma el pipeline en el consumidor.
- **Un productor y un consumidor sobre una `queue.Queue` acotada** dan el orden sin números de secuencia ni reordenador.
  `queue_size` 2 por defecto: absorbe la variación de un frame sin sumar latencia.
- **Errores y fin por la misma cola**: `(error, None)` y un centinela `_END`; el que itera relanza el error recibido,
  así llega después de los items previos.
- **Parada**: el `finally` del generador marca un `Event`, vacía la cola (destraba un `put` bloqueado) y hace `join`
  con timeout; el `put` del worker usa timeout para ver el `Event`. `Pipeline.__iter__` cierra el generador del límite
  de forma explícita: tras una excepción en un paso posterior, el traceback lo mantiene vivo y cerrar solo el generador
  exterior no lo alcanza (lo encontró el unitario de error posterior).
- **Hilo daemon**: si la fuente queda bloqueada en una lectura, el `join` vence y el proceso puede terminar igual.

## Risks / Trade-offs

- [GIL] El solape solo existe donde el código suelta el GIL (OpenCV, NumPy, inferencia CUDA/TensorRT). Con Python puro
  en ambas etapas no hay ganancia. → Se mide en el consumidor (Orin NX, dos canales) antes de dejarlo.
- [Propiedad de los datos] Un paso previo que reutilice un buffer entre frames corrompería lo encolado. → Las fuentes
  del fork crean un array por lectura; el consumidor lo verifica con hash de cada frame a ambos lados del límite.
- [Tiempos] Los pasos posteriores ven el frame hasta `queue_size` frames más tarde; un `FPSCalculatorStep` después del
  límite mide la tasa de salida, igual que hoy.

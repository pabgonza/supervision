## Why

`Pipeline` corre todo en un hilo: cada frame cuesta la suma de sus pasos. En Ball Counter v2 (consumidor del fork) dos
canales densos en un Orin NX se turnan la GPU y quedan en ~55 fps; con el tracker optimizado (`speed-up-bytetrack`,
−35 %) el estimado sigue cerca de 59, bajo el objetivo de 60 (change `raise-detector-fps-dense-flow` del firmware,
tarea 7.4). Mientras el tracker de un frame corre en CPU, la GPU podría estar con la inferencia del siguiente.
`PoolDetectorStep` no sirve para eso: espera el resultado de su propio frame y no solapa nada.

## What Changes

- `sv.ThreadBoundary(queue_size=2)`: se agrega con `|` como un paso pero no toca los datos. Lo anterior (fuente y pasos
  previos) corre en un hilo worker; lo posterior (pasos siguientes y sink), en el hilo que itera. Cola FIFO acotada entre
  ambos.
- **Regla del change: misma salida.** Cada frame sale una vez, en orden, con los mismos pasos aplicados; un error antes
  del límite llega al que itera después de los frames previos; salir antes (`break`, error, sink que devuelve `False`)
  detiene el worker antes de cerrar la fuente. Un solo límite por pipeline.
- `Pipeline.add_step` lo reconoce (como al sink); `Pipeline.__iter__` arma las dos etapas. Sin límite, el pipeline corre
  igual que hoy.

## Capabilities

### New Capabilities
- `pipeline-thread-boundary`: pipeline en dos hilos sin cambio de salida.

### Modified Capabilities
<!-- ninguna -->

## Impact

- **Código**: `supervision/pipeline/core.py` (`ThreadBoundary`, `Pipeline.add_step`/`__iter__`, `_run_steps`);
  exportado en `supervision/pipeline/__init__.py` y `supervision/__init__.py`.
- **Tests**: `test/pipeline/test_thread_boundary.py`.
- **Docs**: docstring de `ThreadBoundary` (esquema, garantías, GIL, ejemplo) y `examples/pipeline/thread_boundary_demo.py`.
- **Consumidores**: Ball Counter v2 lo inserta en el detector tras la traslación de coordenadas y verifica la salida
  bit a bit contra la referencia de un hilo (tareas 7.4.3–7.4.5 del firmware).
- **Upstream**: candidato a PR (el framework de pipeline es propio del fork).

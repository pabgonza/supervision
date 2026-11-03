# Tasks

## 1. Implementación

- [x] 1.1 `ThreadBoundary` en `supervision/pipeline/core.py` (cola FIFO acotada, centinela de fin, errores por la cola,
  parada con `Event` y `join`), `Pipeline.add_step` (un solo límite) y `Pipeline.__iter__` con `_run_steps`; exportado
  en `supervision.pipeline` y `supervision`. Docstring con esquema, garantías, GIL y ejemplo.

## 2. Tests

- [x] 2.1 `test/pipeline/test_thread_boundary.py`: misma salida y orden con el límite en cuatro posiciones (pasos con
  demora al azar, uno con `filter`); error en la fuente y en un paso previo tras los items previos; error posterior,
  `break` y sink que corta detienen el worker antes de cerrar la fuente; cola acotada; último item al fin de la fuente;
  segundo límite y `queue_size` 0 rechazados.
  Hecho el 2026-11-03: 13 passed (tres corridas). El de error posterior falló primero: el worker seguía vivo al cerrar
  la fuente; arreglado cerrando el generador del límite de forma explícita en `Pipeline.__iter__`.
- [x] 2.2 `ruff check`, `ruff format --check` y `pytest test/pipeline test/tracker` en verde.
  Hecho: 140 passed, 11 skipped.

## 3. Docs

- [x] 3.1 `examples/pipeline/thread_boundary_demo.py`: el mismo pipeline con y sin límite, fps de cada uno y
  comprobación de salida igual.
  Hecho el 2026-11-03 en la PC (video de 221 en 1080p sin ROI, modelo de bolas `.pt`, 1500 frames): 66,6 → 71,4 fps,
  mismos ids. En la PC la GPU sobra; la medición que cuenta es la de dos canales en el Orin NX (firmware 7.4.5).
  Los fps se miden siempre sin anotar; `--output` agrega dos pasadas sin cronometrar que escriben el video anotado
  (cajas, ids y trazas) para revisarlo a ojo: los dos `.mp4` salen idénticos byte a byte. Con la PC cargada los fps
  varían entre corridas (20 a 73): sirven solo con la máquina quieta.

## Workflow follow-up

- Commit en `develop` y push junto con `speed-up-bytetrack`; el firmware sube su pin (`raise-detector-fps-dense-flow`,
  tareas 7.4.5 y 7.5) tras verificar la salida bit a bit contra su referencia de un hilo (7.4.3, 7.4.4).
- Archivar tras integrarlo en el firmware.

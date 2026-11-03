# Tasks

## 1. Implementación

- [x] 1.1 `StreamCapture`: `self.dropped_frames = 0` en `__init__`; en `update()`, descartar tras una lectura exitosa (no
  antes de leer) y sumar 1 solo cuando el `get_nowait()` del descarte saca un frame (si el consumidor lo tomó antes, no hubo descarte); `dropped_frames` en `get_health()` y en
  su docstring.
- [x] 1.2 Test nuevo `test/utils/test_capture.py` con un `cv2.VideoCapture` falso inyectado con `monkeypatch` que entrega
  frames numerados: (a) cola 4, sin consumidor, 10 frames → `dropped_frames == 6` y en la cola quedan los 4 últimos;
  (b) consumidor que lee cada frame → 0; (c) captura recién creada → 0 y las claves previas presentes.
- [x] 1.3 `ruff check` y `ruff format --check` de los archivos tocados y `pytest test/utils` en `supervision_dev` en verde.
  Hecho el 2026-11-02: 51 passed; `test_capture.py` 5 de 5 corridas en verde.

## Workflow follow-up

- Commit en `develop` y push; el firmware sube su pin (`raise-detector-fps-dense-flow`, tarea 3.2).
- Archivar tras integrarlo en el firmware.

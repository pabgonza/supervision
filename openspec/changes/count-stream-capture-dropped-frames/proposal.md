## Why

`StreamCapture` descarta el frame más viejo de su cola cuando está llena (el consumidor no alcanza al stream) y no lo
cuenta. Quien la usa no tiene forma de saber cuántos frames perdió. En Ball Counter v2 (consumidor del fork) eso cuesta
bolas sin aviso: con flujo denso a 60 fps, procesar 35 de cada 60 frames baja la cuenta 1,6 %, y con 30 de 60 el tracker
se rompe (banco rg0002, 2026-11-02; change `raise-detector-fps-dense-flow` del firmware).

## What Changes

- `StreamCapture` cuenta cada frame que descarta por cola llena desde que se crea.
- `get_health()` agrega la clave `dropped_frames` (entero, 0 al crear la captura). Las claves actuales no cambian.
- El descarte pasa a hacerse cuando llega un frame nuevo y la cola está llena, no antes de leer: hoy, si el stream se
  detiene con la cola llena, se descarta un frame sin que llegue otro que lo reemplace (y el conteo quedaría en K + 1).
- Sin cambios de firma: un contador y una clave más.
- Fuera de alcance: `WebcamVideoCapture` tiene el mismo descarte; se deja para otro change, como en
  `serialize-stream-capture-reconnect`.

## Capabilities

### New Capabilities
- `stream-capture-dropped-frames`: reporte de los frames que `StreamCapture` descarta cuando el consumidor no alcanza al
  stream.

### Modified Capabilities
<!-- ninguna -->

## Impact

- **Código**: `supervision/utils/capture.py` (`StreamCapture.__init__`, `update`, `get_health`). Test nuevo
  `test/utils/test_capture.py` con un `cv2.VideoCapture` falso (sin red).
- **Consumidores**: `StreamSource` ya pasa `get_health()` en cada frame (`health`); Ball Counter v2 publicará el valor en
  su `/metrics` tras reconstruir la imagen del detector.
- **Convive con** `serialize-stream-capture-reconnect` (abierto, sin implementar): ambos tocan `update()`; el contador
  queda en la rama del descarte, que ese change no modifica.
- **Upstream**: candidato a PR (sin lógica de Ball Counter).

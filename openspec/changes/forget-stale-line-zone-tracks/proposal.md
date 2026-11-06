## Why

`LineZone` guarda en `crossing_state_history` una entrada por cada `tracker_id` que pasa cerca de la línea, y nunca la
borra. Solo la vacía `LineZoneStep.reset()`, que nadie llama en operación continua. Esto se midió en la PC el 2026-11-05:
son **701 B por `tracker_id`**.

| Escenario | Bolas | Crecimiento |
|---|---|---|
| Ball Counter v2 con flujo denso | ~26 bolas/s por cámara | ~1,6 GB/día por cámara |
| rg0001, flujo actual | ~30 mil bolas/día | ~21 MB/día (coincide con lo observado) |

Con flujo denso, un Jetson de 16 GB con dos cámaras se queda sin RAM en 3–4 días. En el banco no se veía porque el detector
reiniciaba al final de cada pasada del video.

## What Changes

- **Parámetro nuevo** `forget_after_frames` en `LineZone` y en `LineZoneStep`. Es opcional y por defecto `None`, que
  mantiene el comportamiento actual bit a bit.
- **Con un entero N ≥ 1:**
  - `LineZone` cuenta los `trigger()` como frames;
  - olvida (historia y última vez visto) cada `tracker_id` que no apareció en las detecciones durante más de N frames.
- **Equivalencia con el tracker:** con N bastante mayor que el tiempo que el tracker conserva un track perdido
  (`max_time_lost` de `ByteTrack`), un id olvidado ya no vuelve y las cuentas no cambian. N = `max_time_lost` no basta:
  `update_with_detections` puede dejar fuera de su salida un track vivo (medido: hasta 2 × `max_time_lost` frames). Se
  recomienda 4 × `max_time_lost`.
- **Validación:** un valor que no sea `None` ni un entero ≥ 1 da `ValueError` en el constructor.
- **`LineZoneStep.reset()`** limpia también el estado nuevo.

## Capabilities

### New Capabilities
- `line-zone-track-history`: qué guarda `LineZone` por cada `tracker_id` para decidir un cruce y cuánto tiempo lo
  conserva.

### Modified Capabilities
<!-- ninguna -->

## Impact

- **Código:**
  - `supervision/detection/line_zone.py`: `LineZone.__init__` y `trigger`;
  - `supervision/pipeline/steps/tracking.py`: `LineZoneStep.__init__` y `reset`;
  - tests en `test/detection/test_line_counter.py`.
- **API:** solo agrega un argumento con keyword y default. Quien no lo usa no ve ningún cambio.
- **Consumidores:** Ball Counter v2 pasa `forget_after_frames = 4 × ByteTrack.max_time_lost` de su tracker. Es el change
  `bound-detector-line-zone-memory` del firmware, y llega al equipo al reconstruir la imagen del detector (release v1.5.0).
- **Riesgo:** bajo con el default. Si alguien usa un N corto, un objeto que reaparece en la salida del tracker tras más
  de N frames empieza la historia de cero y puede no contarse. El docstring lo dice.
- **Upstream:** candidato a PR, porque el crecimiento ocurre con cualquier uso continuo de `LineZone`.

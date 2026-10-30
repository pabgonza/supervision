## Why

`ByteTrack` no logra confirmar objetos rápidos: un track recién nacido (sin confirmar) no pasa por la predicción del Kalman (`multi_predict` solo corre sobre tracked + lost) y además nace con velocidad 0 (`KalmanFilter.initiate`). En el frame siguiente se compara contra su caja de nacimiento con el umbral fijo 0.7, y si el objeto avanzó más de ~la mitad de su caja (IoU × conf < 0.3) no casa: se borra y nace otro, frame tras frame, sin id estable. En Ball Counter v2 (consumidor del fork) esto se midió sobre videos reales de la línea de Molycop (2026-10-29): bolas de ~53 px a ~30 px/frame quedan sin id en el 53–73 % de las detecciones cerca de la línea de conteo y el conteo cae a ~20 % del real (58 contra 283 en un tramo contado a mano). Subir `minimum_matching_threshold` no lo arregla porque ese umbral no se aplica a los tracks sin confirmar.

## What Changes

- Nuevo parámetro opcional `velocity_prior_alpha` en `ByteTrack` (y su paso directo en `ByteTrackerStep`). Por defecto `None`: comportamiento idéntico al actual, bit a bit.
- Con un valor en `(0, 1]`, el tracker:
  - mantiene una **velocidad previa** (`vx`, `vy` en px/frame): promedio móvil exponencial, con ese `alpha`, de la velocidad Kalman media de los tracks confirmados con historia suficiente; en frames sin tracks así conserva el último valor (no decae a 0);
  - **inicia** cada track nuevo con esa velocidad en vez de 0;
  - **predice** con el Kalman los tracks sin confirmar antes de asociarlos, igual que ya hace con los confirmados y perdidos.
- `ByteTrack.reset()` vuelve la velocidad previa a 0.
- `KalmanFilter.initiate()` acepta una velocidad inicial opcional (por defecto 0, sin cambio).
- Valor fuera de `(0, 1]` → `ValueError` en el constructor.

## Capabilities

### New Capabilities
- `bytetrack-track-initialization`: cómo nace y se confirma un track en `ByteTrack` (estado inicial del Kalman, predicción y asociación de tracks sin confirmar), incluida la velocidad previa opcional.

### Modified Capabilities
<!-- ninguna: el repo no tiene specs de tracking -->

## Impact

- **Código**: `supervision/tracker/byte_tracker/core.py` (`ByteTrack.__init__`, `update_with_tensors`, `reset`), `kalman_filter.py` (`initiate`), `single_object_track.py` (`activate`), `supervision/pipeline/steps/tracking.py` (`ByteTrackerStep.__init__`). Tests nuevos en `test/tracker/test_byte_tracker.py` con objetos sintéticos rápidos.
- **API**: solo agrega un argumento con keyword y default; sin cambios para quien no lo usa.
- **Consumidores**: Ball Counter v2 lo expone como `tracker.velocity_prior_alpha` en su YAML de canal (change propio en el firmware); llega al detector tras reconstruir la imagen.
- **Riesgo**: bajo con el default. Activado, supone que los objetos de la escena se mueven de forma parecida (cinta, línea de producción, tráfico en un sentido); en escenas con movimientos opuestos la velocidad previa media puede ser mala para los tracks nuevos, que igual corrigen con su primera actualización.
- **Upstream**: candidato a PR (no tiene lógica de Ball Counter).

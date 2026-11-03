## Why

En Ball Counter v2 (consumidor del fork) ByteTrack cuesta ~5,5 ms por frame en el Orin NX con flujo denso (~12 bolas por
frame) y corre en el mismo hilo que la inferencia: con dos canales densos el detector queda en ~55 fps y el objetivo es
≥ 60 (banco rg0002, 2026-11-03; change `raise-detector-fps-dense-flow` del firmware). En la PC el perfil muestra que el
tiempo se va en trabajo por pista desde Python, no en la asignación: la conversión de cajas (`tlwh`/`tlbr`, ~100
llamadas por frame), el update de Kalman pista a pista (con las envolturas de scipy) y la covarianza de movimiento armada
con un `np.diag` por pista.

## What Changes

- Mismo algoritmo, menos trabajo por pista: cajas de todas las pistas en una operación, covarianza de movimiento sin
  bucle, update de Kalman sin pasos redundantes (proyección con una matriz de selección, envolturas de scipy).
- **Regla del change: solo rendimiento.** Para la misma secuencia de detecciones, la salida es la misma frame a frame:
  mismo `tracker_id` para cada detección y mismas pistas de salida; en el estado de Kalman solo cabe redondeo
  (≤ 1e-4 px). Sin parámetros nuevos ni defaults distintos; la API pública no cambia.
- Referencia grabada antes de optimizar (fork `0dc53f92`) y test de equivalencia que la exige.
- Fuera de alcance: pipeline en dos hilos (solo si esto no alcanza; tarea 7.4 del firmware).

## Capabilities

### New Capabilities
- `bytetrack-performance`: optimizaciones de ByteTrack que no cambian su salida.

### Modified Capabilities
<!-- ninguna -->

## Impact

- **Código**: `supervision/tracker/byte_tracker/` (`core.py`, `single_object_track.py`, `kalman_filter.py`,
  `matching.py`).
- **Tests**: `test/tracker/test_byte_tracker_equivalence.py` con `test/tracker/data/bytetrack_reference.npz` (tramos de
  los videos 221 y 253 del banco, un clip de rg0001 y tres secuencias sintéticas; cuatro configuraciones); con
  `BYTETRACK_REFERENCE_DIR` también las grabaciones completas y el estado de Kalman de cada pista.
- **Consumidores**: Ball Counter v2 sube el pin del fork y repite la medición de dos canales (tareas 7.5 del firmware).
- **Upstream**: candidato a PR (sin lógica de Ball Counter), salvo la velocidad previa que ya es propia del fork.

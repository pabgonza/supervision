# Tasks

## 1. Referencia y test de equivalencia (antes de optimizar)

- [x] 1.1 `test/tracker/bytetrack_replay.py` (reproducción por frame y secuencias sintéticas) y
  `make_bytetrack_reference.py`: graba con `0dc53f92` el `tracker_id` de cada detección en tramos de 221 (1500 frames),
  253 (1500) y rg0001 (3000) y en tres sintéticas, para las configuraciones `base`, `prior` (0.1), `mcf3` y
  `prior_mcf3`; y las grabaciones completas con el estado de cada pista en un directorio local. Verificación: dos
  corridas dan archivos iguales.
  Hecho el 2026-11-03: fixture de 764 kB; las grabaciones completas (221, 249, 253 y rg0001) en
  `ballcounter_firmware/data/temp/tracker_ref/`. Dos corridas idénticas. Las sintéticas llevan además objetos lentos
  ocultos 20–90 frames (sin ellos un `lost_track_buffer` de 29 en vez de 30 no se detectaba).
- [x] 1.2 `test/tracker/test_byte_tracker_equivalence.py`. Verificación: verde con `0dc53f92`; rojo con un cambio
  deliberado de comportamiento (otro `minimum_matching_threshold` en la reproducción).
  Hecho: verde con `0dc53f92` (35 passed, también tras `git stash` de la optimización); `minimum_matching_threshold`
  0.79 cambia 14 473 ids en 221 y 258 en `synthetic0`; `lost_track_buffer` 29 cambia 1128 en `synthetic0`.

## 2. Optimización

- [x] 2.1 Medir el tiempo por frame antes (`bench.py` del firmware, `data/temp/tracker_ref/`, 221, 3000 frames) y el
  perfil por función.
  Antes: 1,343 ms/frame en la PC (12,4 det/frame); el perfil: `tlwh`/`tlbr` ~100 llamadas por frame, update de Kalman
  por pista, `multi_predict` con un `np.diag` por pista.
- [x] 2.2 Optimizaciones del perfil, una a una con el test en verde tras cada una (también con
  `BYTETRACK_REFERENCE_DIR`).
  Hecho: `STrack.multi_tlbr` (cajas de todas las pistas con la misma aritmética), update de Kalman con
  `dpotrf`/`dpotrs` directos y proyección por slicing, `multi_predict` con sumas en vez de productos por `_motion_mat`
  (identidad + dt = 1: dos términos no nulos por entrada, exacto). 1,343 → 0,894 (cajas + update) → **0,837 ms/frame
  (−38 %)**. Con
  `BYTETRACK_REFERENCE_DIR`: 51 passed (ids, pistas y estado ≤ 1e-4 px en las cuatro grabaciones completas).
- [x] 2.3 `ruff check`, `ruff format --check` y `pytest test/tracker` en verde. Verificación: tiempo por frame después
  en la PC y en rg0002.
  2026-11-03: ruff y pytest en verde; PC 1,343 → 0,837 ms (−38 %); rg0002 (Orin NX, imagen del detector) 3,388 →
  2,201 ms (−35 %).

## Workflow follow-up

- Commit en `develop` y push; el firmware sube su pin (`raise-detector-fps-dense-flow`, tarea 7.5).
- Archivar tras integrarlo en el firmware.

# Tasks

## 1. Kalman y track

- [x] 1.1 `KalmanFilter.initiate(measurement, velocity=None)`: con `velocity` (vx, vy) inicia `mean[4:6]` con ella, el resto igual; docstring actualizado. Verificar con un test que sin argumento la media de velocidad es 0 y con `(30, 0)` queda `mean[4:6] == (30, 0)`.
- [x] 1.2 `STrack.activate(kalman_filter, frame_id, velocity=None)` pasa la velocidad a `initiate`. Verificar con el test de 2.3 (track nuevo nace con la velocidad previa).

## 2. ByteTrack

- [x] 2.1 `ByteTrack.__init__(..., velocity_prior_alpha=None)`: valida `None` o `(0, 1]` (`ValueError` si no), guarda la velocidad previa en `(0, 0)`; docstring de clase con el parámetro, cuándo usarlo y el valor recomendado 0.1. Verificar con test de valores inválidos (0, -0.1, 1.5).
- [x] 2.2 En `update_with_tensors`, con el parámetro activo: predecir los tracks sin confirmar antes de su asociación, pasar la velocidad previa a `activate` en el paso 4 y, al final, actualizar el promedio móvil con los tracks de salida con `tracklet_len >= 3` (conservar si no hay). `reset()` vuelve la velocidad previa a 0. Verificar con los tests de 2.3.
- [x] 2.3 Tests en `test/tracker/test_byte_tracker.py` con objetos sintéticos (caja 50 px): (a) default y `None` dan los mismos ids en una secuencia mixta; (b) objeto que ya viene a 30 px/frame sin velocidad previa no obtiene id estable; (c) con `alpha=0.1`, tras un calentamiento con objetos que aceleran de 5 a 30 px/frame, objetos nuevos que llegan a 30 px/frame conservan un único id; (d) la velocidad previa converge cerca de (30, 0) y se conserva en frames sin tracks elegibles; (e) `reset()` la vuelve a 0; (f) objetos lentos (2 px/frame) siguen con un id cada uno. Correr `pytest test/tracker` en `supervision_dev` en verde.

## 3. Pipeline step

- [x] 3.1 `ByteTrackerStep(..., velocity_prior_alpha=None)` lo pasa a `ByteTrack`; docstring de Args actualizado. Verificar con un test que el tracker del step recibe el valor.

## 4. Calidad y validación

- [x] 4.1 `ruff check` y `ruff format --check` sobre los archivos tocados en verde; `pytest test/tracker test/pipeline` (si existe) en verde. Queda un E501 previo en `tracking.py:329` (docstring de otra clase, no tocado).
- [x] 4.2 Replay sobre las detecciones capturadas de los tres videos de Ball Counter (`linea221`, `linea249`, `linea253`) con el fork modificado (sin parches en memoria): con `velocity_prior_alpha=None` reproduce los conteos actuales (out 1105 / in 203 en `linea249`, 5678 en `linea221`, out 1874 en `linea253` con mmt 0.8), y con 0.1 da resultados iguales o mejores que el replay de validación (detecciones sin id cerca de la línea ≤ 0,5 % en 249 y 221, ≤ 7,4 % en 253; tramo 2507–3158 de 249 cerca de 283). Resultado 2026-10-30: sin velocidad previa reproduce exacto; con 0.1 y mmt 0.8: 249 out 5775 / in 0 (0,5 %), tramo 281; 221 in 5825 (0,0 %); 253 out 4530 / in 1 (9,0 %; con mmt 0.95, 4618 y 7,5 %), dentro de ±0,1 % del replay de validación.

## Workflow follow-up

- Commit en `develop` del fork y push para que el firmware lo instale al reconstruir la imagen del detector.
- Archivar el change tras integrarlo en el firmware (`enable-tracker-velocity-prior`).
- Evaluar PR upstream a `roboflow/supervision`.

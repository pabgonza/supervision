## 1. LineZone

- [x] 1.1 `LineZone.__init__(..., forget_after_frames=None)`:
  - valida `None` o entero ≥ 1 (`ValueError` si no);
  - inicializa el contador de frames y el dict de última aparición;
  - documenta el parámetro en el docstring: qué hace, que se recomienda 4 × `max_time_lost` del tracker (la salida del
    tracker puede omitir un track vivo por más de `max_time_lost`) y el riesgo de un valor menor.
  Verificación: test de valores inválidos (0, -5, 1.5).
- [x] 1.2 `trigger()` con el parámetro activo:
  - incrementa el contador al empezar cada llamada, también en las salidas tempranas;
  - registra la última aparición de todos los `tracker_id` presentes;
  - cada N frames borra de la historia y de las apariciones los ids con más de N frames sin aparecer.
  Sin el parámetro, el camino es el de hoy. Verificación: tests de 3.1.

## 2. LineZoneStep

- [x] 2.1 `LineZoneStep(..., forget_after_frames=None)` lo pasa a `LineZone`, con el docstring de Args actualizado.
  `reset()` limpia también el contador y las apariciones. Verificación: test de que el `LineZone` del paso recibe el
  valor, y de que tras `reset()` todo queda vacío o en cero.

## 3. Tests y calidad

- [x] 3.1 En `test/detection/test_line_counter.py`, con detecciones sintéticas:
  - (a) sin el parámetro y con `None`, las mismas cuentas e historia en una secuencia que cruza, se pierde y reaparece;
  - (b) con N=30, un id ausente 30 frames que reaparece al otro lado cuenta el cruce;
  - (c) con N=30, un id ausente más de 60 frames ya no está en la historia;
  - (d) 100 000 ids, 10 nuevos por frame: la historia nunca pasa de 610 entradas (10 × (2N + 1)).
  - (e) un id presente pero fuera de los límites durante más de N frames conserva su historia (D2).
  Verificación: `conda run -n supervision_dev pytest test/detection test/pipeline` en verde.
- [x] 3.2 Calidad. Verificación: `ruff check` y `ruff format --check` en verde sobre los archivos tocados.
- [x] 3.3 Replay de equivalencia, sin parches. Se corre `replay_fork.py` de Ball Counter
  (`data/temp/tracker_ref/` del firmware) con el fork, sobre `linea221`, `linea249`, `linea253` y `rg0001_molycom_04`:
  - parámetros: `velocity_prior_alpha` `None` y 0.1, con `forget_after_frames = 4 × tracker.max_time_lost`;
  - se agrega al replay el tamaño máximo de la historia.
  Verificación:
  - cuentas `in`/`out` acumuladas por frame idénticas con y sin el parámetro;
  - historia máxima ≤ ids por frame × (2N + 1).
  Hecho el 2026-11-06 con `data/temp/tracker_ref/replay_forget.py <dir> 4`: 8 de 8 idénticos, historia máxima 243 (cota
  9620 en linea221). Con 1 × `max_time_lost` falla linea253 sin velocidad previa: de ahí el 4× (design D4).
- [x] 3.4 Costo. Verificación: `bench.py` del firmware con y sin el parámetro sobre `linea221`, con una diferencia del
  tiempo por frame del tracker más la línea dentro del ruido (< 2 %).
  Hecho el 2026-11-06 (`bench.py <npz> --line [--forget 4]`, PC, dos rondas): 0,885 → 0,893 y 0,844 → 0,833 ms/frame
  (+0,9 % y −1,3 %, ruido).

## Workflow follow-up

- Commit en `develop` del fork y push, para que el firmware lo instale al reconstruir la imagen del detector.
- Integración: change `bound-detector-line-zone-memory` del firmware (v1.5.0).
- Archivar tras integrarlo en el firmware. Evaluar un PR upstream a `roboflow/supervision`.

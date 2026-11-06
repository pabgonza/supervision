## Context

`LineZone.trigger()` toma cada detección dentro de los límites de la línea que no la cruza con sus anclas y agrega su lado
al `deque` de su `tracker_id` (`crossing_state_history`, un `defaultdict`). Cada entrada pesa ~701 B: el `deque` con su
`maxlen`, la clave y la ranura del dict. Nada la borra.

`ByteTrack` descarta un track perdido cuando `frame_id - track.frame_id > max_time_lost`, donde
`max_time_lost = int(frame_rate / 30 * lost_track_buffer)`. Sus ids son monótonos, así que un id descartado no vuelve.

## Goals / Non-Goals

**Goals:**
- Memoria de `LineZone` acotada en uso continuo, sin cambiar las cuentas cuando el plazo es ≥ al del tracker.
- Default idéntico al actual.

**Non-Goals:**
- Que `LineZone` conozca al tracker o lea su `max_time_lost`: el consumidor pasa el valor.
- Acotar `class_id_to_name` o los contadores por clase: crecen con las clases, no con los objetos.
- Cambiar `PolygonZone` u otras zonas.

## Decisions

### D1. Plazo en frames, contado por `trigger()`

`LineZone` no recibe tiempos ni números de frame, así que cada llamada a `trigger()` cuenta como un frame. Es la misma
unidad del tracker cuando los dos corren una vez por frame, como en `ByteTrackerStep` → `LineZoneStep`.

Alternativa descartada: un plazo en segundos con reloj de pared. Depende del fps real y rompe la equivalencia exacta con el
tracker.

### D2. «Visto» = presente en las detecciones, no «escrito en la historia»

Se registra el último frame de **todo** `tracker_id` presente en las detecciones, aunque ese frame no escriba historia
(fuera de los límites o con las anclas a ambos lados).

Un objeto que se queda más de N frames junto a la línea, fuera de los límites, sigue vivo para el tracker. Si se olvidara
su historia, al volver contaría distinto.

Costo: un dict `tracker_id → último frame` del mismo orden que la historia.

### D3. Poda cada N frames

- **Cuándo:** cuando el contador es múltiplo de N, se recorren los ids vistos y se borran de los dos dicts los que llevan
  más de N frames sin aparecer.
- **Cota:** un id vive como máximo 2N frames después de su última aparición, así que las entradas quedan acotadas por
  ids por frame × (2N + 1).
- **Costo:** O(entradas) cada N frames, amortizado O(ids por frame).

Alternativa descartada: podar en cada frame con un heap o un `OrderedDict` por último visto. Da la cota exacta N, pero
agrega reordenamiento por frame sin que la memoria lo pida.

### D4. Equivalencia con N = 4 × `max_time_lost`, medida

ByteTrack descarta un track perdido tras `max_time_lost` frames. Pero un track vivo puede faltar de la **salida** de
`update_with_detections` por más tiempo: ese método cruza las detecciones con las cajas del Kalman con IoU 0,5 y deja
fuera las que no casan. Si su id se olvida antes de que el track muera, la cuenta cambia.

Medido en el replay de Ball Counter (cuatro videos, `max_time_lost` = 60, con y sin velocidad previa, 2026-11-06):

| Plazo | Cuentas por frame idénticas |
|---|---|
| 1 × `max_time_lost` | 7 de 8 (falla linea253 sin velocidad previa, con huecos de hasta 120 frames en la salida) |
| 2 × | 8 de 8, en el borde (hueco máximo = plazo) |
| 4 × | 8 de 8 |

Decisión: se recomienda 4 × `max_time_lost`. La memoria sigue acotada: 243 entradas de historia como máximo en el replay,
contra 5869 y creciendo sin el plazo. `LineZone` no promete equivalencia exacta: depende del tracker que la alimenta.

## Risks / Trade-offs

- **[N corto]** Un objeto que falta de la salida del tracker más de N frames reaparece sin historia y puede no contarse.
  → El docstring recomienda 4 × `max_time_lost` del tracker y explica por qué.
- **[Ids reutilizados tras `ByteTrack.reset()`]** Ya pasa hoy sin el parámetro; olvidar no lo empeora.
- **[Paso vectorizado]** El registro de «visto» agrega un `dict.update` por frame con los ids de las detecciones. Es del
  orden de microsegundos con ~10–20 ids por frame. → Se mide con `bench.py` de Ball Counter, que no debe cambiar fuera del
  ruido.

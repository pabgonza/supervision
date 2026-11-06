## Purpose

Qué estado guarda `LineZone` por cada `tracker_id` para decidir si cruzó la línea, y cuánto tiempo lo conserva, de modo
que un uso continuo (días, millones de objetos) tenga memoria acotada sin cambiar las cuentas.

## ADDED Requirements

### Requirement: Historia por tracker_id sin olvido por defecto

Sin `forget_after_frames`, o con `None`, `LineZone` SHALL conservar la historia de cada `tracker_id` hasta un `reset()`.
Las cuentas, los `last_crossed_*_ids` y `crossing_state_history` MUST ser idénticos a los de la versión anterior.

#### Scenario: Default sin cambios

- **GIVEN** una secuencia de detecciones con ids que cruzan, se pierden y reaparecen
- **WHEN** se procesa con `LineZone` sin el parámetro y con `forget_after_frames=None`
- **THEN** ambas corridas dan las mismas cuentas por frame y la misma historia que la versión anterior

### Requirement: Olvido de ids inactivos

Con `forget_after_frames=N` (entero ≥ 1), `LineZone` SHALL olvidar todo `tracker_id` que no aparezca en las detecciones de
más de N llamadas a `trigger()` seguidas. Cada llamada cuenta como un frame, venga vacía o no. Un id visto dentro de los
últimos N frames MUST conservar su historia completa.

#### Scenario: Reaparece dentro del plazo

- **GIVEN** `forget_after_frames=30` y un id visto a un lado de la línea
- **WHEN** desaparece 30 frames y reaparece al otro lado
- **THEN** el cruce se cuenta igual que sin el parámetro

#### Scenario: Ausente más del plazo

- **GIVEN** `forget_after_frames=30` y un id visto a un lado de la línea
- **WHEN** pasan más de 60 frames sin que aparezca
- **THEN** el id ya no está en `crossing_state_history`

#### Scenario: Memoria acotada con muchos objetos

- **GIVEN** `forget_after_frames=30` y 100 000 ids distintos, 10 nuevos por frame, cada uno visto un solo frame
- **WHEN** se procesan todos
- **THEN** `crossing_state_history` nunca supera 10 × (2 × 30 + 1) entradas

#### Scenario: Equivalente al tracker

- **GIVEN** detecciones reales pasadas por `ByteTrack` y `forget_after_frames` igual a 4 × su `max_time_lost`
- **WHEN** se procesan con y sin el parámetro
- **THEN** las cuentas `in`/`out` de cada frame son idénticas

### Requirement: Valor inválido

`LineZone` y `LineZoneStep` SHALL rechazar con `ValueError` un `forget_after_frames` que no sea `None` ni un entero ≥ 1.

#### Scenario: Cero o negativo

- **WHEN** se construye `LineZone` con `forget_after_frames=0` o `-5`
- **THEN** el constructor lanza `ValueError`

### Requirement: Paso de pipeline

`LineZoneStep` SHALL aceptar `forget_after_frames` y pasarlo a su `LineZone`. Su `reset()` MUST dejar el estado de olvido
como recién construido.

#### Scenario: Reset del paso

- **GIVEN** un `LineZoneStep` con `forget_after_frames=30` que ya procesó frames
- **WHEN** se llama `reset()`
- **THEN** la historia, los ids vistos y el contador de frames quedan vacíos o en cero

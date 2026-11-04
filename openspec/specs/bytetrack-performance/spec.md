# bytetrack-performance Specification

## Purpose
Fija qué garantiza una optimización de rendimiento de `ByteTrack`: misma salida que la versión de referencia, con los mismos parámetros y la misma API.

## Requirements

### Requirement: Optimizaciones de ByteTrack sin cambio de salida

Una optimización de ByteTrack MUST entregar, para la misma secuencia de detecciones y los mismos parámetros, el mismo
`tracker_id` para cada detección que la versión de referencia (`0dc53f92`). El estado de Kalman de cada pista de salida
MUST coincidir salvo redondeo (≤ 1e-4 px). Los parámetros, sus defaults y la API pública MUST quedar iguales.

#### Scenario: Grabaciones de flujo denso y ralo

- **GIVEN** las detecciones grabadas de los videos 221 y 253 del banco y de un clip de rg0001
- **WHEN** se procesan con o sin `velocity_prior_alpha` y con `minimum_consecutive_frames` 1 o 3
- **THEN** cada detección recibe el mismo `tracker_id` que en la referencia

#### Scenario: Casos de borde sintéticos

- **GIVEN** secuencias con pistas que se pierden y vuelven, pistas que mueren, trayectorias que se cruzan, frames sin
  detecciones y detecciones de baja confianza
- **WHEN** se procesan con las mismas cuatro configuraciones
- **THEN** cada detección recibe el mismo `tracker_id` que en la referencia

#### Scenario: Estado de las pistas

- **WHEN** se procesan las grabaciones completas
- **THEN** en cada frame las pistas de salida son las mismas, en el mismo orden, y sus cajas de Kalman difieren a lo
  más 1e-4 px de la referencia

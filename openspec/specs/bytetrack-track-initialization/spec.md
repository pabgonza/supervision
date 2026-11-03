# bytetrack-track-initialization Specification

## Purpose
Define cómo nace y se confirma un track en `ByteTrack`: el estado de movimiento con que se inicia, si se predice antes de asociarlo y la velocidad previa opcional que permite seguir objetos rápidos.

## Requirements

### Requirement: Default behavior is unchanged without a velocity prior
Sin `velocity_prior_alpha` (o con `None`), `ByteTrack` SHALL producir exactamente los mismos `tracker_id` que la versión anterior para cualquier secuencia de detecciones: los tracks nuevos nacen con velocidad 0 y los tracks sin confirmar no se predicen.

#### Scenario: Same ids as before
- **WHEN** se procesa la misma secuencia de detecciones con `ByteTrack()` y con `ByteTrack(velocity_prior_alpha=None)`
- **THEN** ambos devuelven los mismos `tracker_id` frame a frame

#### Scenario: Fast object without prior is not tracked
- **WHEN** un objeto de caja 50 px avanza 30 px por frame y el tracker no tiene velocidad previa
- **THEN** el objeto no obtiene un `tracker_id` estable (comportamiento conocido que la velocidad previa corrige)

### Requirement: Velocity prior parameter is validated
`ByteTrack` y `ByteTrackerStep` SHALL aceptar `velocity_prior_alpha` como `None` o un número en `(0, 1]`, y SHALL rechazar cualquier otro valor con `ValueError` al construirse.

#### Scenario: Invalid alpha
- **WHEN** se construye `ByteTrack(velocity_prior_alpha=0)` o `ByteTrack(velocity_prior_alpha=1.5)`
- **THEN** se lanza `ValueError`

#### Scenario: Step forwards the parameter
- **WHEN** se construye `ByteTrackerStep(velocity_prior_alpha=0.1)`
- **THEN** su tracker usa velocidad previa con `alpha` 0.1

### Requirement: Velocity prior tracks the motion of confirmed tracks
Con `velocity_prior_alpha` definido, tras cada actualización el tracker SHALL mover la velocidad previa hacia la velocidad media de los tracks confirmados con historia suficiente, como promedio móvil exponencial con ese `alpha`; sin tracks así, SHALL conservar el valor anterior. Parte en 0 y `reset()` la vuelve a 0.

#### Scenario: Prior converges to the scene motion
- **WHEN** varios objetos se mueven a 30 px por frame hacia la derecha durante decenas de frames
- **THEN** la velocidad previa queda cerca de (30, 0) px por frame

#### Scenario: Prior is held without confirmed tracks
- **WHEN** hay frames con detecciones pero sin tracks confirmados con historia suficiente
- **THEN** la velocidad previa no cambia

#### Scenario: Reset clears the prior
- **WHEN** se llama `reset()`
- **THEN** la velocidad previa vuelve a (0, 0)

### Requirement: New tracks start with the velocity prior and are predicted
Con `velocity_prior_alpha` definido, cada track nuevo SHALL iniciar su estado de movimiento con la velocidad previa vigente, y los tracks sin confirmar SHALL predecirse con su filtro de movimiento antes de asociarse con las detecciones del frame.

#### Scenario: Fast object is tracked with prior
- **WHEN** el tracker tiene `velocity_prior_alpha=0.1`, ya confirmó objetos que avanzan 30 px por frame (por ejemplo, porque entraron más lentos y aceleraron) y luego aparecen objetos nuevos de caja 50 px que ya vienen a 30 px por frame en la misma dirección
- **THEN** cada objeto nuevo obtiene un único `tracker_id` y lo conserva mientras está visible

#### Scenario: Slow objects keep working
- **WHEN** objetos se mueven 2 px por frame con `velocity_prior_alpha=0.1`
- **THEN** cada objeto conserva un único `tracker_id`, igual que sin velocidad previa

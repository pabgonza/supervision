## ADDED Requirements

### Requirement: Pipeline en dos hilos sin cambio de salida

Un `Pipeline` con un `ThreadBoundary` MUST entregar los mismos datos, en el mismo orden, que el mismo pipeline sin él:
cada item de la fuente pasa una sola vez por los mismos pasos (con la semántica de `filter` de siempre) y sale una vez.
Los pasos previos al límite MUST correr en un hilo worker y los posteriores en el hilo que itera. Un pipeline MUST
aceptar a lo más un límite.

#### Scenario: Orden y completitud

- **GIVEN** una fuente de N items y pasos con duraciones al azar, uno de ellos con `filter`
- **WHEN** se itera con el límite al principio, entre pasos o al final
- **THEN** salen los N items en orden, cada uno con la misma traza de pasos que sin el límite, incluido el último

#### Scenario: Error antes del límite

- **WHEN** la fuente o un paso previo lanza una excepción en el item k
- **THEN** el que itera recibe los items 0 a k−1 y después la misma excepción, y la fuente queda cerrada

#### Scenario: Salida anticipada

- **WHEN** la iteración termina antes del fin de la fuente (`break`, excepción en un paso posterior o sink que devuelve
  `False`)
- **THEN** el worker se detiene antes de cerrar la fuente y la fuente deja de leerse

#### Scenario: Memoria acotada

- **GIVEN** `queue_size` q y una etapa posterior lenta
- **THEN** el worker no adelanta más de q + 1 items a la etapa posterior

#### Scenario: Un solo límite

- **WHEN** se agrega un segundo `ThreadBoundary`, o uno con `queue_size` < 1
- **THEN** se lanza `ValueError`

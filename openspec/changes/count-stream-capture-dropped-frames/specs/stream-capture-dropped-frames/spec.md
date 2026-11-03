## ADDED Requirements

### Requirement: Conteo de frames descartados por la captura de streams

La captura de streams SHALL contar cada frame que descarta porque su cola está llena y MUST exponer el total en su
estado de salud como `dropped_frames`: un entero que empieza en 0 al crear la captura y nunca disminuye mientras
existe. Las demás claves del estado de salud MUST mantenerse sin cambios.

#### Scenario: Consumidor detenido

- **GIVEN** una captura con cola de tamaño N cuyo consumidor no lee
- **WHEN** el stream entrega N + K frames
- **THEN** `dropped_frames` vale K

#### Scenario: Consumidor al ritmo del stream

- **WHEN** el consumidor lee cada frame antes de que la cola se llene
- **THEN** `dropped_frames` se mantiene en 0

#### Scenario: Captura nueva

- **WHEN** se crea una captura
- **THEN** su estado de salud trae `dropped_frames` igual a 0 junto a las claves que ya existían

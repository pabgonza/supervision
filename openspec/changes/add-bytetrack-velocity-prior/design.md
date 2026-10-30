# Design

## Context

En `ByteTrack.update_with_tensors` un track nace en el paso 4 (`STrack.activate` → `KalmanFilter.initiate`, velocidad 0) y queda sin confirmar (`is_activated=False`, salvo en el frame 1). En el frame siguiente solo se asocia en el bloque de tracks sin confirmar, con costo `1 − IoU·conf` y umbral fijo 0.7, contra la caja de nacimiento: ese grupo no entra en `multi_predict`, que solo recibe tracked + lost. Un objeto que avanza más de ~la mitad de su caja por frame nunca se confirma. Subir `minimum_matching_threshold` no lo cambia, porque ese umbral no se aplica a ese bloque.

Se reprodujo y se validó con un replay del tracker sobre las detecciones reales de tres videos de Ball Counter, parcheando la clase en memoria: solo predecir sin velocidad previa no cambia nada (con velocidad 0 la predicción deja la caja en su lugar); solo la velocidad previa recupera menos de la mitad; las dos juntas llevan las detecciones sin id cerca de la línea de 72,6 % a 0,0 % y el conteo de un tramo contado a mano a 281 contra 283.

## Goals / Non-Goals

**Goals:**
- Confirmar objetos rápidos que se mueven de forma coherente, con un parámetro opcional y sin cambiar el comportamiento por defecto.

**Non-Goals:**
- Velocidad previa por zona o por clase, o asociación por distancia de centros en vez de IoU.
- Cambiar los umbrales fijos (0.7, 0.5) de las asociaciones internas.
- Sembrar la velocidad previa desde fuera (ver Decisiones).

## Decisions

1. **Un solo parámetro activa las dos piezas** (velocidad previa + predicción de tracks sin confirmar). Por separado ninguna sirve, y atar la predicción al parámetro mantiene el default idéntico bit a bit (predecir cambia también la covarianza). Alternativa descartada: dos flags; más superficie sin caso de uso.
2. **Fuente de la velocidad previa**: media de `mean[4:6]` (vx, vy del centro, en px/frame) de los tracks de salida (`is_activated`) con `tracklet_len >= 3`. Con menos historia, el Kalman aún no convergió y arrastraría la media hacia 0. Es la regla validada en el replay.
3. **Promedio móvil exponencial con `alpha`** (`prior = (1−α)·prior + α·media`), recomendado 0.1: suaviza tracks ruidosos y sigue cambios de ritmo de la línea en ~1 s a 60 fps. **Sin tracks elegibles se conserva el valor**: el replay de validación lo hacía decaer hacia 0 en esos frames, lo que debilita justo los nacimientos que siguen a un hueco; la implementación lo conserva y se vuelve a medir sobre los tres videos. Los frames sin detecciones no pasan por el tracker (`ByteTrackerStep` los salta), así que tampoco lo tocan.
4. **Inyección en el nacimiento**: `KalmanFilter.initiate(measurement, velocity=None)` acepta vx, vy opcionales (va y vh siguen en 0); `STrack.activate(kalman_filter, frame_id, velocity=None)` lo pasa. Covarianza inicial sin cambios: el Kalman corrige rápido si el objeto va distinto.
5. **Predicción de tracks sin confirmar**: `STrack.multi_predict(unconfirmed, self.shared_kalman)` antes de su asociación, solo con el parámetro activo.
6. **Sin semilla externa**: medido sobre los tres videos, la velocidad previa llega al 90 % de su valor estable en 0,4–0,6 s desde el frame 1, porque siempre hay objetos que entran lentos y se confirman. Una semilla configurable (`initial_velocity`) queda como posible extensión si aparece un caso de arranque en frío sin objetos lentos.

## Risks / Trade-offs

- [Escena con movimientos opuestos o sin dirección dominante] → la media puede ser ~0 o engañosa; el comportamiento degrada hacia el actual, no peor que hoy salvo que la media apunte en contra del objeto. Es opcional y se documenta para escenas de flujo coherente.
- [Arranque en frío con todos los objetos ya rápidos] → la velocidad previa no se calienta y el tracker se comporta como hoy. Ver decisión 6.
- [Más costo por frame] → un `multi_predict` extra sobre los tracks sin confirmar y una media por frame; despreciable frente a la detección.

## Migration Plan

Sin migración: parámetro nuevo con default `None`. Rollback = no pasarlo. Los consumidores lo reciben al reinstalar el fork desde `develop` (Ball Counter: rebuild de la imagen del detector).

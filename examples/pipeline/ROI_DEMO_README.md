# ROI-based Pool Detection + Line Zone Counter Demo

Demo de detección de objetos usando ROI (Region of Interest) configurable, con detección paralela, tracking y conteo de cruces de línea.

## Características

- **ROI configurable**: Posición y tamaño personalizables o auto-centrado
- **Detección paralela**: Utiliza `PoolYOLODetectionStep` para alto rendimiento
- **Tracking**: Seguimiento de objetos con ByteTracker
- **Conteo de línea**: Cuenta cruces de objetos a través de una línea
- **Logging dual**: Registra eventos en socket TCP y archivo rotating log
- **Visualización**: Muestra ROI con rectángulo cyan en el frame original
- **Configuración YAML**: Parámetros en formato YAML con soporte para comentarios

## Archivos

- `roi_pool_line_zone_demo.py` - Script principal
- `config.yaml` - Configuración para stream RTSP de producción
- `config_test_webcam.yaml` - Configuración de prueba con webcam
- `config_custom_roi.yaml` - Ejemplo con ROI en posición específica
- `socket_client_test.py` - Cliente de prueba para socket TCP

## Uso

### Opción 1: Usar config.yaml (stream RTSP de producción)

```bash
cd examples/pipeline
conda run -n supervision_dev python roi_pool_line_zone_demo.py
```

### Opción 2: Usar config de prueba con webcam

```bash
cd examples/pipeline
conda run -n supervision_dev python roi_pool_line_zone_demo.py --config config_test_webcam.yaml
```

### Opción 3: Usar ROI personalizado

```bash
conda run -n supervision_dev python roi_pool_line_zone_demo.py --config config_custom_roi.yaml
```

### Opción 4: Sobrescribir parámetros desde línea de comandos

```bash
conda run -n supervision_dev python roi_pool_line_zone_demo.py --config config.yaml --pool-size 6 --conf 0.5
```

## Configuración YAML

### Configuración Básica

```yaml
# Video Source
source: stream  # webcam, file, o stream
input: rtsp://192.168.18.169:554/live1s3.sdp

# Detection Model
model: path/to/model.pt
device: cuda  # cuda o cpu
conf: 0.4  # Umbral de confianza (0.0-1.0)

# ROI Configuration
roi:
  x: null  # null = auto-center horizontally, or specify pixel coordinate
  y: null  # null = auto-center vertically, or specify pixel coordinate
  width: 640   # ROI width in pixels
  height: 640  # ROI height in pixels

# Pool Detection
pool_size: 4  # Number of parallel workers
queue_size: 10

# Tracking
track_threshold: 0.25
lost_buffer: 30
match_threshold: 0.8

# Line Zone (coordinates in FULL FRAME space, not ROI)
line_start: "640,0"
line_end: "640,720"

# Logging
socket_port: 7777
log_file: count.log

# Display
show_metrics: true
```

### Ejemplos de Configuración de ROI

#### 1. ROI Centrado (Default)
```yaml
roi:
  x: null  # Auto-center horizontally
  y: null  # Auto-center vertically
  width: 640
  height: 640
```

#### 2. ROI en Esquina Superior Izquierda
```yaml
roi:
  x: 0
  y: 0
  width: 640
  height: 640
```

#### 3. ROI en Cuadrante Inferior Derecho
```yaml
roi:
  x: 960   # Para frame 1920px de ancho
  y: 540   # Para frame 1080px de alto
  width: 640
  height: 640
```

#### 4. ROI Rectangular (No Cuadrado)
```yaml
roi:
  x: 100
  y: 200
  width: 800   # Más ancho
  height: 400  # Más bajo
```

## Logging de Cruces de Línea

Cada vez que un objeto cruza la línea, se registra:

### Socket TCP (puerto 7777)
```
2025-10-14T10:30:45.123456,1,0
2025-10-14T10:30:46.234567,0,1
```

### Archivo count.log (rotating, 10MB max, 5 backups)
Mismo formato que socket TCP.

### Consola
```
Line crossing: IN +1, OUT +0
```

## Pipeline Steps Utilizados

El demo utiliza los siguientes steps de supervision:

1. **sv.PoolYOLODetectionStep** - Detección paralela
2. **sv.ByteTrackerStep** - Tracking de objetos
3. **sv.LineZoneStep** - Conteo de cruces
4. **sv.TrackerAnnotatorStep** - Anotación de tracking
5. **sv.LineZoneAnnotatorStep** - Anotación de línea
6. **sv.CallbackStep** - Overlay de métricas
7. **sv.DisplaySink** - Visualización
8. **sv.VideoFileSink** - Guardado de video (opcional)

## Steps Personalizados

Además, se implementaron steps personalizados siguiendo el patrón de supervision:

1. **ROIExtractionStep** - Extrae ROI configurable
2. **CoordinateTranslationStep** - Traduce coordenadas ROI → frame completo
3. **ROIVisualizationStep** - Dibuja rectángulo del ROI
4. **FrameRestoreStep** - Restaura frame original para anotación
5. **LineCrossingLoggerStep** - Logging de eventos de cruce

## Flujo del Pipeline

```
Source (webcam/file/stream)
  ↓
ROIExtractionStep (extrae ROI configurable)
  ↓
PoolYOLODetectionStep (detección paralela en ROI)
  ↓
CoordinateTranslationStep (ROI coords → full frame coords)
  ↓
ByteTrackerStep (tracking)
  ↓
LineZoneStep (conteo de cruces)
  ↓
ROIVisualizationStep (dibuja rectángulo cyan)
  ↓
FrameRestoreStep (switch a frame original)
  ↓
TrackerAnnotatorStep (dibuja tracking)
  ↓
LineZoneAnnotatorStep (dibuja línea y conteos)
  ↓
LineCrossingLoggerStep (log eventos)
  ↓
CallbackStep (métricas overlay)
  ↓
DisplaySink / VideoFileSink
```

## Conectarse al Socket

Para leer los eventos de cruce en tiempo real desde otro programa:

```python
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 7777))

while True:
    data = client.recv(1024).decode('utf-8')
    if data:
        for line in data.split('\n'):
            if line:
                timestamp, in_increment, out_increment = line.split(',')
                print(f"Cruce detectado: {timestamp} - IN: +{in_increment}, OUT: +{out_increment}")
```

O usar el cliente de prueba incluido:

```bash
python socket_client_test.py
```

## Ventajas del ROI Configurable

### 1. **Performance Optimizado**
- Procesar 640x640 en vez de 1920x1080 = 2.8x más rápido
- Permite usar pools más grandes con menos recursos
- Reduce latencia de detección

### 2. **Precisión Mejorada**
- Elimina detecciones en áreas irrelevantes
- Reduce falsos positivos
- Enfoca recursos en zona de interés

### 3. **Flexibilidad**
- Adaptable a diferentes escenarios de cámara
- Múltiples zonas de interés con múltiples instancias
- Fácil reconfiguración sin cambiar código

### 4. **Casos de Uso**

#### Entrada/Salida de Edificios
```yaml
roi:
  x: 800    # Centrado en puerta
  y: 400
  width: 400
  height: 600
line_start: "1000,400"  # Línea vertical en puerta
line_end: "1000,1000"
```

#### Cinta Transportadora
```yaml
roi:
  x: null   # Centrado en cinta
  y: 300
  width: 1200  # ROI ancho para toda la cinta
  height: 400
line_start: "960,500"  # Línea perpendicular al movimiento
line_end: "960,700"
```

#### Cancha Deportiva - Zona Específica
```yaml
roi:
  x: 640    # Área de gol
  y: 200
  width: 640
  height: 480
line_start: "960,200"  # Línea de gol
line_end: "960,680"
```

## Métricas Mostradas en Pantalla

Cuando `show_metrics: true`:

```
Workers: 4
ROI: 640x640 at (640,220)
Detections: 5
Tracked: 3
IN Count: 12
OUT Count: 8
Processed: 1543
Dropped: 0
Input Queue: 2/10
Avg Inference: 45.3ms
```

## Notas Importantes

### Coordenadas de Línea
- Las coordenadas de `line_start` y `line_end` están en el **espacio del frame completo**, NO relativas al ROI
- Esto permite que la línea esté dentro o fuera del ROI según sea necesario
- Para línea dentro del ROI, calcular: `ROI_x + offset_in_roi`

### Tamaño del ROI
- ROI puede ser rectangular (no necesariamente cuadrado)
- Si ROI excede límites del frame, se ajusta automáticamente
- ROI mínimo recomendado: 320x320 para detección confiable
- ROI máximo: tamaño del frame

### Auto-Centrado
- Si `x: null` → ROI se centra horizontalmente
- Si `y: null` → ROI se centra verticalmente
- Útil para cámaras fijas con sujeto centrado

## Troubleshooting

### ROI no visible en frame
- Verificar que las coordenadas x,y estén dentro del frame
- Verificar que width/height sean apropiados
- El rectángulo cyan debe ser visible si ROI está correctamente configurado

### Detecciones fuera del ROI
- Las detecciones siempre ocurren SOLO dentro del ROI
- Si ves detecciones fuera, las coordenadas fueron traducidas correctamente
- Esto es comportamiento esperado: detección en ROI, visualización en frame completo

### Line zone no cuenta
- Verificar que la línea esté en coordenadas de frame completo
- Verificar que objetos pasen por la línea
- Verificar que objetos estén siendo trackeados (tracker_id presente)

### Performance bajo
- Reducir `pool_size`
- Reducir tamaño de ROI (`width`, `height`)
- Aumentar `conf` threshold
- Usar `device: cuda` si disponible

## Dependencias

```bash
pip install supervision ultralytics opencv-python numpy pyyaml
```

## Archivos de Configuración Incluidos

1. **`config.yaml`** - Producción con stream RTSP
   - ROI centrado 640x640
   - Pool size 4, GPU
   - Modelo custom de detección de pelotas

2. **`config_test_webcam.yaml`** - Pruebas con webcam
   - ROI centrado 640x640
   - Pool size 2, CPU
   - Modelo YOLOv8n estándar

3. **`config_custom_roi.yaml`** - Ejemplo ROI posicionado
   - ROI en cuadrante inferior derecho
   - Demuestra uso de coordenadas fijas
   - Incluye notas explicativas

# ROI Detection + Tracking + Line Zone Pipeline

Esquema visual del pipeline implementado en `examples/pipeline/roi_line_zone.py`

## Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROI LINE ZONE PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   SOURCE     │
                              │ (video/rtsp) │
                              └──────┬───────┘
                                     │ frame
                                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      CORE PIPELINE (Always runs)                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                     │
                    ┌────────────────┴────────────────┐
                    │    1. ROIExtractionStep         │
                    │  input: frame                   │
                    │  output: roi_frame              │
                    │  (Extrae región ROI del frame)  │
                    └────────────────┬────────────────┘
                                     │ roi_frame
                                     ▼
                    ┌────────────────┴────────────────┐
                    │    2. YOLODetectionStep         │
                    │  input: roi_frame               │
                    │  output: roi_detections         │
                    │  (Detección solo en ROI)        │
                    └────────────────┬────────────────┘
                                     │ roi_detections
                                     ▼
                    ┌────────────────┴────────────────┐
                    │  3. CoordinateTranslationStep   │
                    │  input: roi_detections          │
                    │  output: detections             │
                    │  (Traduce coords ROI→frame)     │
                    └────────────────┬────────────────┘
                                     │ detections
                                     ▼
                    ┌────────────────┴────────────────┐
                    │    4. ByteTrackerStep           │
                    │  input: detections              │
                    │  output: detections (con IDs)   │
                    │  (Tracking de objetos)          │
                    └────────────────┬────────────────┘
                                     │ detections + tracker_id
                                     ▼
                    ┌────────────────┴────────────────┐
                    │    5. LineZoneStep              │
                    │  input: detections              │
                    │  output: line_zone              │
                    │  (Cuenta cruces de línea)       │
                    └────────────────┬────────────────┘
                                     │ + line_zone
                                     ▼
                    ┌────────────────┴────────────────┐
                    │    6. FPSCalculatorStep         │
                    │  (Calcula FPS del pipeline)     │
                    └────────────────┬────────────────┘
                                     │ + fps
                                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          ANNOTATION PIPELINE (Si display OR output enabled)               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                     │
                    ┌────────────────┴────────────────┐
                    │   7. ROIVisualizationStep       │
                    │  (Dibuja rectángulo ROI)        │
                    │  color: amarillo (255,255,0)    │
                    └────────────────┬────────────────┘
                                     │ frame con ROI
                                     ▼
                    ┌────────────────┴────────────────┐
                    │   8. TrackerAnnotatorStep       │
                    │  (Boxes + Labels + Trayectorias)│
                    │  - Muestra tracker ID           │
                    │  - Muestra clase & confianza    │
                    │  - Traces de 30 frames          │
                    └────────────────┬────────────────┘
                                     │ frame anotado
                                     ▼
                    ┌────────────────┴────────────────┐
                    │   9. LineZoneAnnotatorStep      │
                    │  (Dibuja línea de conteo)       │
                    │  + Muestra IN/OUT counts        │
                    └────────────────┬────────────────┘
                                     │ frame completo
                                     ▼
                    ┌────────────────┴────────────────┐
                    │   10. CallbackStep              │
                    │   (Si show_metrics = True)      │
                    │   - Overlay de métricas         │
                    │   - FPS, detections, tracked    │
                    │   - Line counts, inference time │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              SINKS                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
          ┌──────────────────┐          ┌──────────────────┐
          │  DisplaySink     │          │ VideoFileSink    │
          │  (si enabled)    │          │  (si enabled)    │
          │  - Ventana CV    │          │  - Guarda MP4    │
          │  - Presiona 'q'  │          │  - Codec XVID    │
          └──────────────────┘          └──────────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                            (Si ambos enabled:
                             usa MultiSink)
```

## Flujo de Datos (Data Dictionary Keys)

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Dictionary Evolution                  │
└─────────────────────────────────────────────────────────────┘

Step 0: Source
├── frame: np.ndarray                    # Frame original completo

Step 1: ROIExtractionStep
├── frame: np.ndarray
├── roi_frame: np.ndarray                # Frame recortado del ROI

Step 2: YOLODetectionStep
├── frame: np.ndarray
├── roi_frame: np.ndarray
├── roi_detections: Detections           # Detecciones en coords ROI
└── yolo_metrics: dict                   # Tiempos de inferencia

Step 3: CoordinateTranslationStep
├── frame: np.ndarray
├── roi_frame: np.ndarray
├── roi_detections: Detections
├── detections: Detections               # Detecciones en coords frame
└── yolo_metrics: dict

Step 4: ByteTrackerStep
├── frame: np.ndarray
├── roi_frame: np.ndarray
├── roi_detections: Detections
├── detections: Detections               # Ahora con tracker_id
├── yolo_metrics: dict
└── tracker_processing_time_ms: float    # Tiempo de tracking

Step 5: LineZoneStep
├── frame: np.ndarray
├── roi_frame: np.ndarray
├── roi_detections: Detections
├── detections: Detections
├── yolo_metrics: dict
├── tracker_processing_time_ms: float
└── line_zone: LineZone                  # Objeto con in_count/out_count

Step 6: FPSCalculatorStep
├── frame: np.ndarray
├── roi_frame: np.ndarray
├── roi_detections: Detections
├── detections: Detections
├── yolo_metrics: dict
├── tracker_processing_time_ms: float
├── line_zone: LineZone
└── fps: float                           # FPS del pipeline

Steps 7-10: Annotation Pipeline (modify frame in-place with copy_frame=False)
└── frame: np.ndarray                    # Frame con todas las anotaciones
```

## Descripción de Cada Step

### **Core Pipeline (Siempre se ejecuta)**

#### 1. ROIExtractionStep
- **Propósito**: Extrae una región rectangular del frame original
- **Input**: `frame` (frame completo)
- **Output**: `roi_frame` (región recortada)
- **Parámetros**: x, y, width, height del ROI
- **Beneficio**: Reduce área de procesamiento → mejor rendimiento

#### 2. YOLODetectionStep
- **Propósito**: Detecta objetos SOLO dentro del ROI
- **Input**: `roi_frame`
- **Output**: `roi_detections` (coordenadas relativas al ROI)
- **Beneficio**: Menos procesamiento, más FPS

#### 3. CoordinateTranslationStep
- **Propósito**: Traduce coordenadas del ROI al frame completo
- **Input**: `roi_detections` (coords ROI)
- **Output**: `detections` (coords frame)
- **Operación**: `x_frame = x_roi + offset_x`, `y_frame = y_roi + offset_y`

#### 4. ByteTrackerStep
- **Propósito**: Tracking de objetos entre frames
- **Input**: `detections`
- **Output**: `detections` (con `tracker_id` agregado)
- **Algoritmo**: ByteTrack (maneja oclusiones)
- **Métricas**: Agrega `tracker_processing_time_ms`

#### 5. LineZoneStep
- **Propósito**: Cuenta objetos que cruzan una línea virtual
- **Input**: `detections` (con tracker_id)
- **Output**: `line_zone` (objeto con contadores)
- **Datos disponibles**:
  - `line_zone.in_count`: Objetos que cruzaron hacia adentro
  - `line_zone.out_count`: Objetos que cruzaron hacia afuera
  - `line_zone.last_crossed_in_ids`: IDs que cruzaron último frame

#### 6. FPSCalculatorStep
- **Propósito**: Calcula FPS del pipeline
- **Output**: `fps` (frames por segundo)

---

### **Annotation Pipeline (Condicional)**

Se ejecuta si `display.enabled = True` OR `output.enabled = True`

#### 7. ROIVisualizationStep
- **Propósito**: Dibuja rectángulo del ROI
- **Color**: Amarillo (255, 255, 0)
- **Thickness**: 2px
- **Note**: `copy_frame=False` para mejor rendimiento

#### 8. TrackerAnnotatorStep
- **Propósito**: Visualización completa de tracking
- **Componentes**:
  - **Boxes**: Bounding boxes coloreados por tracker ID
  - **Labels**: Muestra `ID: clase confianza%`
  - **Traces**: Trayectorias de 30 frames de historial
- **Colores**: Cada objeto tiene color único y consistente

#### 9. LineZoneAnnotatorStep
- **Propósito**: Dibuja la línea de conteo
- **Elementos**:
  - Línea virtual (configurable: color, grosor)
  - Labels "IN: X" y "OUT: Y" con contadores
- **Color default**: Blanco

#### 10. CallbackStep (Opcional)
- **Propósito**: Overlay de métricas en el frame
- **Condición**: Solo si `display.show_metrics = True`
- **Métricas mostradas**:
  - FPS del pipeline
  - Número de detecciones
  - Objetos trackeados activos
  - Contadores de línea (IN/OUT)
  - Tiempo de inferencia
  - Tiempo de tracking
- **Posición**: Configurable (top-left, top-right, etc.)

---

### **Sinks**

#### DisplaySink
- **Propósito**: Muestra frames en ventana OpenCV
- **Controles**: Presiona 'q' o 'ESC' para salir
- **Input key**: `frame` (con todas las anotaciones)

#### VideoFileSink
- **Propósito**: Guarda video a archivo
- **Formato**: MP4/AVI
- **Codec**: XVID (configurable)
- **FPS**: Heredado del source original

#### MultiSink
- **Cuándo**: Si ambos `display.enabled` y `output.enabled` son True
- **Ventaja**: Procesa ambos sinks en paralelo eficientemente

## Configuración Condicional

### Anotaciones
```python
if display_cfg.get("enabled", False) or output_cfg.get("enabled", False):
    # Se agregan todos los annotation steps
```

**Razón**: Evita procesamiento innecesario si no hay salida visual

### Metrics Overlay
```python
if display_cfg.get("show_metrics", False):
    # Solo si usuario quiere ver métricas detalladas
```

### Sinks
```python
sinks = []
if display_cfg.get("enabled"):
    sinks.append(DisplaySink(...))
if output_cfg.get("enabled"):
    sinks.append(VideoFileSink(...))

# Usa MultiSink solo si hay 2 o más sinks
if len(sinks) == 1:
    pipeline = pipeline | sinks[0]
elif len(sinks) > 1:
    pipeline = pipeline | sv.MultiSink(sinks)
```

## Optimizaciones de Rendimiento

1. **ROI Processing**: Solo procesa región de interés (~4x más rápido)
2. **copy_frame=False**: Evita copias innecesarias de frames
3. **Conditional Annotations**: Solo anota si hay salida visual
4. **MultiSink**: Procesa display y video en paralelo

## Ejemplo de Uso

```bash
# Display + métricas
python roi_line_zone.py --display --show-metrics

# Guardar a video sin display
python roi_line_zone.py --output result.avi

# Display + guardar + métricas
python roi_line_zone.py --display --show-metrics --output result.avi

# Guardar métricas a JSON
python roi_line_zone.py --save-metrics metrics.json
```

## Configuración (config.yaml)

```yaml
video:
  input: "video.mp4"  # o rtsp://... o 0 (webcam)

detector:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.4

tracker:
  track_activation_threshold: 0.25
  lost_track_buffer: 30

rois:
  - id: roi_1
    x: 320
    y: 20
    w: 640
    h: 640

counting_lines:
  - id: line_1
    start: [640, 0]
    end: [640, 720]

display:
  enabled: true
  show_metrics: true
```

## Métricas de Salida

Si se usa `--save-metrics`, genera archivo JSON con:

```json
[
  {
    "frame": 1,
    "fps": 68.5,
    "detections": 3,
    "tracked_objects": 2,
    "line_in_count": 5,
    "line_out_count": 3,
    "inference_time_ms": 12.5,
    "tracking_time_ms": 0.8,
    "crossed_objects": [
      {
        "tracker_id": 42,
        "direction": "in",
        "x": 450.5,
        "y": 230.8,
        "w": 85.2,
        "h": 120.4
      },
      {
        "tracker_id": 17,
        "direction": "out",
        "x": 380.0,
        "y": 190.5,
        "w": 78.6,
        "h": 115.2
      }
    ]
  }
]
```

### Campos de Métricas

- **frame**: Número de frame
- **fps**: Frames por segundo del pipeline
- **detections**: Número total de objetos detectados en el ROI
- **tracked_objects**: Número de objetos activos con tracker ID
- **line_in_count**: Contador acumulado de cruces hacia adentro
- **line_out_count**: Contador acumulado de cruces hacia afuera
- **inference_time_ms**: Tiempo de inferencia YOLO en milisegundos
- **tracking_time_ms**: Tiempo de procesamiento del tracker en milisegundos
- **crossed_objects**: Lista de objetos que cruzaron la línea en este frame
  - **tracker_id**: ID único del objeto
  - **direction**: "in" o "out" según dirección del cruce
  - **x, y**: Coordenadas de la esquina superior izquierda del bounding box
  - **w, h**: Ancho y alto del bounding box

Y genera gráfico PNG automáticamente con:
- FPS over time
- Detections per frame
- Inference time per frame

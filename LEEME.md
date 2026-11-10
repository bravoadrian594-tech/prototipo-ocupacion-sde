# Prototipo de pronóstico de ocupación - Santiago del Estero

Trabajo Final de Grado, Licenciatura en Inteligencia Artificial y Robótica (Universidad Siglo 21).
Autor: Milton Adrián Bravo.

El prototipo pronostica la tasa de ocupación de plazas de la ciudad de Santiago del Estero y de
Termas de Río Hondo para los doce meses siguientes al último mes publicado por el INDEC (Encuesta
de Ocupación Hotelera), compara tres modelos (suavizado exponencial simple, XGBoost sin eventos y
XGBoost con eventos) y muestra el resultado en una aplicación web para hospedadores y turistas.

- Aplicación publicada: https://pronostico-ocupacion-sde.streamlit.app
- Video de demostración: [link del video]

## Archivos

| Archivo | Qué hace |
|---|---|
| `base.py` | Rutas, zonas y creación de las siete tablas de la base SQLite |
| `01_preparar_datos.py` | Lee las planillas de la EOH, calcula la tasa y reemplaza 2020-2021 (PB-01) |
| `02_cargar_eventos.py` | Valida y carga el calendario de eventos (PB-02) |
| `03_entrenar_modelos.py` | Entrena, evalúa con 2024 y 2025 y genera el pronóstico (PB-03 a PB-05) |
| `app.py` | Aplicación web en Streamlit, solo lectura de la base (PB-06 a PB-09) |
| `datos/crudos/indec/` | Planillas originales del INDEC, 2015 a 2025 (no se modifican) |
| `datos/crudos/calendario_eventos.csv` | Calendario de eventos con fuente de cada uno (se edita con Excel) |
| `datos/prototipo.db` | Base de datos con la serie, los eventos, las corridas, los pronósticos y las métricas |
| `datos/procesados/` | Tablas de resultados (métricas e hiperparámetros) |
| `modelos/` | Modelos entrenados de la última corrida |
| `figuras/` | Gráficos de la serie y de la evaluación 2025 |

## Instalación

Requisitos: Python 3.13.

Descomprimir el .zip, abrir una terminal en la carpeta `prototipo-ocupacion-sde` y ejecutar:

```
python -m venv .venv
.venv\Scripts\activate          (en Linux o macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Si PowerShell no deja activar el entorno, ejecutar antes:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Ejecución

El repositorio ya incluye la base con los resultados, así que para ver la aplicación alcanza con:

```
streamlit run app.py
```

y abrir http://localhost:8501. Agregando `?pantalla=celular` a la dirección se ve la versión para celulares.

Para regenerar todo desde los datos originales:

```
python 01_preparar_datos.py
python 02_cargar_eventos.py
python 03_entrenar_modelos.py
```

## Actualización mensual

1. Descargar del INDEC la planilla más reciente de la EOH y copiarla en `datos/crudos/indec/`.
2. Agregar los eventos nuevos en `datos/crudos/calendario_eventos.csv`.
3. Ejecutar los pasos 1 a 3. Cada ejecución queda registrada como una corrida nueva en la base.
4. Subir los cambios: `git add .`, `git commit -m "Actualización"` y `git push`. La aplicación publicada se actualiza sola.

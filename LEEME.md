# Prototipo de pronóstico de ocupación - Santiago del Estero

Pronostica la tasa de ocupación de plazas de la ciudad de Santiago del Estero y de
Termas de Río Hondo para los doce meses siguientes al último mes publicado por el
INDEC, con tres modelos (suavizado exponencial simple, XGBoost sin eventos y XGBoost
con eventos), y la muestra en una aplicación web para hospedadores y turistas.

## Archivos

| Archivo | Qué hace |
|---|---|
| `base.py` | Rutas, zonas y creación de las siete tablas de la base SQLite |
| `01_preparar_datos.py` | Lee las planillas de la EOH, calcula la tasa y reemplaza 2020-2021 |
| `02_cargar_eventos.py` | Carga y controla el calendario de eventos |
| `03_entrenar_modelos.py` | Entrena, evalúa (2024 y 2025) y genera el pronóstico |
| `app.py` | Aplicación web en Streamlit (solo lectura de la base) |
| `datos/crudos/indec/` | Planillas originales del INDEC (no se modifican) |
| `datos/crudos/calendario_eventos.csv` | Calendario de eventos (se puede editar con Excel) |
| `datos/prototipo.db` | Base de datos que generan los scripts |
| `datos/procesados/` | Tablas de resultados (métricas, hiperparámetros, series) |
| `figuras/` | Gráficos para la tesis |

## Ciclo mensual

Con el entorno activado (`.venv\Scripts\activate`):

```
python 01_preparar_datos.py
python 02_cargar_eventos.py
python 03_entrenar_modelos.py
streamlit run app.py
```

Para actualizar: bajar del INDEC la planilla más nueva de la EOH y ponerla en
`datos/crudos/indec`, agregar los eventos nuevos al CSV y volver a correr los tres pasos.

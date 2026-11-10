"""
Lo que comparten todos los scripts: rutas de carpetas, zonas y la base de datos.

Aca se crean las siete tablas (zona, carga_datos, ocupacion_mensual, evento,
corrida, pronostico y metrica_modelo). Los otros scripts importan este archivo.
"""

import sqlite3
from pathlib import Path

CARPETA = Path(__file__).parent
CRUDOS = CARPETA / "datos" / "crudos"
INDEC = CRUDOS / "indec"
PROCESADOS = CARPETA / "datos" / "procesados"
FIGURAS = CARPETA / "figuras"
BASE = CARPETA / "datos" / "prototipo.db"

# Nombre de la hoja en las planillas del INDEC -> zona del prototipo
ZONAS = {
    "Santiago del Estero": {"id_zona": 1, "nombre": "Ciudad de Santiago del Estero"},
    "Termas": {"id_zona": 2, "nombre": "Termas de Río Hondo"},
}

MODELOS = ["base", "xgboost_sin_eventos", "xgboost_con_eventos"]

ESQUEMA = """
CREATE TABLE IF NOT EXISTS zona (
    id_zona INTEGER PRIMARY KEY,
    nombre  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS carga_datos (
    id_carga    INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo        TEXT NOT NULL CHECK (tipo IN ('serie_eoh', 'calendario')),
    fuente      TEXT NOT NULL,
    fecha_corte TEXT NOT NULL,
    fecha_carga TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocupacion_mensual (
    id_zona               INTEGER NOT NULL REFERENCES zona(id_zona),
    anio                  INTEGER NOT NULL CHECK (anio >= 2015),
    mes                   INTEGER NOT NULL CHECK (mes BETWEEN 1 AND 12),
    plazas_disponibles    INTEGER CHECK (plazas_disponibles >= 0),
    plazas_ocupadas       INTEGER CHECK (plazas_ocupadas >= 0),
    tasa_ocupacion_plazas REAL NOT NULL CHECK (tasa_ocupacion_plazas BETWEEN 0 AND 100),
    origen                TEXT NOT NULL CHECK (origen IN ('observado', 'imputado')),
    tasa_original         REAL,
    regla_imputacion      TEXT,
    id_carga              INTEGER NOT NULL REFERENCES carga_datos(id_carga),
    PRIMARY KEY (id_zona, anio, mes)
);

CREATE TABLE IF NOT EXISTS evento (
    id_evento         INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre            TEXT NOT NULL,
    tipo              TEXT NOT NULL CHECK (tipo IN ('deportivo', 'cultural')),
    id_zona           INTEGER NOT NULL REFERENCES zona(id_zona),
    fecha_inicio      TEXT NOT NULL,
    fecha_fin         TEXT NOT NULL,
    nivel_alcance     INTEGER NOT NULL CHECK (nivel_alcance BETWEEN 1 AND 4),
    fuente            TEXT NOT NULL,
    fecha_publicacion TEXT NOT NULL,
    estado            TEXT NOT NULL CHECK (estado IN ('confirmado', 'supuesto')),
    id_carga          INTEGER NOT NULL REFERENCES carga_datos(id_carga),
    UNIQUE (nombre, id_zona, fecha_inicio)
);

CREATE TABLE IF NOT EXISTS corrida (
    id_corrida          INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_corrida       TEXT NOT NULL,
    corte_entrenamiento TEXT NOT NULL,
    evaluacion_desde    TEXT NOT NULL,
    evaluacion_hasta    TEXT NOT NULL,
    version             TEXT NOT NULL,
    id_carga_serie      INTEGER NOT NULL REFERENCES carga_datos(id_carga),
    id_carga_calendario INTEGER NOT NULL REFERENCES carga_datos(id_carga)
);

CREATE TABLE IF NOT EXISTS pronostico (
    id_corrida             INTEGER NOT NULL REFERENCES corrida(id_corrida),
    id_zona                INTEGER NOT NULL REFERENCES zona(id_zona),
    modelo                 TEXT NOT NULL,
    anio                   INTEGER NOT NULL,
    mes                    INTEGER NOT NULL CHECK (mes BETWEEN 1 AND 12),
    ocupacion_pronosticada REAL NOT NULL CHECK (ocupacion_pronosticada BETWEEN 0 AND 100),
    PRIMARY KEY (id_corrida, id_zona, modelo, anio, mes)
);

CREATE TABLE IF NOT EXISTS metrica_modelo (
    id_corrida INTEGER NOT NULL REFERENCES corrida(id_corrida),
    id_zona    INTEGER NOT NULL REFERENCES zona(id_zona),
    modelo     TEXT NOT NULL,
    mae        REAL NOT NULL CHECK (mae >= 0),
    rmse       REAL NOT NULL CHECK (rmse >= 0),
    es_ganador INTEGER NOT NULL CHECK (es_ganador IN (0, 1)),
    PRIMARY KEY (id_corrida, id_zona, modelo)
);
"""


def conectar():
    """Abre la base (la crea si no existe) con las siete tablas."""
    BASE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(BASE)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ESQUEMA)
    for datos in ZONAS.values():
        con.execute("INSERT OR IGNORE INTO zona (id_zona, nombre) VALUES (?, ?)",
                    (datos["id_zona"], datos["nombre"]))
    con.commit()
    return con

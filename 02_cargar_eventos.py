"""
Paso 2: carga el calendario de eventos.

El calendario esta en datos/crudos/calendario_eventos.csv y se puede editar con
Excel. Antes de cargarlo reviso que cada evento tenga bien el tipo, la zona, el
nivel (1 a 4), las fechas y el estado. Si alguno esta mal no se carga y avisa en
que fila esta. Despues registra la carga, reemplaza la tabla evento y guarda en
datos/procesados/eventos_por_mes.csv cuantos eventos de cada nivel hay por zona
y mes. Si un evento ocupa dos meses, cuenta en los dos.

Uso: python 02_cargar_eventos.py
"""

from datetime import date

import pandas as pd

from base import CRUDOS, PROCESADOS, ZONAS, conectar

CALENDARIO = CRUDOS / "calendario_eventos.csv"
COLUMNAS = ["nombre", "tipo", "id_zona", "fecha_inicio", "fecha_fin", "nivel_alcance",
            "fuente", "fecha_publicacion", "estado"]


def leer_calendario():
    cal = pd.read_csv(CALENDARIO, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    cal = cal[cal["nombre"].str.strip() != ""].copy()
    errores = []
    validos = []
    zonas_validas = {str(z["id_zona"]) for z in ZONAS.values()}
    for i, ev in cal.iterrows():
        fila = i + 2  # numero de fila en Excel (la 1 es el encabezado)
        problema = None
        if ev["tipo"].strip().lower() not in ("deportivo", "cultural"):
            problema = "el tipo debe ser 'deportivo' o 'cultural'"
        elif ev["id_zona"].strip() not in zonas_validas:
            problema = "la zona debe ser 1 (ciudad) o 2 (Termas)"
        elif ev["nivel_alcance"].strip() not in ("1", "2", "3", "4"):
            problema = "falta el nivel de alcance o no está entre 1 y 4"
        elif ev["estado"].strip().lower() not in ("confirmado", "supuesto"):
            problema = "el estado debe ser 'confirmado' o 'supuesto'"
        elif ev["fuente"].strip() == "":
            problema = "falta la fuente"
        else:
            try:
                ini = date.fromisoformat(ev["fecha_inicio"].strip())
                fin = date.fromisoformat(ev["fecha_fin"].strip())
                pub = date.fromisoformat(ev["fecha_publicacion"].strip())
                if fin < ini:
                    problema = "la fecha de fin es anterior a la de inicio"
                elif pub > ini:
                    problema = "la fecha de publicación es posterior al inicio"
            except ValueError:
                problema = "alguna fecha no tiene el formato AAAA-MM-DD"
        if problema:
            errores.append(f"  Fila {fila} ({ev['nombre']}): {problema}")
        else:
            validos.append(ev)
    return pd.DataFrame(validos)[COLUMNAS], errores


def guardar(cal):
    con = conectar()
    corte = max(cal["fecha_publicacion"])[:7] + "-01"
    cur = con.execute(
        "INSERT INTO carga_datos (tipo, fuente, fecha_corte, fecha_carga) VALUES ('calendario', ?, ?, ?)",
        ("Calendario de eventos del autor (datos/crudos/calendario_eventos.csv); fuente de cada evento en la tabla Evento",
         corte, date.today().isoformat()))
    id_carga = cur.lastrowid
    con.execute("DELETE FROM evento")
    con.executemany("""
        INSERT INTO evento (nombre, tipo, id_zona, fecha_inicio, fecha_fin, nivel_alcance,
                            fuente, fecha_publicacion, estado, id_carga)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(r.nombre.strip(), r.tipo.strip().lower(), int(r.id_zona), r.fecha_inicio.strip(),
          r.fecha_fin.strip(), int(r.nivel_alcance), r.fuente.strip(),
          r.fecha_publicacion.strip(), r.estado.strip().lower(), id_carga)
         for r in cal.itertuples()])
    con.commit()
    con.close()
    return id_carga


def eventos_por_mes(cal):
    """Cantidad de eventos de cada nivel por zona y mes."""
    filas = []
    for r in cal.itertuples():
        meses = pd.period_range(r.fecha_inicio[:7], r.fecha_fin[:7], freq="M")
        for m in meses:
            filas.append({"id_zona": int(r.id_zona), "anio": m.year, "mes": m.month,
                          "nivel": int(r.nivel_alcance)})
    df = pd.DataFrame(filas)
    tabla = (df.pivot_table(index=["id_zona", "anio", "mes"], columns="nivel",
                            aggfunc="size", fill_value=0)
               .reindex(columns=[1, 2, 3, 4], fill_value=0))
    tabla.columns = [f"eventos_nivel_{n}" for n in tabla.columns]
    return tabla.reset_index()


if __name__ == "__main__":
    cal, errores = leer_calendario()
    if errores:
        print("Eventos rechazados (corregilos en el CSV y volvé a correr):")
        print("\n".join(errores))
    id_carga = guardar(cal)
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    eventos_por_mes(cal).to_csv(PROCESADOS / "eventos_por_mes.csv", index=False, encoding="utf-8-sig")

    print(f"\nCarga N.º {id_carga}: {len(cal)} eventos guardados en la tabla evento.")
    resumen = cal.groupby(["id_zona", "nivel_alcance"]).size().unstack(fill_value=0)
    resumen.index = [z["nombre"] for z in ZONAS.values() if str(z["id_zona"]) in resumen.index]
    resumen.columns = [f"Nivel {c}" for c in resumen.columns]
    print(resumen.to_string())

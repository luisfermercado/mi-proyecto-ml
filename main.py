"""
API FastAPI que expone el modelo @champion servido en Databricks Model Serving.

La API no carga el modelo: solo valida los datos de entrada y llama al endpoint
de Databricks. Si el modelo cambia en Databricks, esta API no se modifica.

Credenciales: nunca se escriben en el código. Azure Container Apps las inyecta
como variables de entorno (DATABRICKS_URL y DATABRICKS_TOKEN).
"""
import os
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VERSION = "1.0.0"
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
TIMEOUT_SEGUNDOS = 90  # el endpoint puede tardar si estaba en scale-to-zero

# Las 7 variables del modelo mlops-posgrado-estudiantes (NB01), en el mismo orden y tipo.
FEATURES = [
    "horas_estudio", "horas_sueno", "asistencia", "promedio_previo",
    "indice_dedicacion", "en_riesgo", "destacado",
]

app = FastAPI(
    title="API MLOps — Predicción de aprobación de estudiantes",
    description="Consume el modelo @champion desplegado en Databricks Model Serving.",
    version=VERSION,
)


class Estudiante(BaseModel):
    horas_estudio: float = Field(..., ge=0, le=24, examples=[7.5])
    horas_sueno: float = Field(..., ge=0, le=24, examples=[7.0])
    asistencia: float = Field(..., ge=0, le=100, examples=[90.0])
    promedio_previo: float = Field(..., ge=0, le=5, examples=[4.2])
    indice_dedicacion: float = Field(..., ge=0, examples=[0.81])
    en_riesgo: int = Field(..., ge=0, le=1, examples=[0])
    destacado: int = Field(..., ge=0, le=1, examples=[1])


class Lote(BaseModel):
    estudiantes: List[Estudiante] = Field(..., min_length=1, max_length=500)


def _llamar_databricks(registros: list) -> list:
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        raise HTTPException(status_code=503, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN en el entorno.")
    try:
        resp = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
            json={"dataframe_records": registros},
            timeout=TIMEOUT_SEGUNDOS,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar el endpoint de Databricks: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Databricks respondió {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("predictions", [])


def _etiqueta(pred) -> str:
    return "Aprueba" if int(pred) == 1 else "Reprueba"


# 1) Bienvenida
@app.get("/")
def inicio():
    return {"mensaje": "API MLOps EAN activa", "documentacion": "/docs", "version": VERSION}


# 2) Estado del servicio (no llama a Databricks, responde rápido)
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN),
        "endpoint_databricks": DATABRICKS_URL.split("/serving-endpoints/")[-1].split("/")[0] if DATABRICKS_URL else None,
    }


# 3) Variables que espera el modelo
@app.get("/features")
def features():
    return {"features": FEATURES, "n_features": len(FEATURES)}


# 4) Predicción para un estudiante
@app.post("/predecir")
def predecir(estudiante: Estudiante):
    pred = _llamar_databricks([estudiante.model_dump()])[0]
    return {"prediccion": int(pred), "resultado": _etiqueta(pred)}


# 5) Predicción por lote
@app.post("/predecir/lote")
def predecir_lote(lote: Lote):
    preds = _llamar_databricks([e.model_dump() for e in lote.estudiantes])
    return {
        "n": len(preds),
        "predicciones": [{"prediccion": int(p), "resultado": _etiqueta(p)} for p in preds],
    }

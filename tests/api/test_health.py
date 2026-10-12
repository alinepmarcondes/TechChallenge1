"""Testes do endpoint de verificação de saúde da API."""

import importlib

from fastapi.testclient import TestClient

app_module = importlib.import_module("src.api.app")

def test_health_modelo_carregado(monkeypatch) -> None:
    """Deve retornar 200 e informar que o modelo esta disponivel."""
    modelo_fake = object()
    monkeypatch.setattr(
        app_module,
        "carregar_modelo",
        lambda _caminho: modelo_fake
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "LogisticRegression",
        "model_version": "1.0.0",
    }

def test_health_modelo_nao_carregado(monkeypatch) -> None:
    """Deve indicar estado degradado quando o artefato nao esta disponivel."""

    def falhar_ao_carregar(_caminho):
        raise FileNotFoundError("Modelo não encontrado")

    monkeypatch.setattr(
        app_module,
        "carregar_modelo",
        falhar_ao_carregar,
    )

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "model_loaded": False,
        "model_name": "LogisticRegression",
        "model_version": "1.0.0",
    }
"""Testes do endpoint de predição de churn."""

import importlib
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api.schemas import PredicaoOutput

app_module = importlib.import_module("src.api.app")

def payload_valido() -> dict[str,object]:
    """Retorna uma entrada válida para o schema ClienteInput."""
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",  
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.50,
        "total_charges": 1074.00,
        "cltv": 3200,
    }


def preparar_api_com_modelo(monkeypatch) -> tuple[object, Mock]:
    """Substitui carregamento e inferência por objetos controlados no teste."""
    modelo_fake = object()
    predicao_fake = Mock(
        return_value=PredicaoOutput(
        previsao_churn=1,
        probabilidade_churn=0.7342,
        )
    )

    monkeypatch.setattr(
        app_module,
        "carregar_modelo",
        lambda _caminho: modelo_fake,
    )
    monkeypatch.setattr(
        app_module,
        "realizar_predicao",
        predicao_fake,
    )

    return modelo_fake, predicao_fake

def test_predict_retorna_classe_e_probabilidade(monkeypatch) -> None:
    """Deve retornar 200 com uma predição válida a partir de uma entrada válida."""
    modelo_fake, predicao_fake = preparar_api_com_modelo(monkeypatch)

    with TestClient(app_module.app) as client:
        response =  client.post("/predict", json=payload_valido())

    assert response.status_code == 200
    assert response.json() == {
        "previsao_churn": 1,
        "probabilidade_churn": 0.7342,
    }

    predicao_fake.assert_called_once()
    modelo_recebido, cliente_recebido = predicao_fake.call_args.args
    assert modelo_recebido is modelo_fake
    assert cliente_recebido.tenure_months == 12
    assert cliente_recebido.device_protection == "No"

def test_predict_retorna_422_campo_obrigatorio(monkeypatch) -> None:
    """Deve rejeitar a requisicao antes de chamar o modelo."""
    _, predicao_fake = preparar_api_com_modelo(monkeypatch)
    payload = payload_valido()
    payload.pop("tenure_months")

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    predicao_fake.assert_not_called()

def test_predict_retorna_422_categoria_invalida(monkeypatch) -> None:
    """Deve rejeitar valores fora das categorias declaradas no schema."""
    _, predicao_fake = preparar_api_com_modelo(monkeypatch)
    payload = payload_valido()
    payload["contract"] = "Invalid contract"

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    predicao_fake.assert_not_called()

def test_predict_retorna_503_modelo_indisponivel(monkeypatch) -> None:
    """Deve impedir a inferência quando o artefato não foi carregado."""

    def falhar_ao_carregar(_caminho):
        raise FileNotFoundError("Modelo não encontrado")

    monkeypatch.setattr(
    app_module,
    "carregar_modelo",
    falhar_ao_carregar,
    )

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload_valido())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Modelo indisponível para inferência"
    }
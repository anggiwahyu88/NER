
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_valid_article():
    response = client.post(
        "/extract-entities",
        json={
            "title": "Microsoft announces new product",
            "body": "The company announced the product in Jakarta."
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "entity_count" in data
    assert "model" in data


def test_empty_article():
    response = client.post(
        "/extract-entities",
        json={"title": " ", "body": " "}
    )
    assert response.status_code == 400


def test_missing_body():
    response = client.post(
        "/extract-entities",
        json={"title": "Example article"}
    )
    assert response.status_code == 422


def test_text_too_long():
    response = client.post(
        "/extract-entities",
        json={
            "title": "Example",
            "body": "a" * 10001
        }
    )
    assert response.status_code == 413

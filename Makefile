.PHONY: install dev test lint format run ui docker-up docker-down

install:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

run:
	python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

ui:
	streamlit run frontend/streamlit_app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

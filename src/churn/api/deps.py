"""Dependências compartilhadas pelas rotas."""

from __future__ import annotations

from fastapi import Request

from churn.config import Settings
from churn.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_settings(request: Request) -> Settings:
    return request.app.state.settings

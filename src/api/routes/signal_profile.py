"""Read and replace the global pluggable signal profile."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cryptotrader.profiles.models import ComponentWeight, SignalProfile, validate_signal_profile

router = APIRouter(prefix="/api/signal-profile", tags=["signal-profile"])


class ComponentWeightPayload(BaseModel):
    component_id: str
    enabled: bool
    weight: float


class SignalProfileUpdate(BaseModel):
    components: list[ComponentWeightPayload]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    hitl_required: bool


class InstalledComponentPayload(BaseModel):
    component_id: str
    display_name: str
    description: str


class SignalProfileResponse(SignalProfileUpdate):
    revision: int
    installed_components: list[InstalledComponentPayload]


def _dependencies(request: Request):
    repository = getattr(request.app.state, "signal_profile_repository", None)
    registry = getattr(request.app.state, "signal_registry", None)
    if repository is None or registry is None:
        raise HTTPException(status_code=503, detail="signal profile persistence is not configured")
    return repository, registry


def _response(profile: SignalProfile, registry) -> SignalProfileResponse:
    return SignalProfileResponse(
        revision=profile.revision,
        components=[ComponentWeightPayload(**item.__dict__) for item in profile.components],
        neutral_threshold=profile.neutral_threshold,
        max_target_ratio=profile.max_target_ratio,
        atr_stop_multiplier=profile.atr_stop_multiplier,
        reward_ratio=profile.reward_ratio,
        hitl_required=profile.hitl_required,
        installed_components=[InstalledComponentPayload(**item.__dict__) for item in registry.metadata()],
    )


@router.get("", response_model=SignalProfileResponse)
async def get_signal_profile(request: Request) -> SignalProfileResponse:
    repository, registry = _dependencies(request)
    profile = await repository.get()
    if profile is None:
        raise HTTPException(status_code=503, detail="global signal profile has not been initialized")
    return _response(profile, registry)


@router.put("", response_model=SignalProfileResponse)
async def replace_signal_profile(payload: SignalProfileUpdate, request: Request) -> SignalProfileResponse:
    repository, registry = _dependencies(request)
    candidate = SignalProfile(
        revision=0,
        components=tuple(ComponentWeight(**item.model_dump()) for item in payload.components),
        neutral_threshold=payload.neutral_threshold,
        max_target_ratio=payload.max_target_ratio,
        atr_stop_multiplier=payload.atr_stop_multiplier,
        reward_ratio=payload.reward_ratio,
        hitl_required=payload.hitl_required,
    )
    try:
        validate_signal_profile(candidate, registry.ids())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _response(await repository.replace(candidate), registry)

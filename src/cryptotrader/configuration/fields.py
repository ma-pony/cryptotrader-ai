"""Typed, locale-aware descriptions for configuration form fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel


@dataclass(frozen=True)
class LocalizedText:
    zh_CN: str  # noqa: N815 - transport contract preserves locale code spelling.
    en_US: str  # noqa: N815 - transport contract preserves locale code spelling.


@dataclass(frozen=True)
class FieldOption:
    value: str
    label: LocalizedText


@dataclass(frozen=True)
class ConfigurationField:
    key: str
    label: LocalizedText
    description: LocalizedText
    kind: Literal["text", "number", "integer", "boolean", "select", "string_list"]
    default_value: Any
    required: bool
    minimum: float | int | None = None
    maximum: float | int | None = None
    exclusive_minimum: float | int | None = None
    exclusive_maximum: float | int | None = None
    step: float | int | None = None
    unit: str | None = None
    advanced: bool = False
    options: tuple[FieldOption, ...] = ()


def _localized(value: Any, fallback: str) -> LocalizedText:
    if isinstance(value, LocalizedText):
        return value
    if isinstance(value, dict) and {"zh_CN", "en_US"} <= set(value):
        return LocalizedText(zh_CN=str(value["zh_CN"]), en_US=str(value["en_US"]))
    text = str(value or fallback)
    return LocalizedText(zh_CN=text, en_US=text)


def _constraints(field) -> dict[str, float | int | None]:
    constraints = dict.fromkeys(("minimum", "maximum", "exclusive_minimum", "exclusive_maximum", "step"))
    for metadata in field.metadata:
        for attribute, key in (
            ("ge", "minimum"),
            ("le", "maximum"),
            ("gt", "exclusive_minimum"),
            ("lt", "exclusive_maximum"),
            ("multiple_of", "step"),
        ):
            value = getattr(metadata, attribute, None)
            if value is not None:
                constraints[key] = value
    return constraints


def _field_kind(annotation: Any) -> tuple[str, tuple[FieldOption, ...]]:
    origin = get_origin(annotation)
    if origin is Literal:
        if not all(type(value) is str for value in get_args(annotation)):
            raise TypeError("Unsupported configuration field: select values must be strings")
        return "select", tuple(FieldOption(str(value), _localized(None, str(value))) for value in get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not all(type(member.value) is str for member in annotation):
            raise TypeError("Unsupported configuration field: enum values must be strings")
        return "select", tuple(
            FieldOption(str(member.value), _localized(None, member.name.replace("_", " ").title()))
            for member in annotation
        )
    if annotation is bool:
        return "boolean", ()
    if annotation is int:
        return "integer", ()
    if annotation is float:
        return "number", ()
    if (origin is list and get_args(annotation) == (str,)) or (
        origin is tuple and get_args(annotation) == (str, Ellipsis)
    ):
        return "string_list", ()
    if annotation is str:
        return "text", ()
    raise TypeError(f"Unsupported configuration field annotation: {annotation!r}")


def _static_default(field) -> Any:
    if field.is_required() or field.default_factory is not None:
        return None
    return field.get_default()


def configuration_fields(parameter_model: type[BaseModel], *, prefix: str = "") -> tuple[ConfigurationField, ...]:
    """Flatten a parameter model into the stable dot-path fields consumed by forms."""
    descriptors: list[ConfigurationField] = []
    for name, field in parameter_model.model_fields.items():
        key = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            descriptors.extend(configuration_fields(annotation, prefix=key))
            continue
        extra = field.json_schema_extra or {}
        kind, inferred_options = _field_kind(annotation)
        options = extra.get("options", inferred_options)
        if any(
            type(option.value if isinstance(option, FieldOption) else option["value"]) is not str for option in options
        ):
            raise TypeError("Unsupported configuration field: select values must be strings")
        normalized_options = tuple(
            option
            if isinstance(option, FieldOption)
            else FieldOption(str(option["value"]), _localized(option["label"], ""))
            for option in options
        )
        label = getattr(parameter_model, "field_labels", {}).get(
            name, _localized(extra.get("label"), field.title or name.replace("_", " ").title())
        )
        description = getattr(parameter_model, "field_descriptions", {}).get(
            name,
            _localized(extra.get("description"), field.description or ""),
        )
        descriptors.append(
            ConfigurationField(
                key=key,
                label=label,
                description=description,
                kind=kind,
                default_value=_static_default(field),
                required=field.is_required(),
                **_constraints(field),
                unit=extra.get("unit"),
                advanced=bool(extra.get("advanced", False)),
                options=normalized_options,
            )
        )
    return tuple(descriptors)

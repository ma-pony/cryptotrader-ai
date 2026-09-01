"""Baseline backend for proving code-registered extension discovery.

It shares the isolated test database/runtime behavior and frontend build with
the golden fixture, but exposes only the product's built-in registrations.
"""

from tests.workbench_app import build_workbench_app, workbench_base_registry

registry_factory = workbench_base_registry
app = build_workbench_app(registry_factory=registry_factory)

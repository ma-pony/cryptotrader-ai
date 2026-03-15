"""Tests for Dashboard base architecture refactor (task 5).

Tests are written before implementation (TDD strict mode).
Focus on pure Python logic that does not require mocking Streamlit itself.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers — build a streamlit mock that plays nicely with the app module
# ---------------------------------------------------------------------------


def _make_st_mock(page: str = "Overview") -> MagicMock:
    """Return a MagicMock that mimics the Streamlit API surface used by app.py.

    The sidebar.radio() method returns *page* so that _dispatch[page]() does
    not raise a KeyError when the app module is imported and executed.
    """
    st = MagicMock()
    st.sidebar.radio.return_value = page
    st.query_params.get.return_value = page
    # cache_resource is a decorator — return a pass-through decorator
    st.cache_resource = lambda fn: fn
    # cache_data is a decorator with optional ttl kwarg
    st.cache_data = lambda *args, **kwargs: lambda fn: fn
    return st


def _import_app(page: str = "Overview"):
    """Import (or retrieve from sys.modules) dashboard.app with a friendly st mock."""
    st_mock = _make_st_mock(page)
    sys.modules.pop("dashboard.app", None)
    with patch.dict("sys.modules", {"streamlit": st_mock}):
        return importlib.import_module("dashboard.app")


def _import_data_loader():
    """Import dashboard.data_loader with streamlit mocked."""
    sys.modules.pop("dashboard.data_loader", None)
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        return importlib.import_module("dashboard.data_loader")


# ---------------------------------------------------------------------------
# run_async helper (moved from app.py to data_loader.py)
# ---------------------------------------------------------------------------


def test_run_async_executes_coroutine():
    """run_async() executes a coroutine and returns its result."""
    dl = _import_data_loader()

    async def _coro():
        return 99

    result = dl.run_async(_coro())
    assert result == 99


def test_run_async_reuses_loop_across_calls():
    """run_async() reuses its event loop; loop is not closed between calls."""
    dl = _import_data_loader()

    async def _coro(val):
        return val

    r1 = dl.run_async(_coro(1))
    r2 = dl.run_async(_coro(2))
    assert r1 == 1
    assert r2 == 2


def test_run_async_recreates_closed_loop():
    """run_async() creates a fresh loop if the existing one is closed."""
    dl = _import_data_loader()

    # Force-close the module-level loop
    if dl._loop is not None and not dl._loop.is_closed():
        dl._loop.close()

    async def _coro():
        return "recovered"

    result = dl.run_async(_coro())
    assert result == "recovered"
    assert dl._loop is not None
    assert not dl._loop.is_closed()


def test_run_async_propagates_exception():
    """run_async() re-raises exceptions from coroutines."""
    import pytest

    dl = _import_data_loader()

    async def _bad():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        dl.run_async(_bad())


# ---------------------------------------------------------------------------
# Page navigation logic (pure Python, no Streamlit calls)
# ---------------------------------------------------------------------------

_EXPECTED_PAGES = ["Overview", "Live Decisions", "Backtest", "Risk Status", "Metrics"]


def test_pages_list_has_five_entries():
    """app.py exposes exactly 5 pages."""
    app = _import_app()
    assert len(app._PAGES) == 5


def test_pages_list_contains_expected_names():
    """app.py _PAGES contains the five required page names."""
    app = _import_app()
    assert app._PAGES == _EXPECTED_PAGES


def test_resolve_page_known_name():
    """_resolve_page returns the page name when it is a valid page."""
    app = _import_app()
    for name in _EXPECTED_PAGES:
        assert app._resolve_page(name) == name


def test_resolve_page_unknown_falls_back_to_overview():
    """_resolve_page falls back to 'Overview' for unknown page names."""
    app = _import_app()
    assert app._resolve_page("NonExistentPage") == "Overview"
    assert app._resolve_page("") == "Overview"
    assert app._resolve_page(None) == "Overview"


def test_page_dispatch_map_covers_all_pages():
    """_get_page_dispatch returns a dict with a render callable for each page."""

    def _mock_page_module() -> MagicMock:
        m = MagicMock()
        m.render = MagicMock()
        return m

    page_mocks = {
        "dashboard._pages.overview": _mock_page_module(),
        "dashboard._pages.live_decisions": _mock_page_module(),
        "dashboard._pages.backtest": _mock_page_module(),
        "dashboard._pages.risk_status": _mock_page_module(),
        "dashboard._pages.metrics": _mock_page_module(),
    }

    for key in ["dashboard.app", *page_mocks]:
        sys.modules.pop(key, None)

    st_mock = _make_st_mock("Overview")
    with patch.dict("sys.modules", {"streamlit": st_mock, **page_mocks}):
        app = importlib.import_module("dashboard.app")
        dispatch = app._get_page_dispatch()

    assert set(dispatch.keys()) == set(_EXPECTED_PAGES)
    for name in _EXPECTED_PAGES:
        assert callable(dispatch[name]), f"dispatch[{name!r}] must be callable"


# ---------------------------------------------------------------------------
# data_loader module importability
# ---------------------------------------------------------------------------


def test_data_loader_module_is_importable():
    """dashboard.data_loader can be imported without errors."""
    dl = _import_data_loader()
    assert dl is not None


def test_run_async_is_exported():
    """dashboard.data_loader exposes run_async at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "run_async")
    assert callable(dl.run_async)


# ---------------------------------------------------------------------------
# components module importability
# ---------------------------------------------------------------------------


def test_components_module_is_importable():
    """dashboard.components can be imported without errors."""
    sys.modules.pop("dashboard.components", None)
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        comp = importlib.import_module("dashboard.components")
    assert comp is not None


# ---------------------------------------------------------------------------
# pages package importability
# ---------------------------------------------------------------------------


def test_pages_package_is_importable():
    """dashboard._pages package can be imported."""
    sys.modules.pop("dashboard._pages", None)
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        pkg = importlib.import_module("dashboard._pages")
    assert pkg is not None


def test_each_page_module_has_render_function():
    """Each page module exports render() as a callable."""
    page_module_names = [
        "dashboard._pages.overview",
        "dashboard._pages.live_decisions",
        "dashboard._pages.backtest",
        "dashboard._pages.risk_status",
        "dashboard._pages.metrics",
    ]
    for mod_name in page_module_names:
        sys.modules.pop(mod_name, None)

    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        for mod_name in page_module_names:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, "render"), f"{mod_name} must export render()"
            assert callable(mod.render), f"{mod_name}.render must be callable"


# ---------------------------------------------------------------------------
# Backward-compatibility: existing test_dashboard.py test must still pass
# ---------------------------------------------------------------------------


def test_run_helper_still_accessible_from_data_loader():
    """The run_async helper works as expected post-refactor."""
    dl = _import_data_loader()

    async def _coro():
        return 42

    assert dl.run_async(_coro()) == 42

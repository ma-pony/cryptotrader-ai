"""Streamlit dashboard entry point for CryptoTrader AI.

Responsibilities (this file only):
- Page-config call (must come first in any Streamlit app)
- Config loading via @st.cache_resource
- Sidebar navigation rendered from _PAGES list
- URL query-param synchronisation (?page=) for browser forward/back
- Dispatch to the correct page module's render() function

All inline page code has been removed.  Each page lives in dashboard/pages/.
All data loading lives in dashboard/data_loader.py.
Shared rendering components live in dashboard/components.py.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Constants (module-level, importable without triggering Streamlit I/O)
# ---------------------------------------------------------------------------

_PAGES = ["Overview", "Live Decisions", "Backtest", "Risk Status", "Metrics"]


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit I/O — easy to unit-test)
# ---------------------------------------------------------------------------


def _resolve_page(name: str | None) -> str:
    """Return *name* if it is a valid page, otherwise return the default page.

    Args:
        name: Page name candidate (typically from st.query_params).

    Returns:
        A valid page name from _PAGES, defaulting to "Overview".
    """
    if name in _PAGES:
        return name
    return _PAGES[0]


def _get_page_dispatch() -> dict[str, object]:
    """Build and return a mapping from page name to its render callable.

    Imports are deferred to this function so that page modules (which import
    streamlit) are only loaded after st.set_page_config() has been called.

    Returns:
        Dict mapping each page name to a callable render() function.
    """
    from dashboard._pages import backtest as _backtest
    from dashboard._pages import live_decisions as _live_decisions
    from dashboard._pages import metrics as _metrics
    from dashboard._pages import overview as _overview
    from dashboard._pages import risk_status as _risk_status

    return {
        "Overview": _overview.render,
        "Live Decisions": _live_decisions.render,
        "Backtest": _backtest.render,
        "Risk Status": _risk_status.render,
        "Metrics": _metrics.render,
    }


# ---------------------------------------------------------------------------
# Config loader (cached for the lifetime of the Streamlit process)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_config():
    """Load and cache the application configuration.

    Returns:
        The application Config object.

    Raises:
        Exception: Propagated to the caller; the main block converts this to
            st.error() + st.stop() so the page does not render with bad config.
    """
    from cryptotrader.config import load_config

    return load_config()


# ---------------------------------------------------------------------------
# Main rendering entrypoint — called by Streamlit at top level
# ---------------------------------------------------------------------------


def _main() -> None:
    """Execute the full dashboard rendering pipeline.

    Separated into a function so that module-level constants and pure helpers
    remain importable without triggering Streamlit I/O (useful for testing).
    """
    st.set_page_config(page_title="CryptoTrader AI", page_icon="📊", layout="wide")

    # Load config — halt rendering on failure so no page runs with missing config.
    try:
        _get_config()
    except Exception as _cfg_exc:
        st.error(f"Configuration load failed: {_cfg_exc}")
        st.stop()
        return  # unreachable in Streamlit, but satisfies static analysis

    # --- Sidebar navigation + URL param synchronisation ---
    _qp = st.query_params.get("page", "Overview")
    _current_page = _resolve_page(_qp)
    _default_idx = _PAGES.index(_current_page)

    page = st.sidebar.radio("Navigation", _PAGES, index=_default_idx, key="nav")
    if page != st.query_params.get("page"):
        st.query_params["page"] = page

    # --- Dispatch to the selected page ---
    _dispatch = _get_page_dispatch()
    _dispatch[page]()


_main()

"""Dashboard pages sub-package.

Each module in this package exports a single render() function that
Streamlit calls when the user navigates to that page.

Pages
-----
overview        -- Portfolio overview and scheduler status (task 9.1)
live_decisions  -- Live decision history and full pipeline detail (task 7)
backtest        -- Backtest runner and session comparison (task 8)
risk_status     -- Risk gate status and circuit-breaker management (task 9.2)
metrics         -- Prometheus metrics snapshot and latency trends (task 10)
"""

from __future__ import annotations

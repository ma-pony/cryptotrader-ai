"""MetricsCollector 与 Prometheus 指标端点测试。"""


# -- MetricsCollector 单元测试 --


def test_get_metrics_collector_returns_singleton():
    """get_metrics_collector() 每次调用返回同一个实例。"""
    from cryptotrader.metrics import get_metrics_collector

    a = get_metrics_collector()
    b = get_metrics_collector()
    assert a is b


def test_inc_llm_calls():
    """ct_llm_calls_total 计数器可按 model/node 标签递增。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o", "node": "agents"})
    mc.inc_llm_calls(model="gpt-4o", node="agents")
    after = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o", "node": "agents"})
    assert after == (before or 0) + 1


def test_inc_debate_skipped():
    """ct_debate_skipped_total 计数器可递增。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before = _sample_value(REGISTRY, "ct_debate_skipped_total", {})
    mc.inc_debate_skipped()
    after = _sample_value(REGISTRY, "ct_debate_skipped_total", {})
    assert after == (before or 0) + 1


def test_inc_verdict():
    """ct_verdict_total 计数器按 action 标签递增。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before = _sample_value(REGISTRY, "ct_verdict_total", {"action": "buy"})
    mc.inc_verdict(action="buy")
    after = _sample_value(REGISTRY, "ct_verdict_total", {"action": "buy"})
    assert after == (before or 0) + 1


def test_inc_risk_rejected():
    """ct_risk_rejected_total 计数器按 check_name 标签递增。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": "volatility"})
    mc.inc_risk_rejected(check_name="volatility")
    after = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": "volatility"})
    assert after == (before or 0) + 1


def test_inc_trade_executed():
    """ct_trade_executed_total 计数器按 engine/side 标签递增。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before = _sample_value(REGISTRY, "ct_trade_executed_total", {"engine": "paper", "side": "buy"})
    mc.inc_trade_executed(engine="paper", side="buy")
    after = _sample_value(REGISTRY, "ct_trade_executed_total", {"engine": "paper", "side": "buy"})
    assert after == (before or 0) + 1


def test_observe_execution_latency():
    """ct_execution_latency_ms Histogram 可记录观测值。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before_count = _sample_count(REGISTRY, "ct_execution_latency_ms", {"engine": "paper"})
    mc.observe_execution_latency(engine="paper", ms=42.5)
    after_count = _sample_count(REGISTRY, "ct_execution_latency_ms", {"engine": "paper"})
    assert after_count == (before_count or 0) + 1


def test_observe_pipeline_duration():
    """ct_pipeline_duration_ms Histogram 可记录观测值。"""
    from prometheus_client import REGISTRY

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    before_count = _sample_count(REGISTRY, "ct_pipeline_duration_ms", {})
    mc.observe_pipeline_duration(ms=1234.0)
    after_count = _sample_count(REGISTRY, "ct_pipeline_duration_ms", {})
    assert after_count == (before_count or 0) + 1


def test_generate_latest_contains_metric_names():
    """generate_latest() 输出包含所有预期指标名称。"""
    from prometheus_client import generate_latest

    from cryptotrader.metrics import get_metrics_collector

    mc = get_metrics_collector()
    # 确保各指标至少被调用一次, 使其出现在输出中
    mc.inc_llm_calls(model="test", node="test")
    mc.inc_debate_skipped()
    mc.inc_verdict(action="hold")
    mc.inc_risk_rejected(check_name="drawdown")
    mc.inc_trade_executed(engine="live", side="sell")
    mc.observe_execution_latency(engine="live", ms=10.0)
    mc.observe_pipeline_duration(ms=500.0)

    output = generate_latest().decode("utf-8")
    assert "ct_llm_calls_total" in output
    assert "ct_debate_skipped_total" in output
    assert "ct_verdict_total" in output
    assert "ct_risk_rejected_total" in output
    assert "ct_trade_executed_total" in output
    assert "ct_execution_latency_ms" in output
    assert "ct_pipeline_duration_ms" in output


# -- GET /metrics 端点测试 --


def test_prometheus_metrics_endpoint_returns_200():
    """GET /metrics 返回 200 状态码。"""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200


def test_prometheus_metrics_endpoint_content_type():
    """GET /metrics 响应的 Content-Type 为 Prometheus 文本格式。"""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    r = client.get("/metrics")
    assert "text/plain" in r.headers["content-type"]


def test_prometheus_metrics_endpoint_contains_ct_metrics():
    """GET /metrics 响应体包含 ct_ 前缀的指标。"""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    r = client.get("/metrics")
    body = r.text
    # 至少包含一个 ct_ 指标
    assert "ct_" in body


# -- 辅助函数 --


def _sample_value(registry, metric_name: str, labels: dict) -> float | None:
    """从 Prometheus 注册表中读取 Counter/Gauge 样本值。

    prometheus_client 的 m.name 是不含 _total 后缀的基础名称;
    Counter 的样本名为 {base}_total, Gauge 的样本名等于基础名称。
    metric_name 参数传入完整名称(如 ct_llm_calls_total),
    此函数自动推导基础名称。
    """
    base_name = metric_name.removesuffix("_total")
    for m in registry.collect():
        if m.name == base_name:
            for sample in m.samples:
                if (sample.name == f"{base_name}_total" or sample.name == base_name) and _labels_match(
                    sample.labels, labels
                ):
                    return sample.value
    return None


def _sample_count(registry, metric_name: str, labels: dict) -> float | None:
    """从 Prometheus 注册表中读取 Histogram _count 样本值。"""
    base_name = metric_name.removesuffix("_total")
    for m in registry.collect():
        if m.name == base_name:
            for sample in m.samples:
                if sample.name == f"{base_name}_count" and _labels_match(sample.labels, labels):
                    return sample.value
    return None


def _labels_match(sample_labels: dict, expected: dict) -> bool:
    """检查 sample_labels 是否包含 expected 中的所有键值对。"""
    return all(sample_labels.get(k) == v for k, v in expected.items())

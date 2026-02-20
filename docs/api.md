# API Reference

## Endpoints

### POST /analyze
Run full analysis cycle for a pair.

```json
{"pair": "BTC/USDT", "timeframe": "1h"}
```

### GET /journal/log?limit=10&pair=BTC/USDT
List recent decisions.

### GET /journal/{hash}
Get decision detail.

### GET /health
System health check.

## Start Server

```bash
arena serve --port 8003
```

OpenAPI docs available at `http://localhost:8003/docs`.

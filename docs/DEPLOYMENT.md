# Deployment

This bot runs continuously and trades on a real account (sandbox or live).
Running it on a developer laptop is not production-safe — sleep, lid-close,
Docker Desktop crashes, and macOS daemon recovery glitches all cause silent
outages. The recommended target is a small Linux VPS.

This doc walks through provisioning that VPS end-to-end with the existing
`docker-compose.yml`, plus Caddy for HTTPS-fronted dashboard access.

## Sizing & host choice

| Item | Recommendation | Why |
|---|---|---|
| Provider | Hetzner CPX21 ($6/mo) or DigitalOcean Basic ($6/mo) | Tested price/performance for 4-service compose |
| RAM | **4 GB minimum** | postgres + redis + api + scheduler + caddy comfortably; AWS t3.micro 1 GB OOMs under load |
| CPU | 2 vCPU shared | Cycles peak at ~30s of LLM I/O wait, never CPU-bound |
| Disk | 80 GB SSD | Postgres journal grows ~50 MB/day; volume snapshots fit easily |
| Region | **Singapore** for OKX HK; **Tokyo** for Binance | Round-trip < 30 ms to exchange REST endpoints |
| OS | Ubuntu 24.04 LTS | Long-term support, default Docker repo Just Works |

Avoid: AWS Free Tier (RAM too small), Fargate (cold start drifts cycle
schedule), Kubernetes (one bot on K8s is 10× the operational cost).

## Architecture on the VPS

```
                   ┌─────────────────────────────────────┐
                   │            VPS (Ubuntu 24.04)        │
                   │                                     │
  internet  →  443 │  caddy  ──/api/*──→  api:8003        │
                   │     │   ──/*──────→  web:80          │
                   │     ↑                ↓               │
                   │  Let's Encrypt       │               │
                   │  auto-renew     postgres:5432        │
                   │                 redis:6379           │
                   │                 scheduler (no port)  │
                   │                                     │
                   │  /var/lib/docker/volumes/{pgdata,    │
                   │     redisdata, ctdata, caddydata}    │
                   │                                     │
                   │  /etc/cron.d/cryptotrader-backup     │
                   │     daily pg_dump → Backblaze B2     │
                   └─────────────────────────────────────┘
```

All inter-service traffic stays on the docker bridge network. **Only ports
22, 80, 443 are exposed to the internet** (postgres/redis/api/scheduler are
docker-internal only).

## One-time host setup (15 min)

```bash
# 1. SSH in as root
ssh root@<vps-ip>

# 2. Create non-root user with docker + sudo
adduser trader
usermod -aG sudo trader
# Drop password login, key auth only:
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload ssh

# 3. Install Docker (official repo)
curl -fsSL https://get.docker.com | sh
usermod -aG docker trader

# 4. Firewall — only 22/80/443
apt-get install -y ufw
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp
ufw --force enable

# 5. Switch to trader user for everything below
su - trader
```

## App deployment (10 min)

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/cryptotrader-ai.git
cd cryptotrader-ai

# 2. Populate secrets
cp .env.example .env
chmod 600 .env
$EDITOR .env
# Required:
#   POSTGRES_PASSWORD=<strong-random>
#   API_KEY=<strong-random>      # frontend uses this, do not commit
#   AUTH_MODE=enabled            # production
# Plus exchange creds in config/local.toml (NOT in git):
#   [exchanges.okx]
#   api_key = "..."
#   secret = "..."
#   passphrase = "..."
#   sandbox = true               # flip to false for live trading

# 3. Add Caddy to compose (one-time edit, see deploy/Caddyfile in this repo)
# Reverse-proxies /api/* to api:8003 and / to web:80, auto Let's Encrypt.

# 4. Bring everything up
docker compose up -d

# 5. Watch the first scheduler cycle land
docker compose logs -f scheduler
# Look for: "Cycle complete [BTC/USDT] action=... risk=..."
```

If you have an existing local database to migrate:

```bash
# On laptop
PGPASSWORD=<local> pg_dump -h localhost -U postgres cryptotrader \
  | gzip > /tmp/cryptotrader.sql.gz
scp /tmp/cryptotrader.sql.gz trader@<vps-ip>:/tmp/

# On VPS
gunzip -c /tmp/cryptotrader.sql.gz \
  | docker compose exec -T postgres psql -U cryptotrader -d cryptotrader
```

## Domain + HTTPS (5 min)

1. Point a domain at the VPS:

   ```
   trader.yourdomain.com.   A    <vps-ip>   TTL 300
   ```

2. Edit `deploy/Caddyfile` to use that hostname (default ships as
   `trader.example.com`).

3. `docker compose restart caddy` — Caddy fetches a Let's Encrypt cert
   automatically the first time it sees an HTTPS request to that host.

4. Visit `https://trader.yourdomain.com` — dashboard.

## Backups (one-time + daily)

`docker compose` already creates named volumes for `pgdata`, `redisdata`,
`ctdata`. Back up the postgres volume offsite daily:

```bash
# Install rclone (Backblaze B2 is cheapest at $0.005/GB/month)
sudo apt-get install -y rclone
rclone config       # add a B2 remote called 'b2'

# Drop a daily backup into /etc/cron.d/cryptotrader-backup
sudo tee /etc/cron.d/cryptotrader-backup <<'EOF'
0 3 * * * trader cd /home/trader/cryptotrader-ai && \
  docker compose exec -T postgres pg_dump -U cryptotrader cryptotrader \
  | gzip | rclone rcat b2:cryptotrader-backups/$(date +\%Y-\%m-\%d).sql.gz \
  >> /var/log/cryptotrader-backup.log 2>&1
EOF
```

Costs: ~30 days of dumps at ~10 MB each = 0.3 GB = $0.0015/month on B2.

## Monitoring (free)

Add `https://trader.yourdomain.com/health` to UptimeRobot:

- Type: HTTP keyword monitor
- Keyword: `"status":"ok"`
- Interval: 5 minutes
- Alert contacts: email or Telegram

When the bot goes degraded (DB down, redis down, LLM creds expired) you get
a notification within 5 minutes instead of finding out hours later.

## Telegram alerts on trade events

The repo already supports this — fill in `[notifications.telegram]` in
`config/local.toml`:

```toml
[notifications.telegram]
bot_token = "<from BotFather>"
chat_id = "<your numeric chat id>"

[notifications]
events = ["trade_filled", "circuit_breaker", "risk_block", "scheduler_error"]
```

Restart api+scheduler. Real fills will buzz your phone.

## Updates / CI

The repo ships two GitHub Actions workflows:

- `.github/workflows/ci.yml` — lint + test + docker build, runs on every PR
  and push to main. Reusable via `workflow_call`.
- `.github/workflows/deploy.yml` — calls `ci.yml` first, then SSHs into the
  VPS, pulls main, runs `arena migrate`, brings the stack up under
  `--profile prod` (Caddy + api + web + scheduler + postgres + redis), and
  fails if `/health` doesn't return 200.

### Required secrets (Settings → Secrets → Actions)

| Secret | Example | Notes |
|---|---|---|
| `DEPLOY_HOST` | `203.0.113.42` | VPS IP or hostname |
| `DEPLOY_USER` | `trader` | Non-root user from "host setup" above |
| `SSH_PRIVATE_KEY` | full ed25519 private key (PEM) | A *deploy-only* keypair, public half in `~trader/.ssh/authorized_keys` on the VPS |
| `DEPLOY_PORT` | `22` | Optional, defaults to 22 |

Optional environment variable on the runner (set as a repo Variable, not a
Secret) — `DEPLOY_DIR`, defaults to `/home/trader/cryptotrader-ai`.

The workflow runs in the `production` GitHub Environment, so you can add a
required reviewer there if you want a human to approve every deploy.

### Generate a deploy key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/cryptotrader-deploy -C "github-actions-deploy" -N ""
# Add the public half on the VPS:
ssh-copy-id -i ~/.ssh/cryptotrader-deploy.pub trader@<vps-ip>
# Paste the *private* half into the GitHub secret SSH_PRIVATE_KEY.
```

### Manual fallback

`ssh trader@<vps> 'cd cryptotrader-ai && git pull && docker compose --profile prod up -d --build'`.

## Rolling cost

| Item | Monthly |
|---|---|
| VPS (Hetzner CPX21 / DO Basic) | $6 |
| Domain (.com via Namecheap) | $1 |
| Backblaze B2 (10 GB usage) | $0.05 |
| UptimeRobot free tier | $0 |
| Let's Encrypt (Caddy auto) | $0 |
| **Total** | **~$7** |

## Going live (sandbox → real money)

The sandbox-to-live flip is small but consequential:

1. In `config/local.toml`, change `[exchanges.okx] sandbox = false` and
   replace api_key/secret/passphrase with **production** OKX credentials.
2. **Re-evaluate risk thresholds** in `config/default.toml`. The
   `max_daily_loss_pct = 0.08` / `max_drawdown_pct = 0.30` defaults were
   relaxed for low-stakes sandbox observation; tighten for real money
   (e.g. 0.03 / 0.10 — closer to the original tight defaults).
3. Set a smaller `[risk.position] max_single_pct` for the first weeks
   (e.g. 0.10 = 10% of equity per trade, not 0.90).
4. Run `arena live-check` (already in the CLI) to confirm credentials,
   trading pair listings, and rate-limit headroom before flipping
   `sandbox = false`.
5. **Watch the first 5 cycles in `docker compose logs -f scheduler`.**
   Don't walk away.

## What this doc deliberately skips

- **Kubernetes** — overkill for one bot.
- **Multi-region failover** — adds 10× complexity for a use case where
  missing a 1h cycle costs ~nothing.
- **Secret managers** (Vault / AWS Secrets Manager) — `.env` with 600
  perms is fine until you have multiple operators.
- **WAL streaming replication** — daily pg_dump covers the worst case.

Add any of these only when you outgrow the simple setup.

# 部署

CryptoTrader AI 的运行配置保存在 PostgreSQL 的 `runtime_config` 表。部署进程唯一的外部引导参数是 `DATABASE_URL` 与 `CONFIG_MASTER_KEY`；不要挂载或维护 TOML、`.env`、交易所模式或 API 密钥环境变量。

## 前置条件

- PostgreSQL 16+；
- Docker Compose 或 Python 3.12+、uv；
- 一个 base64 编码的 32 字节 AES-GCM `CONFIG_MASTER_KEY`。主密钥用于加密网页提交的平台凭据，丢失后旧凭据不可恢复。

## Compose

```bash
export CONFIG_MASTER_KEY='base64 编码的 32 字节密钥'
docker compose build
docker compose up -d postgres redis
docker compose run --rm --no-deps api trader schema migrate
docker compose up -d
```

Compose 负责创建 PostgreSQL，并只向 API 与调度器传入数据库地址和主密钥。迁移命令显式安装当前 Workbench schema，不加载 Runtime，也不会创建订单。API 健康检查通过后打开网页，完成初始化向导；容器不读取本地配置文件。

## 首次配置

初始化向导依次保存 LLM、信号组件、市场数据、平台连接、执行资金池、风控、调度和通知。每次保存递增全局配置版本；页面会显示版本、校验错误与保存错误。

可同时创建 Paper、Demo、Testnet 和 Live 连接：

- Paper、Demo、Testnet 只可分配到 `simulated` 资金池；
- Live 只可分配到 `real` 资金池；
- 一个连接最多属于一个启用资金池；
- 每个资金池单独配置 HITL；审批冻结整个计划、保护价格与配置版本。

先在 Demo/Testnet 完成真实模型与只读连接检查。Live 连接只允许只读检查，不允许自动化真实资金订单。

## Staging 门禁

```bash
export DATABASE_URL='postgresql+asyncpg://<db-user>:<db-password>@host:5432/cryptotrader'
export CONFIG_MASTER_KEY='base64 编码的 32 字节密钥'
uv run python scripts/staging_validate.py
```

门禁顺序检查数据库 schema 与激活的配置版本、唯一运行时周期、每个已启用平台连接的只读连接/关闭。它从不创建订单；缺失配置、周期不可装配或任一连接不健康都会失败。

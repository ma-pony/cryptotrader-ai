# cryptotrader-web

CryptoTrader AI 的 React 19、Vite 8、TypeScript 5.9 单页控制台。网页读取数据库运行时配置的脱敏视图；不读取本地前端环境文件，也不保存平台凭据或 API 密钥。

## 本地开发

先启动后端并提供唯一的两个引导变量：

```bash
export DATABASE_URL='postgresql+asyncpg://<db-user>:<db-password>@localhost:5432/cryptotrader'
export CONFIG_MASTER_KEY='base64 编码的 32 字节密钥'
uv run trader serve --port 8003
```

然后启动网页：

```bash
pnpm install
pnpm dev
```

浏览器访问 `http://localhost:5173`。API 使用同源相对路径；开发服务器将请求转给本地 API，无需 `.env.local` 或 `VITE_*` 配置。

## 初始化与配置

未激活的运行时会直接显示初始化向导。向导和配置页写入数据库 `runtime_config`，每次成功保存都会显示新的配置版本；冲突、校验和服务错误会留在页面上，当前运行配置不被覆盖。

可在网页配置并查看：

- 市场数据来源，以及 Kronos、四智能体内部辩论和自定义信号组件的权重与参数；
- 同时存在的 Paper、Demo、Testnet、Live 平台连接及其只读连通性检查；
- `simulated` 与 `real` 执行资金池、固定连接分配权重和每资金池 HITL；
- 周期、融合结论、委员会辩论、资金池/连接执行结果及需要人工处理的状态。

平台凭据只在提交时发送到后端加密存储，随后只显示是否已配置，永不回显。

## 脚本

- `pnpm dev`：Vite 开发服务器。
- `pnpm build`：生产构建。
- `pnpm preview`：本地预览生产构建。
- `pnpm lint`：ESLint。
- `pnpm typecheck`：`tsc --noEmit`。
- `pnpm test`：Vitest 单元与组件测试。
- `pnpm test:e2e`：Playwright 端到端测试，需要已启动的本地服务。

详细前端架构见仓库根 [docs/frontend-architecture.md](../docs/frontend-architecture.md)。

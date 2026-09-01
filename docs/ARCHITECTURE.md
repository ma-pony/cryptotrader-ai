# 运行架构与扩展指南

当前架构维护在 [根目录架构说明](../ARCHITECTURE.md)。配置与代码注册扩展开发使用 [配置指南](CONFIGURATION.md)，其中包含类型化注册示例。

运行配置保存在数据库。旧版 Profile repository、`build_trading_cycle()`、零参数插件工厂和 `[signal_plugins]` TOML 示例不适用于当前运行时。组件、适配器与连接类型只由后端代码注册；不会扫描 Python package entry points、运行时目录或已安装包。

验证结果、隔离预览命令及部署前清理边界见 [配置中心验收记录](verification/configuration-center/README.md)。

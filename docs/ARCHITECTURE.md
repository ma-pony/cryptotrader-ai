# 运行架构与扩展指南

当前架构维护在 [根目录架构说明](../ARCHITECTURE.md)。配置与插件开发使用 [配置指南](CONFIGURATION.md)，其中包含可安装的类型化示例。

运行配置保存在数据库。旧版 Profile repository、`build_trading_cycle()`、零参数插件工厂和 `[signal_plugins]` TOML 示例不适用于当前运行时。安装插件通过 Python package entry points，配置元数据由工厂携带的 `PluginConfiguration` 提供。

验证结果、隔离预览命令及部署前清理边界见 [配置中心验收记录](verification/configuration-center/README.md)。

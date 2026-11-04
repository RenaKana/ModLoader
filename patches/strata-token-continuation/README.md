# Strata 原始 token 续接补丁

普通纯文本 `/v1/chat/completions` 完成后，保留一份完整历史到原始 token 的映射，避免生成文本再次 BPE 编码或工具解析时丢失原始分词与尾部换行。无需识别客户端或对话轮次。

## Strata 0.2.4：应用与开启

两份补丁必须配套使用，先应用到独立源码副本：

- `native.patch`：基于 `6f8800aefe27819ac89545c86f7686c094f099a4`，基线需已包含相邻目录的 startup-prefetch 补丁。修改 EOS/长度上限处的实际 state commit、pipeline 回滚及 consumed token 记录。
- `server.patch`：基于 `manifest.json` 指定的服务端版本。修改 `serve/server.py`，新增 `serve/raw_continuation.py` 与离线测试。

对各自副本执行 `git apply --check <patch>` 后再 `git apply <patch>`；重编 native 引擎后，在加载配置的额外参数中加入 `--raw-token-continuation`，即可配套开启紧凑数组映射和原生边界修复。独立服务端也支持配置 `"raw_token_continuation": true`。判定使用 `user_args` 合并后的有效参数；未显式开启时仍走旧路径。不能沿用不支持该参数的旧引擎。

0.1.48 分支的 `build-strata-prefetch.ps1` 不自动应用这份附加补丁；重建该版本时需把它应用到匹配版本的独立源码中，不要直接拿已修改副本交给会重新导出 Git HEAD 的流程。

## 缓存契约

- 每个 Service 仅保留一个完整前缀，完成后原子替换，不按历史轮次累积。canonical/raw token 均为 32 位紧凑数组，不新增 KV 或显存副本。
- 完整历史、返回 assistant 字段、工具定义、模板文本及 literal spans 必须匹配；不在 assistant 内部截取。实际 EOS 必须与模板结束 token 相同。
- 在上下文余量与 usage 计算前替换实际输入 IDs；PromptEncoder 仍返回完整 canonical BPE。
- 只登记单次正常自然结束的完整回答或工具调用。图片、显式 `strata_prefix`、MCP、forced call、structured response、DeepSeek/effort-end 等路径照旧；取消、错误、截断、custom stop、未完成工具调用、内部重试/recovery 不登记。

例如，112,284 和 112,279 token 的两份数组载荷合计为 `4 × (112284 + 112279) = 898252 B`（约 0.86 MiB）；文本、元数据和请求期间的临时缓冲另计。

## 补丁内的检查

服务端补丁包含 `serve/test_raw_token_continuation.py`，覆盖真实 BPE、非 canonical 分词、工具 XML 尾换行、连续续接、EOS、历史与工具修改、取消和错误等情况。原生补丁将 `decode_boundary_test` 接入 CMake 测试，检查 EOS、长度边界和 pipeline 回滚。

应用补丁后，在匹配版本的源码和依赖环境中运行这些检查，再构建引擎。服务端和引擎需成套更新；保留原入口，以便在停止相关模型后回退。

## Strata 0.2.10 / engine 0.1.56

- `server-0.2.10.patch` 对应 Control 提交 `ffa80d73133ea5c548aefe57889a16e8085a938f`；新安装由 `strata_patches.apply_server_patch` 按精确上下文应用，重复调用不重复修改。
- 原生两项功能合并在 `../strata-prefetch/startup-expert-prefetch-0.1.56.patch`；构建器按 0.1.56 分支选择，并从目标引擎清单的提交获取源码。Windows 官方引擎提交为 `06698c674673008395ba6e971580538d6b1ea6e0`，与 Control 提交不同。
- 原始 token 续接与预取独立开启。关闭预取但指定 `--raw-token-continuation` 时，仍可选择带该功能的已准备引擎，不附加预取参数。

补丁的上游归属与许可见[第三方声明](../../THIRD_PARTY_NOTICES.md#strata-补丁)。

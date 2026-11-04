# 开发与构建

[返回首页](../README.md) · [使用指南](usage.md) · [运行环境与模型格式](backends.md)

ModLoader 使用 Electron / React 界面和 Python 核心。以下步骤在 Windows 项目根目录的 PowerShell 中执行。

## 准备工具

- Python 3.12，下面的命令使用它创建项目虚拟环境。
- Node.js 22.12 或更新的 22.x 版本，以及随附的 npm。
- .NET Framework 4.7.2 或更新版本，用于构建和运行 Windows x64 温度读取器，不需要 .NET SDK。
- [官方 NuGet CLI](https://www.nuget.org/downloads)，用于首次恢复温度读取器依赖。

模型运行环境独立于项目虚拟环境。只开发界面或核心不需要预先安装所有模型后端；实际加载模型时，再按需求安装对应环境。

## 首次启动源码版

1. 准备 Python 和前端依赖：

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   npm ci
   ```

2. 把官方下载的 `nuget.exe` 放到 `work/cpu-sensors-tools/nuget.exe`，然后恢复并编译温度读取器：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-cpu-sensors.ps1 -Restore
   ```

   也可通过 `-NuGetPath` 指定自己的 NuGet CLI 路径。依赖版本由 [packages.config](../native/cpu-sensors/packages.config) 固定，恢复到 `work/cpu-sensors-packages`；产物位于 `core/modloader/sensors`。

3. 构建界面并启动：

   ```powershell
   .\Start-ModLoader-Dev.cmd --build
   ```

`--build` 会编译本地温度读取器并执行 `npm run build`，失败时不启动应用。它不会自动下载温度读取器依赖，也不会安装驱动。前端使用现有构建时，可以直接运行：

```powershell
.\Start-ModLoader-Dev.cmd
```

修改 React/样式后重新执行 `--build`；Python 核心或 Electron 主进程修改后需要完全退出并重启。界面刷新只重新加载前端，不会重启 Python 核心。

根目录的 `Start-ModLoader.cmd` 优先启动 `release/win-unpacked/ModLoader.exe`，没有桌面包时才使用源码入口。开发时使用 `Start-ModLoader-Dev.cmd`，避免误开旧桌面包。本机可选的 `Start-ModLoader.local.cmd` 供启动覆盖使用，不纳入版本控制。

## 隔离开发数据

源码启动默认使用项目的 `data` 目录。根启动脚本也默认使用此目录；直接启动打包目录内的 EXE 则默认使用 EXE 旁的 `data`。

需要独立数据时，在启动前设置环境变量：

```powershell
$env:MODLOADER_DATA_DIR = Join-Path (Get-Location) 'work/dev-data'
.\Start-ModLoader-Dev.cmd
```

不要让两个版本同时操作同一份数据。切换前从托盘“退出并停止模型”，关闭到托盘不算退出。隔离数据目录不会复制已导入的外部模型文件。

温度读取器在完整启动时请求独立管理员授权。无交互测试可在当前进程中关闭自动启动：

```powershell
$env:MODLOADER_SENSOR_AUTO_START = '0'
```

这只影响温度读取，不改变其他硬件指标。驱动和温度组件说明见 [native/cpu-sensors/THIRD-PARTY.md](../native/cpu-sensors/THIRD-PARTY.md)。

## 运行相关测试

Python 测试从 [pytest.ini](../pytest.ini) 读取 `core` 模块路径。按改动范围选择测试，例如核心与基础后端：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_backends.py -q
```

前端构建同时执行 TypeScript 检查：

```powershell
npm run build
```

已准备好本地界面和温度读取器后，可以运行指定的 Playwright 测试，例如使用本机模拟服务的对话测试：

```powershell
npm run test:ui -- tests/ui-chat.spec.ts
```

Playwright 配置默认关闭自动温度授权。`tests/test_sources.py` 使用本机 HTTP fixture 测试模型来源协议；`tests/ui-native.spec.ts` 包含真实网络搜索。

[ui-web-search.spec.ts](../tests/ui-web-search.spec.ts) 中的真实搜图用例默认跳过。启用时需同时设置 `MODLOADER_LIVE_IMAGE_SEARCH=1` 和指向自备 SearXNG 服务的 `MODLOADER_TEST_SEARXNG_URL`；需要代理时另设 `MODLOADER_TEST_PROXY_URL`。

`scripts/prepare_acceptance.py` 等真实环境验收脚本会安装环境、下载模型，运行前应查看脚本的输入和文件位置，不与纯 fixture 测试混用。

## 构建桌面包

先完成上述开发准备，再打包 Python 核心和桌面程序：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-core.ps1
npm run package
```

`build-core.ps1` 使用 PyInstaller，将核心、模型来源工作进程所需模块、温度读取器和补丁资源打包到 `work/core-dist/modloader-core`。

`npm run package` 生成图标、收集依赖许可、构建界面，并由 electron-builder 输出 `release/win-unpacked`。这个目录整体构成桌面包，入口是其中的 `ModLoader.exe`；Python 核心已包含在包中，不需要收件人安装项目 `.venv`。模型、运行环境和个人数据需要另行准备。

桌面打包会自动附带 [LICENSE](../LICENSE)、[第三方声明](../THIRD_PARTY_NOTICES.md)、[第三方许可目录](../third_party/licenses)、[源码获取说明](../third_party/SOURCES.md)、`licenses/Outfit-OFL.txt` 和 `licenses/dependencies/`。许可收集脚本缺少完整文本或发现锁定版本不匹配时会报错；上游包未携带的许可按固定版本记录在 `third_party/npm-license-supplements.json`。运行环境和模型的许可独立适用。

## 代码入口

| 位置 | 职责 |
| --- | --- |
| [src/pages](../src/pages) | 模型、下载、运行环境、对话、监控和设置页面 |
| [electron/main.cjs](../electron/main.cjs) | 桌面窗口、托盘、核心进程和系统交互 |
| [core/modloader/app.py](../core/modloader/app.py) | 管理 API、公开推理 API 和核心协调 |
| [core/modloader/backends](../core/modloader/backends) | 后端能力、启动适配、版本目录和安装配方 |
| [core/modloader/sources](../core/modloader/sources) | 模型仓库查询、文件计划和下载 |
| [native/cpu-sensors](../native/cpu-sensors) | 温度读取器源码、依赖与许可 |
| [tests](../tests) | Python、Electron 和界面测试 |

问题反馈、代码贡献和安全报告分别见 [CONTRIBUTING.md](../CONTRIBUTING.md) 与 [SECURITY.md](../SECURITY.md)。

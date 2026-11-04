# 第三方源码获取

桌面包使用未修改的上游依赖。包内 `licenses/dependencies/` 保存 npm 生产依赖、Electron 和 Python 构建环境的原始许可与版本清单；Python 清单也涵盖仅用于构建和测试的包。Electron 与 Chromium 的声明另见包根目录的 `LICENSE.electron.txt` 和 `LICENSES.chromium.html`。

以下链接提供需保留源码获取方式的组件。可以按链接获取、修改和重新构建；各组件的具体权利和条件以附带许可证为准。

| 组件 | 版本 | 对应源码 |
| --- | --- | --- |
| Viz.js | 3.31.0 | [固定提交源码](https://github.com/mdaines/viz-js/archive/c8ce050d28eb0dc1ed1629ea32ab7f3dc117db0f.zip)；其 WebAssembly 还包含下面的 Graphviz 和 Expat。 |
| Graphviz | 16.1.0，EPL-2.0 | [官方源码包](https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/16.1.0/graphviz-16.1.0.tar.gz)，版本来自 Viz.js 的 `lib/provenance.json`。 |
| Expat | 2.8.5，MIT | [官方源码包](https://github.com/libexpat/libexpat/releases/download/R_2_8_5/expat-2.8.5.tar.gz)。 |
| elkjs / ELK | 0.9.3，EPL-2.0 | [elkjs 固定提交源码](https://github.com/kieler/elkjs/archive/a8304cf79fde75bc2ab1a89d28320f53f8637436.tar.gz)。按[官方发布说明](https://github.com/kieler/elkjs/releases/tag/0.9.3)，Java 部分基于 [ELK 0.9.1 源码](https://github.com/eclipse-elk/elk/archive/refs/tags/v0.9.1.tar.gz)，另含 [PR #955 的改动](https://github.com/eclipse-elk/elk/pull/955)。完整上游源码历史见 [eclipse-elk/elk](https://github.com/eclipse-elk/elk)。 |
| DOMPurify | 3.4.17，MPL-2.0 OR Apache-2.0 | [上游源码](https://github.com/cure53/DOMPurify/tree/3.4.17)；完整双许可文本随包保留。 |
| certifi | 2026.7.22，MPL-2.0 | [PyPI 原始源码包](https://files.pythonhosted.org/packages/a3/c2/24167ea9858356b47a87a50d39908bfdb72ceeefe0041586e704e5376b3a/certifi-2026.7.22.tar.gz)。 |
| tqdm | 4.70.1，MPL-2.0 AND MIT | [PyPI 原始源码包](https://files.pythonhosted.org/packages/0d/ea/b2a5bd54b28a324dae8211928b2d730b6547500342c7e6c6dea08bd0a485/tqdm-4.70.1.tar.gz)。 |
| PyInstaller | 6.22.3，GPLv2-or-later with bootloader exception | [上游源码及构建说明](https://github.com/pyinstaller/pyinstaller/tree/v6.22.3)；引导程序未修改，完整许可和例外条款在 Python 依赖声明中。 |

温度读取组件的版本、源码地址和构建说明见 `resources/core/_internal/modloader/sensors/THIRD-PARTY.md`，包括 LibreHardwareMonitor、PawnIO.Modules 和其 NuGet 依赖。组件 DLL 与应用分别保存，可替换为兼容构建。

其他依赖的版本、上游仓库及软件包地址见 `licenses/dependencies/index.json`。Strata 补丁、NInfer 兼容实现和字体的来源见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

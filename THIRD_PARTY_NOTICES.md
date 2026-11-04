# 第三方声明

ModLoader 自有代码使用 [MIT 许可](LICENSE)。以下材料保留原作者的版权与许可；下载的模型和运行环境也分别遵循其上游许可。

## 随源码保存的材料

### Outfit 字体

- 文件：`src/assets/fonts/outfit-latin-wght.woff2`、`outfit-latin-ext-wght.woff2`。
- 作者：The Outfit Project Authors。
- 来源：[Outfit-Fonts](https://github.com/Outfitio/Outfit-Fonts)。
- 许可：SIL Open Font License 1.1，全文见源码中的 `src/assets/fonts/OFL.txt`；桌面包中的副本为 `licenses/Outfit-OFL.txt`。

### Strata 补丁

`patches/strata-prefetch/` 和 `patches/strata-token-continuation/` 包含对 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的修改：启动专家预取、原始 token 续接，以及相关测试与构建支持。适用基线分别记录在补丁和清单中，包括：

- `6f8800aefe27819ac89545c86f7686c094f099a4`
- `06698c674673008395ba6e971580538d6b1ea6e0`
- `96ca4383c7b05a8a3e6064c65d9e9977d4b3bb59`
- `ffa80d73133ea5c548aefe57889a16e8085a938f`

Copyright (c) 2026 Niko1221 and the Strata contributors。上游采用 MIT，完整文本见 [Strata-MIT.txt](third_party/licenses/Strata-MIT.txt)。

### NInfer 兼容实现

`core/modloader/assets.py` 与 `core/modloader/backends/ninfer_tokenizer.py` 参考了 [Neroued/ninfer 的固定提交](https://github.com/Neroued/ninfer/tree/64d10fbf9f8e6bfc79ab0570b94674c5b9997929) 中的格式、读取器和 tokenizer/frontend 实现，并为 ModLoader 的模型检查、准备流程作了适配。源码注释保留了对应上游位置。

上游及其相关派生部分遵循 [Apache License 2.0](third_party/licenses/Apache-2.0.txt)。

### 测试用模型说明与元数据

以下文件用于格式解析和兼容性回归测试，不包含模型权重。它们保留上游的 Apache License 2.0 许可；完整许可见 [Apache-2.0.txt](third_party/licenses/Apache-2.0.txt)。摘录或固定测试样例的来源版本如下。

| 本地文件 | 来源、作者与固定版本 |
| --- | --- |
| `tests/fixtures/model-readme/README.md`、`README.zh-CN.md` | SC117 的 [Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963)，版本 `eec4e10a2c29b440a2ae85591fa11569ce752963`。两份模型说明均来自上游，含其发布的中文材料。 |
| `tests/fixtures/ninfer-qwen38-tokenizers.json` | Unsloth 的 [Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/tree/f0b7c9e722f5565102fff8481c99e4d86ae099c7)，版本 `f0b7c9e722f5565102fff8481c99e4d86ae099c7`；tokenizer 配置摘录。 |
| `tests/fixtures/ninfer-qwen35-4b-gdn.json` | Alibaba Cloud / Qwen 的 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)，版本 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；safetensors 文件头摘录。另保留含上游版权行的 [Qwen 专属许可副本](third_party/licenses/Qwen3.5-Apache-2.0.txt)。 |
| `tests/fixtures/sglang-granite-moe-legacy-header.json` | IBM 的 [granite-3.1-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/tree/0da7a48b0276d500ce5922fd2b33944091fc6c09)，版本 `0da7a48b0276d500ce5922fd2b33944091fc6c09`；safetensors 文件头摘录。 |

## 安装与打包时引入的组件

桌面包附带 `licenses/dependencies/README.md` 和 `index.json`，保存 npm 生产依赖、Electron、Python 构建环境及 CPython 的完整许可文件、版本和来源。Python 清单覆盖构建环境，包含部分仅用于构建或测试的包。需要源码的组件见[第三方源码获取](third_party/SOURCES.md)。

依赖版本由 `package-lock.json`、`requirements.lock.txt`、`requirements-dev.txt` 和 `native/cpu-sensors/packages.config` 记录。依赖包、模型运行环境和生成的二进制文件不作为 ModLoader 源码提交。

- **界面与桌面**：包括 Electron、React、Mermaid、KaTeX、Lucide、D3、DOMPurify、ELK.js 等。除 MIT/ISC 外，依赖还使用 Apache-2.0、BSD、MPL-2.0、EPL-2.0 和 Unlicense 等许可，不能统一改标为 MIT。
- **Graphviz 渲染**：Viz.js 3.31.0 使用 MIT，其 WebAssembly 中的 Graphviz 16.1.0 使用 EPL-2.0，Expat 2.8.5 使用 MIT；三者的原始许可分别保留于 `third_party/licenses`，固定版本来自 Viz.js 的构建来源清单。
- **Python 核心**：依赖按各包许可使用，包括 BSD、Apache-2.0、MPL-2.0 等。PyInstaller 的引导程序使用 GPLv2-or-later，并带有允许分发冻结应用的 bootloader exception。
- **温度读取器**：组件及许可详见源码中的 `native/cpu-sensors/THIRD-PARTY.md` 和 `native/cpu-sensors/licenses/`。包括 LibreHardwareMonitor（MPL-2.0）、PawnIO.Modules（LGPL-2.1-or-later）等。构建脚本将组件提供的许可文件复制到读取器输出目录。PawnIO 驱动由用户另行安装，不随 ModLoader 分发。

分发桌面包时，应按最终装入包的组件保留版权声明和完整许可文本；对 MPL、EPL、LGPL 等组件，按其条款提供所需源码或源码获取方式。

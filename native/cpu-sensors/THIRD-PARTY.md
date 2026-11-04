# Hardware temperature reader dependencies

The helper uses unmodified LibreHardwareMonitor **0.9.6**, Windows x64 / .NET
Framework 4.7.2 runtime. `packages.config` pins the full NuGet dependency closure;
`scripts/build-cpu-sensors.ps1` selects runtime assemblies, not reference stubs.
The library DLLs remain separate and can be replaced with compatible builds.

## LibreHardwareMonitor and embedded modules

- LibreHardwareMonitor 0.9.6: **MPL-2.0**.
  Source: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/tree/v0.9.6
  Source archive: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/archive/refs/tags/v0.9.6.zip
- Embedded PawnIO.Modules 0.2.2: **LGPL-2.1-or-later**.
  Source and build instructions: https://github.com/namazso/PawnIO.Modules/tree/0.2.2
  Source archive: https://github.com/namazso/PawnIO.Modules/archive/refs/tags/0.2.2.zip
  Its license and notices are included in `licenses/LibreHardwareMonitor-NOTICES.txt`.
- The PawnIO **2.2.0** driver is installed separately with its official installer,
  not bundled or silently installed by ModLoader. Official releases:
  https://github.com/namazso/PawnIO.Setup/releases/tag/2.2.0
  Driver source/license: https://github.com/namazso/PawnIO/tree/2.2.0

## NuGet dependencies

The build copies package-provided license and notice files into `licenses/<id>`.
MPL-2.0 text is included in `licenses/LibreHardwareMonitor-MPL-2.0.txt`; Microsoft
MIT license text is included in `licenses/System.Security.AccessControl/LICENSE.TXT`.

| Package | Version | License / source |
| --- | --- | --- |
| BlackSharp.Core | 1.0.7 | MPL-2.0; https://github.com/Blacktempel/BlackSharp |
| DiskInfoToolkit | 1.1.2 | MPL-2.0; https://github.com/Blacktempel/DiskInfoToolkit |
| RAMSPDToolkit-NDD | 1.4.2 | MPL-2.0; https://github.com/Blacktempel/RAMSPDToolkit |
| HidSharp | 2.6.4 | Apache-2.0; https://software.seekye.com/hidsharp |
| System.Buffers | 4.6.1 | MIT; https://github.com/dotnet/maintenance-packages |
| System.Memory | 4.6.3 | MIT; https://github.com/dotnet/maintenance-packages |
| System.Numerics.Vectors | 4.6.1 | MIT; https://github.com/dotnet/maintenance-packages |
| System.Runtime.CompilerServices.Unsafe | 6.1.2 | MIT; https://github.com/dotnet/maintenance-packages |
| System.CodeDom | 10.0.2 | MIT; https://github.com/dotnet/runtime |
| System.Management | 10.0.2 | MIT; https://github.com/dotnet/runtime; Framework assembly used on net472 |
| System.Threading.AccessControl | 10.0.3 | MIT; https://github.com/dotnet/runtime |
| System.Security.AccessControl | 6.0.0 | MIT; https://github.com/dotnet/runtime |
| System.Security.Principal.Windows | 5.0.0 | MIT; https://github.com/dotnet/runtime |

Original package metadata and source references are also available at
`https://www.nuget.org/packages/<Package>/<Version>` for each pinned entry.

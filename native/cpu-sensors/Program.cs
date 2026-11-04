// CPU/GPU-only, per-parent helper. Build as x64 winexe with the .NET Framework C# 5
// compiler and LibreHardwareMonitor 0.9.6's net472 runtime dependencies.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Pipes;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using LibreHardwareMonitor.Hardware;
using LibreHardwareMonitor.Hardware.Gpu;
using Microsoft.Win32.SafeHandles;

internal static class Program
{
    private const int ConnectTimeoutMs = 30000;
    private static readonly JavaScriptSerializer Json = new JavaScriptSerializer();

    private static int Main(string[] args)
    {
        // --probe is read-only and does not elevate or install anything. A
        // winexe writes to an explicitly inherited stdout handle, not a console.
        if (args.Length == 1 && args[0] == "--probe")
        {
            Dictionary<string, object> result;
            try
            {
                using (ICpuSession sensors = SensorFactory.Create())
                    result = sensors.Sample();
            }
            catch (Exception error) { result = Failure(error); }
            Native.WriteStdout(Json.Serialize(result) + "\n");
            return HasTemperature(result) ? 0 : 1;
        }

        int parentPid;
        if (args.Length != 4 || args[0] != "--pipe" || args[2] != "--parent-pid" ||
            !Regex.IsMatch(args[1], @"\AModLoader\.CpuSensors\.[0-9a-f]{32}\z") ||
            !int.TryParse(args[3], out parentPid) || parentPid <= 0)
            return 2;

        try
        {
            using (Process parent = Process.GetProcessById(parentPid))
            {
                if (parent.HasExited) return 2;
                SecurityIdentifier parentSid = Native.ProcessUserSid(parent.Handle);
                PipeSecurity security = new PipeSecurity();
                security.SetAccessRuleProtection(true, false);
                security.AddAccessRule(new PipeAccessRule(parentSid, PipeAccessRights.ReadWrite, AccessControlType.Allow));
                security.AddAccessRule(new PipeAccessRule(new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid, null),
                    PipeAccessRights.FullControl, AccessControlType.Allow));
                security.AddAccessRule(new PipeAccessRule(new SecurityIdentifier(WellKnownSidType.LocalSystemSid, null),
                    PipeAccessRights.FullControl, AccessControlType.Allow));
                using (NamedPipeServerStream pipe = new NamedPipeServerStream(args[1], PipeDirection.InOut, 1,
                    PipeTransmissionMode.Byte, PipeOptions.Asynchronous, 4096, 4096, security))
                using (ManualResetEvent finished = new ManualResetEvent(false))
                {
                    // Closing the pipe unblocks normal EOF/read/write paths. If
                    // a native sensor call is stuck, parent exit still cannot
                    // leave this privileged helper running indefinitely.
                    Thread watcher = new Thread(delegate()
                    {
                        while (!finished.WaitOne(500))
                        {
                            try { if (!parent.HasExited) continue; }
                            catch (InvalidOperationException) { }
                            pipe.Dispose();
                            if (!finished.WaitOne(1500)) Environment.Exit(0);
                            return;
                        }
                    });
                    watcher.IsBackground = true;
                    watcher.Start();
                    try
                    {
                        IAsyncResult connection = pipe.BeginWaitForConnection(null, null);
                        using (connection.AsyncWaitHandle)
                        {
                            if (!connection.AsyncWaitHandle.WaitOne(ConnectTimeoutMs)) return 3;
                            pipe.EndWaitForConnection(connection);
                        }
                        uint actualPid;
                        if (!Native.GetNamedPipeClientProcessId(pipe.SafePipeHandle, out actualPid) ||
                            actualPid != (uint)parentPid || parent.HasExited)
                            return 4;
                        Serve(pipe, parent);
                    }
                    finally
                    {
                        finished.Set();
                        watcher.Join(2000);
                    }
                }
            }
            return 0;
        }
        catch (IOException) { return 5; }
        catch (UnauthorizedAccessException) { return 6; }
        catch (Win32Exception) { return 6; }
        catch (ArgumentException) { return 2; }
        catch (ObjectDisposedException) { return 0; }
        catch (Exception) { return 7; }
    }

    private static void Serve(NamedPipeServerStream pipe, Process parent)
    {
        // Construct the library-owning object only after authenticating the
        // caller. The first hardware open is later, on an accepted sample.
        ICpuSession sensors = null;
        try
        {
            using (StreamReader reader = new StreamReader(pipe, new UTF8Encoding(false, true), false, 128, true))
            using (StreamWriter writer = new StreamWriter(pipe, new UTF8Encoding(false), 1024, true))
            {
                writer.AutoFlush = true;
                while (!parent.HasExited)
                {
                    string command = ReadCommand(reader);
                    if (command == null || command == "close") return;
                    Dictionary<string, object> result;
                    if (command != "sample")
                    {
                        writer.WriteLine(Json.Serialize(Result("error", "硬件温度读取请求无效", null, null)));
                        return;
                    }
                    try
                    {
                        if (sensors == null) sensors = SensorFactory.Create();
                        result = sensors.Sample();
                    }
                    catch (Exception error) { result = Failure(error); }
                    writer.WriteLine(Json.Serialize(result));
                    if (!KeepSession(result)) return;
                }
            }
        }
        finally { if (sensors != null) sensors.Dispose(); }
    }

    private static string ReadCommand(StreamReader reader)
    {
        // Only the six-byte command "sample" and five-byte command "close"
        // exist; a longer line is a protocol failure, not an arbitrary input.
        StringBuilder command = new StringBuilder();
        while (command.Length <= 6)
        {
            int next = reader.Read();
            if (next < 0) return null;
            if (next == '\n') return command.ToString();
            command.Append((char)next);
        }
        return "invalid";
    }

    internal static Dictionary<string, object> Result(string status, string reason, object value, string source)
    {
        return new Dictionary<string, object> {
            { "cpu_temperature_c", value }, { "cpu_temperature_source", source },
            { "cpu_temperature_reason", reason }, { "cpu_temperature_status", status },
            { "cpu_power_w", null }, { "cpu_power_source", null },
            { "cpu_power_reason", reason ?? "CPU 未提供封装功率传感器" },
            { "gpu_temperatures", new List<Dictionary<string, object>>() },
            { "gpu_temperature_reason", null }
        };
    }

    internal static bool KeepSession(Dictionary<string, object> result)
    {
        // CPU status keeps its old CPU-only meaning. An enumerated GPU can
        // recover from a transient missing value without another UAC prompt.
        return (string)result["cpu_temperature_status"] == "ready" || result["cpu_power_w"] != null ||
            ((List<Dictionary<string, object>>)result["gpu_temperatures"]).Count > 0;
    }

    internal static bool HasTemperature(Dictionary<string, object> result)
    {
        if ((string)result["cpu_temperature_status"] == "ready") return true;
        foreach (Dictionary<string, object> gpu in (List<Dictionary<string, object>>)result["gpu_temperatures"])
            if (gpu["temperature_c"] != null) return true;
        return false;
    }

    internal static Dictionary<string, object> Failure(Exception error)
    {
        while ((error is System.Reflection.TargetInvocationException || error is TypeInitializationException) && error.InnerException != null)
            error = error.InnerException;
        string detail;
        FileNotFoundException missing = error as FileNotFoundException;
        FileLoadException load = error as FileLoadException;
        if (missing != null)
            detail = "缺少硬件温度读取库：" + (missing.FileName ?? missing.Message);
        else if (load != null)
            detail = "硬件温度读取库无法加载：" + (load.FileName ?? load.Message);
        else if (error is BadImageFormatException)
            detail = "硬件温度读取库的架构或格式不匹配（需要 x64 .NET Framework 4.7.2）";
        else if (error is DllNotFoundException || error is TypeLoadException)
            detail = "硬件温度读取库不可用：" + error.Message;
        else if (error is UnauthorizedAccessException)
            return Result("permission_required", "硬件温度读取权限被拒绝：" + error.Message, null, null);
        else
            detail = "硬件温度读取失败（" + error.GetType().Name + "）：" + error.Message;
        return Result("error", detail, null, null);
    }
}

// Keep the entry point and pipe protocol loadable when a managed/native sensor
// dependency is missing. A failed library load then becomes a JSON error rather
// than an unhandled CLR dialog before the parent can connect.
internal interface ICpuSession : IDisposable
{
    Dictionary<string, object> Sample();
}

internal static class SensorFactory
{
    internal static ICpuSession Create() { return new SensorSession(); }
}

internal sealed class SensorSession : ICpuSession
{
    private Computer computer;
    private bool gpuInitializationAttempted;
    private string gpuInitializationReason;

    public Dictionary<string, object> Sample()
    {
        // PawnIO's library wrapper can return a fabricated zero if device open
        // failed. Check real read/write access before opening and every sample.
        Dictionary<string, object> result = Native.CheckPawnIo();
        if (computer == null)
        {
            // Open the empty Computer first, then enable categories separately:
            // a CPU initialization/access failure must not hide usable GPUs.
            computer = new Computer();
            computer.Open();
        }
        try
        {
            computer.IsCpuEnabled = result == null;
            if (result == null) result = ReadCpu(computer.Hardware);
        }
        catch (Exception error) { result = Program.Failure(error); }

        if (!gpuInitializationAttempted)
        {
            gpuInitializationAttempted = true;
            try { computer.IsGpuEnabled = true; }
            catch (Exception error)
            {
                gpuInitializationReason = (string)Program.Failure(error)["cpu_temperature_reason"];
            }
        }
        IList<IHardware> hardwareList = computer.Hardware;
        Dictionary<IHardware, uint> nvidiaBuses = Native.VerifiedNvidiaBuses(hardwareList);
        List<Dictionary<string, object>> gpus = new List<Dictionary<string, object>>();
        foreach (IHardware hardware in hardwareList)
            if (hardware.HardwareType == HardwareType.GpuNvidia || hardware.HardwareType == HardwareType.GpuAmd ||
                hardware.HardwareType == HardwareType.GpuIntel)
                gpus.Add(ReadGpu(hardware, nvidiaBuses));
        result["gpu_temperatures"] = gpus;
        result["gpu_temperature_reason"] = gpuInitializationReason;
        return result;
    }

    private static Dictionary<string, object> ReadCpu(IEnumerable<IHardware> hardwareList)
    {
        List<SensorReading> readings = new List<SensorReading>();
        List<SensorReading> packages = new List<SensorReading>();
        bool completePower = true;
        int cpuCount = 0;
        foreach (IHardware hardware in hardwareList)
            if (hardware.HardwareType == HardwareType.Cpu)
            {
                cpuCount++;
                List<SensorReading> power = new List<SensorReading>();
                UpdateCpu(hardware, readings, power);
                List<SensorReading> principalPower = power.FindAll(delegate(SensorReading reading) { return reading.Name == "CPU Package"; });
                if (principalPower.Count == 0)
                    principalPower = power.FindAll(delegate(SensorReading reading) { return reading.Name == "Package"; });
                if (principalPower.Count == 1) packages.Add(principalPower[0]);
                else completePower = false;
            }
        Dictionary<string, object> result = TemperatureResult(readings);
        if (completePower && cpuCount > 0 && packages.Count == cpuCount)
        {
            double watts = 0;
            List<string> sources = new List<string>();
            foreach (SensorReading package in packages)
            {
                watts += package.Value;
                sources.Add(package.Cpu + " · " + package.Name);
            }
            result["cpu_power_w"] = watts;
            result["cpu_power_source"] = "LibreHardwareMonitor · " + string.Join(" + ", sources.ToArray());
            result["cpu_power_reason"] = null;
        }
        else result["cpu_power_reason"] = "CPU 未提供唯一有效的封装功率传感器";
        return result;
    }

    private static Dictionary<string, object> TemperatureResult(List<SensorReading> readings)
    {
        if (readings.Count == 0)
            return Program.Result("error", "已连接 PawnIO，但 CPU 未提供有效的当前温度传感器", null, null);
        List<SensorReading> principal = readings.FindAll(delegate(SensorReading reading) { return reading.Name == "Core (Tctl/Tdie)"; });
        if (principal.Count == 0)
            principal = readings.FindAll(delegate(SensorReading reading) { return reading.Name == "CPU Package"; });
        bool fallback = principal.Count == 0;
        List<SensorReading> candidates = fallback ? readings : principal;
        SensorReading selected = candidates[0];
        foreach (SensorReading reading in candidates)
            if (reading.Value > selected.Value) selected = reading;
        string source = "LibreHardwareMonitor · " + selected.Cpu + " · " + selected.Name;
        if (fallback) source += " · CPU 传感器当前最高值";
        return Program.Result("ready", null, selected.Value, source);
    }

    private static Dictionary<string, object> ReadGpu(IHardware hardware, Dictionary<IHardware, uint> nvidiaBuses)
    {
        Dictionary<string, object> result = new Dictionary<string, object> {
            { "identifier", hardware.Identifier.ToString() }, { "name", hardware.Name },
            { "pci_bus_id", null }, { "temperature_c", null }, { "source", null }, { "reason", null }
        };
        try
        {
            GenericGpu gpu = hardware as GenericGpu;
            string pci = gpu == null ? null : Native.PciBusId(gpu.DeviceId);
            if (hardware.HardwareType == HardwareType.GpuNvidia)
            {
                uint bus;
                if (!nvidiaBuses.TryGetValue(hardware, out bus) || pci == null ||
                    !pci.StartsWith(bus.ToString("x2", CultureInfo.InvariantCulture) + ":", StringComparison.Ordinal))
                    pci = null;
            }
            result["pci_bus_id"] = pci;
            hardware.Update();
            foreach (ISensor sensor in hardware.Sensors)
            {
                // Main GPU temperature only. Hot spot and memory temperatures
                // can have finer precision, but are different measurements.
                if (sensor.SensorType != SensorType.Temperature || sensor.Name != "GPU Core") continue;
                result["source"] = sensor.Name;
                if (sensor.Value.HasValue && !float.IsNaN(sensor.Value.Value) && !float.IsInfinity(sensor.Value.Value))
                {
                    result["temperature_c"] = sensor.Value.Value;
                    return result;
                }
            }
            result["reason"] = "GPU 未提供有效的 GPU Core 当前温度";
        }
        catch (Exception error)
        {
            result["reason"] = (string)Program.Failure(error)["cpu_temperature_reason"];
        }
        return result;
    }

    private static void UpdateCpu(IHardware hardware, List<SensorReading> readings, List<SensorReading> power)
    {
        hardware.Update();
        foreach (ISensor sensor in hardware.Sensors)
        {
            if (!sensor.Value.HasValue) continue;
            float value = sensor.Value.Value;
            if (float.IsNaN(value) || float.IsInfinity(value)) continue;
            if (sensor.SensorType == SensorType.Temperature)
                readings.Add(new SensorReading { Cpu = hardware.Name, Name = sensor.Name, Value = value });
            else if (sensor.SensorType == SensorType.Power && value >= 0 && (sensor.Name == "CPU Package" || sensor.Name == "Package"))
                power.Add(new SensorReading { Cpu = hardware.Name, Name = sensor.Name, Value = value });
        }
        foreach (IHardware child in hardware.SubHardware) UpdateCpu(child, readings, power);
    }

    public void Dispose()
    {
        Computer previous = computer;
        computer = null;
        if (previous != null) previous.Close();
    }

    private sealed class SensorReading
    {
        internal string Cpu;
        internal string Name;
        internal float Value;
    }
}

internal static class Native
{
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern SafeFileHandle CreateFile(string name, uint access, uint share,
        IntPtr security, uint creation, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern bool GetNamedPipeClientProcessId(SafePipeHandle pipe, out uint pid);
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool OpenProcessToken(IntPtr process, uint access, out SafeFileHandle token);
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool GetTokenInformation(SafeFileHandle token, int informationClass,
        IntPtr information, int size, out int needed);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetStdHandle(int id);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool WriteFile(IntPtr handle, byte[] buffer, uint count, out uint written, IntPtr overlapped);
    [DllImport("cfgmgr32.dll", CharSet = CharSet.Unicode, EntryPoint = "CM_Locate_DevNodeW")]
    private static extern uint LocateDevNode(out uint device, string instanceId, uint flags);
    [DllImport("cfgmgr32.dll", CharSet = CharSet.Unicode, EntryPoint = "CM_Get_DevNode_Registry_PropertyW")]
    private static extern uint GetDevNodeProperty(uint device, uint property, out uint type,
        out uint value, ref uint length, uint flags);

    internal static Dictionary<IHardware, uint> VerifiedNvidiaBuses(IEnumerable<IHardware> hardwareList)
    {
        Dictionary<IHardware, uint> buses = new Dictionary<IHardware, uint>();
        HashSet<uint> seen = new HashSet<uint>(), ambiguous = new HashSet<uint>();
        foreach (IHardware hardware in hardwareList)
        {
            if (hardware.HardwareType != HardwareType.GpuNvidia) continue;
            uint bus;
            if (!NvidiaBus(hardware, out bus)) return new Dictionary<IHardware, uint>();
            buses.Add(hardware, bus);
            if (!seen.Add(bus)) ambiguous.Add(bus);
        }
        Dictionary<IHardware, uint> verified = new Dictionary<IHardware, uint>();
        foreach (KeyValuePair<IHardware, uint> entry in buses)
            if (!ambiguous.Contains(entry.Value)) verified.Add(entry.Key, entry.Value);
        return verified;
    }

    private static bool NvidiaBus(IHardware hardware, out uint bus)
    {
        bus = 0;
        // The pinned 0.9.6 public DeviceId can use an internal NVML-index
        // fallback. Verify it against the very NVAPI physical handle used for
        // this hardware's temperature, not another GPU enumeration's order.
        try
        {
            FieldInfo handle = hardware.GetType().GetField("_handle", BindingFlags.NonPublic | BindingFlags.Instance);
            Type api = typeof(Computer).Assembly.GetType("LibreHardwareMonitor.Interop.NvApi");
            PropertyInfo getBus = api == null ? null : api.GetProperty("NvAPI_GPU_GetBusId", BindingFlags.Public | BindingFlags.Static);
            Delegate query = getBus == null ? null : getBus.GetValue(null, null) as Delegate;
            if (handle == null || query == null) return false;
            object[] arguments = { handle.GetValue(hardware), (uint)0 };
            if (Convert.ToInt32(query.DynamicInvoke(arguments), CultureInfo.InvariantCulture) != 0) return false;
            bus = (uint)arguments[1];
            return bus <= 0xFF;
        }
        catch (Exception) { return false; } // Unknown identity must never become an index guess.
    }

    internal static string PciBusId(string instanceId)
    {
        // GenericGpu.DeviceId is an exact Windows PnP instance ID. NVIDIA's
        // public GPU API exposes no PCI address; never map /gpu-nvidia/N to an
        // NVML index or guess from a GPU name. ConfigMgr properties are numeric
        // and independent of localized "Location Information" text.
        uint device, bus, address;
        if (string.IsNullOrEmpty(instanceId) || !instanceId.StartsWith(@"PCI\", StringComparison.OrdinalIgnoreCase) ||
            LocateDevNode(out device, instanceId, 0) != 0 ||
            !RegistryDword(device, 0x16, out bus) || // CM_DRP_BUSNUMBER
            !RegistryDword(device, 0x1D, out address)) // CM_DRP_ADDRESS
            return null;
        return FormatPciBusId(bus, address);
    }

    internal static string FormatPciBusId(uint bus, uint address)
    {
        uint device = address >> 16, function = address & 0xFFFF;
        if (bus > 0xFF || device > 0x1F || function > 7) return null;
        // These Windows properties have no PCI domain. The caller must match
        // this BDF only when exactly one NVML GPU has that BDF across domains.
        return bus.ToString("x2", CultureInfo.InvariantCulture) + ":" +
            device.ToString("x2", CultureInfo.InvariantCulture) + "." + function.ToString(CultureInfo.InvariantCulture);
    }

    private static bool RegistryDword(uint device, uint property, out uint value)
    {
        uint type, length = 4;
        return GetDevNodeProperty(device, property, out type, out value, ref length, 0) == 0 && type == 4 && length == 4;
    }

    internal static SecurityIdentifier ProcessUserSid(IntPtr process)
    {
        SafeFileHandle token;
        if (!OpenProcessToken(process, 0x8, out token)) throw new Win32Exception(Marshal.GetLastWin32Error());
        using (token)
        {
            int needed;
            GetTokenInformation(token, 1, IntPtr.Zero, 0, out needed); // TokenUser
            if (needed <= 0) throw new Win32Exception(Marshal.GetLastWin32Error());
            IntPtr buffer = Marshal.AllocHGlobal(needed);
            try
            {
                if (!GetTokenInformation(token, 1, buffer, needed, out needed))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                // TOKEN_USER begins with SID_AND_ATTRIBUTES, whose first
                // field is the SID pointer on both x86 and x64.
                return new SecurityIdentifier(Marshal.ReadIntPtr(buffer));
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }
    }

    internal static Dictionary<string, object> CheckPawnIo()
    {
        using (SafeFileHandle handle = CreateFile(@"\\?\GLOBALROOT\Device\PawnIO", 0xC0000000,
            3, IntPtr.Zero, 3, 0, IntPtr.Zero))
        {
            if (!handle.IsInvalid) return null;
            int error = Marshal.GetLastWin32Error();
            if (error == 2 || error == 3)
                return Program.Result("driver_missing", "未找到 PawnIO 驱动；请先安装官方 PawnIO", null, null);
            if (error == 5)
                return Program.Result("permission_required", "PawnIO 驱动要求管理员读取权限；请授权硬件温度读取器", null, null);
            return Program.Result("error", "PawnIO 驱动设备无法读取（Windows 错误 " + error + "）", null, null);
        }
    }

    internal static void WriteStdout(string value)
    {
        IntPtr output = GetStdHandle(-11);
        if (output == IntPtr.Zero || output == new IntPtr(-1)) return;
        byte[] data = new UTF8Encoding(false).GetBytes(value);
        uint written;
        WriteFile(output, data, (uint)data.Length, out written, IntPtr.Zero);
    }
}

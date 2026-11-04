"""Local inference backends; importing this package never starts an engine."""
from .adapters import BackendAdapter, get_adapters, import_configuration, import_runtime
from .installers import install_backend, uninstall_runtime
from .telemetry import collect_telemetry

__all__ = ["BackendAdapter", "get_adapters", "import_configuration", "import_runtime", "install_backend", "uninstall_runtime", "collect_telemetry"]

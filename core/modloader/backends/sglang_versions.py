"""Fixed, version-specific SGLang installation and CLI contracts."""
from .resolve_packages import SGLANG_CUDA_TILE_SHA256, SGLANG_CUDA_TILE_URL

RECOMMENDED_VERSION = '0.5.23'
LEGACY_VERSION = '0.5.21'
CUDA_COMPONENTS = ('cuda_nvcc', 'cuda_crt', 'cuda_cudart', 'cccl', 'libnvvm', 'cuda_nvrtc')

# PyPI wheel metadata and each tag's python/pyproject.toml declare the dependency
# versions. The managed recipe keeps its own CUDA 13.4.3 JIT toolchain; it does
# not borrow another runtime's CUDA_HOME or update an existing environment.
RECIPES = {
    '0.5.23': {
        'version': '0.5.23', 'repository': 'sgl-project/sglang', 'tag': 'v0.5.23',
        'source_commit': '297e54fd105e974cdf401e5b4d20253a22c16f90',
        'python_requires': '>=3.11',
        'requirements': ('--pre', 'sglang==0.5.23', 'torch==2.14.1', 'transformers==5.19.0',
                         'flashinfer_python[cu13]==0.7.2', 'sglang-kernel==0.4.10'),
        'cuda_release': '13.4.3', 'cuda_components': CUDA_COMPONENTS,
        'approved_wheels': (),
        'cuda_graph_backends': ('full', 'breakable', 'disabled'),
        'paged_experts': True,
        'removed_flags': ('--speculative-skip-dp-mlp-sync',),
    },
    '0.5.21': {
        'version': '0.5.21', 'repository': 'sgl-project/sglang', 'tag': 'v0.5.21',
        'source_commit': 'e00930c5489053f26d86b179cee0d087f846acbb',
        'python_requires': '>=3.10',
        'requirements': ('--pre', 'sglang==0.5.21', 'torch==2.13.0'),
        'cuda_release': '13.4.3', 'cuda_components': CUDA_COMPONENTS,
        'approved_wheels': ({
            'role': 'cuda_tile', 'name': SGLANG_CUDA_TILE_URL.rsplit('/', 1)[1],
            'purpose': 'CUDA Tile 1.6.0rc5 官方 NVIDIA wheel（Python 3.12 / Linux x86_64）',
            'url': SGLANG_CUDA_TILE_URL, 'digest': 'sha256:' + SGLANG_CUDA_TILE_SHA256,
            'package': 'cuda-tile', 'version': '1.6.0rc5',
        },),
        'cuda_graph_backends': ('full', 'breakable', 'tc_piecewise', 'disabled'),
        'paged_experts': False,
        'removed_flags': (),
    },
}


def recipe(version):
    if version not in RECIPES:
        raise ValueError('SGLang 自动安装只支持固定配方：' + '、'.join(RECIPES))
    return RECIPES[version]

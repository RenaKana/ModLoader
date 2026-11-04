"""Versioned SGLang recipes and launch contracts; no destination execution."""
import pytest


def test_recipes_keep_the_legacy_nvidia_tile_wheel_out_of_the_new_recipe():
    from modloader.backends.sglang_versions import RECIPES, RECOMMENDED_VERSION, recipe

    assert RECOMMENDED_VERSION == '0.5.23'
    assert list(RECIPES) == ['0.5.23', '0.5.21']
    old, new = recipe('0.5.21'), recipe('0.5.23')
    assert old['requirements'] == ('--pre', 'sglang==0.5.21', 'torch==2.13.0')
    assert old['python_requires'] == '>=3.10'
    assert old['source_commit'] == 'e00930c5489053f26d86b179cee0d087f846acbb'
    tile, = old['approved_wheels']
    assert tile['role'] == 'cuda_tile' and tile['package'] == 'cuda-tile'
    assert tile['version'] == '1.6.0rc5'
    assert tile['url'].endswith('cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl')
    assert tile['digest'] == 'sha256:b74c20348210d2182cd998a0ecb60c518989a79b28592d72eb8294b38ddb93d7'
    assert new['approved_wheels'] == ()
    assert new['python_requires'] == '>=3.11'
    assert new['source_commit'] == '297e54fd105e974cdf401e5b4d20253a22c16f90'
    assert set(new['requirements']) == {'--pre', 'sglang==0.5.23', 'torch==2.14.1',
                                        'transformers==5.19.0', 'flashinfer_python[cu13]==0.7.2',
                                        'sglang-kernel==0.4.10'}
    assert old['cuda_release'] == new['cuda_release'] == '13.4.3'
    assert old['cuda_components'] == new['cuda_components'] == (
        'cuda_nvcc', 'cuda_crt', 'cuda_cudart', 'cccl', 'libnvvm', 'cuda_nvrtc')


def test_only_known_automatic_recipes_are_accepted():
    from modloader.backends.sglang_versions import recipe

    with pytest.raises(ValueError, match='0.5.23.*0.5.21'):
        recipe('0.5.17')

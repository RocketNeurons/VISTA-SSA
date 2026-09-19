"""Build the three isolated paper environments against Raylib 5.5."""
import os
from pathlib import Path
import sys
from setuptools import Extension, setup
import numpy

ROOT = Path(__file__).parent
RAYLIB = Path(os.environ.get('RAYLIB_HOME', ROOT / '.build_deps/raylib-5.5_linux_amd64'))
if sys.platform != 'linux':
    raise RuntimeError('VISTA native environments currently support Linux x86_64 / WSL2.')
if not (RAYLIB / 'lib/libraylib.a').exists():
    raise RuntimeError('Run: python scripts/fetch_build_dependencies.py (or set RAYLIB_HOME)')

setup(ext_modules=[Extension(
    f'vista.envs.{phase}.binding', [f'vista/envs/{phase}/binding.c'],
    include_dirs=[numpy.get_include(), str(RAYLIB / 'include')],
    extra_objects=[str(RAYLIB / 'lib/libraylib.a')],
    libraries=['m', 'dl', 'pthread'],
    extra_compile_args=['-O2', '-fPIC', '-DPLATFORM_DESKTOP',
                        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
) for phase in ('phase1', 'phase2', 'phase3')])

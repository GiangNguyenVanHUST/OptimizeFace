from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("FaceAnalyzer",  ["FaceAnalyzer.py"]),
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='My Program Name',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)

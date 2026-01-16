from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# Define the extension
ext_modules = [
    Extension(
        name="ntt",        # Python module name (package.module)
        sources=["ntt.pyx"],   # Your Cython file
        language="c++"         # Compile as C++
    )
]

# Setup
setup(
    name="ntt",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"}  # Python 3 syntax
    ),
)

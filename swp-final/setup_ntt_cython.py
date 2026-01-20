from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="ntt_function",
    sources=["ntt_function.pyx"],
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
)
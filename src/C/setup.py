from distutils.core import setup, Extension

module1 = Extension('ctrace', sources = ['ctrace.c'])


setup(name = "FastTrace",
      version = '0.9b',
      description = 'Provides C interface to fast Ray Tracing methods',
      ext_modules = [module1])

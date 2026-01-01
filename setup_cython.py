from setuptools import setup
from Cython.Build import cythonize
import os

# List of files to compile
# We only compile the "secret sauce" (Adaptive Controller & Transformer Core)
# Perturbations can remain open as they are standard logic
modules_to_compile = [
    "core/adaptive_controller.py",
    "core/inversion_transformer.py",
]

# Verify files exist
existing_modules = [f for f in modules_to_compile if os.path.exists(f)]

if not existing_modules:
    print("⚠️ Warning: No core modules found to compile!")
else:
    print(f"🔒 Compiling {len(existing_modules)} modules for IP protection...")

setup(
    name="InversionTransformerCore",
    ext_modules=cythonize(
        existing_modules,
        compiler_directives={
            'language_level': "3",
            'always_allow_keywords': True,
            'boundscheck': False, # Optimization
            'wraparound': False   # Optimization
        },
        annotate=False
    ),
)

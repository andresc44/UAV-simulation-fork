from setuptools import setup

setup(
    name="safe_control_gym",
    version="2.0.0",
    install_requires=[
        "matplotlib",
        "munch",
        "pyyaml",
        "imageio",
        "dict-deep",
        "scikit-optimize",
        "scikit-learn",
        "gymnasium",
        "torch",
        "gpytorch",
        "tensorboard",
        "casadi",
        "pybullet",
        "numpy",
        "cvxpy",
        "pytope",
        "Mosek",
        "termcolor",
        "pytest",
        "pre-commit",
    ],
)

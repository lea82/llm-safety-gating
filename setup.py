from setuptools import setup, find_packages

setup(
    name="llm-safety-gating",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.30.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "safety-gate=cli:main",
        ],
    },
)

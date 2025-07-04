"""
Setup configuration for Credit Risk Modeling package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-risk-modeling",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive machine learning toolkit for credit risk assessment and modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/credit-risk-modeling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "streamlit>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "credit-risk-train=src.cli:train_model",
            "credit-risk-predict=src.cli:predict",
            "credit-risk-evaluate=src.cli:evaluate_model",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/credit-risk-modeling/issues",
        "Source": "https://github.com/yourusername/credit-risk-modeling",
        "Documentation": "https://credit-risk-modeling.readthedocs.io/",
    },
    keywords="credit risk, machine learning, finance, risk assessment, xgboost, lightgbm",
    zip_safe=False,
)
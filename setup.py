from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="faoru",
    version="1.0.0",
    author="FAORU Team",
    author_email="contact@faoru-project.org",
    description="Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/faoru-project/FAORU",
    project_urls={
        "Bug Tracker": "https://github.com/faoru-project/FAORU/issues",
        "Documentation": "https://faoru-project.github.io/FAORU",
        "Source Code": "https://github.com/faoru-project/FAORU",
        "Paper": "https://arxiv.org/abs/2025.xxxxx",
    },
    packages=find_packages(exclude=["tests", "scripts", "tools"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "vis": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "faoru-train=train:main",
            "faoru-eval=validate:main",
        ],
    },
)

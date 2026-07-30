"""NIPS — Network Intrusion Prevention System."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and line.strip() != "-e ."
    ]

setup(
    name="nips",
    version="1.0.0",
    description="Network Intrusion Prevention System — real-time traffic interception and ML-based threat detection",
    author="梓铭",
    author_email="2147514473@qq.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.12",
)

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scikit-pytsk",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/YuqiCui/PyTSK",
    license="MIT",
    author="Yuqi Cui",
    author_email="yqcui@qq.com",
    description="PyTSK provides tools for uses conveniently developing TSK fuzzy systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "scikit-learn"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
from setuptools import setup, find_packages

setup(
    name="StudentPerformancePrediction-End-To-End",
    version="0.1.0",
    author="Susant Kumar",
    author_email="susantkumar.sks96@gmail.com",
    description="End-to-End Student Performance Prediction project with ML and Flask.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/StudentPerformancePrediction-End-To-End",  # optional
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "catboost",
        "xgboost",
        "Flask",
        "from_root"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",   # change if you use a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)


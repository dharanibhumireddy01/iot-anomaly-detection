from setuptools import setup, find_packages

setup(
    name="iot-anomaly-detection",
    version="1.0.0",
    author="Dharani Bhumireddy",
    author_email="dharanibhumireddy.ds@gmail.com",
    description="IoT Sensor Anomaly & Fraud Pattern Detection — ARIMA + Isolation Forest",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    ],
)

from setuptools import setup, find_packages

setup(
    name="glitchminer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        # 添加其他依赖项
    ],
    author="Zihui Wu",
    description="A tool for mining glitch tokens in language models",
    url="https://github.com/yourusername/glitch-miner",
)

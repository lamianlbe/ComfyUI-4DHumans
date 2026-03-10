from setuptools import setup, find_packages

setup(
    name="smplest-x",
    version="1.0.0",
    description="SMPLest-X: whole-body mesh recovery (body, hands, face) from a single image",
    license="S-Lab License 1.0",
    packages=find_packages(exclude=["datasets", "datasets.*",
                                     "humandata_prep", "humandata_prep.*",
                                     "scripts", "scripts.*"]),
    package_data={
        "configs": ["*.py"],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.1",
        "smplx==0.1.28",
        "torch",
        "torchvision",
        "tqdm>=4.67.1",
        "opencv-python>=4.11.0.86",
        "trimesh>=4.6.2",
        "matplotlib>=3.7.5",
        "json_tricks>=3.17.3",
        "einops>=0.8.1",
        "timm>=1.0.14",
        "ultralytics>=8.3.75",
        "pyopengl",
    ],
)

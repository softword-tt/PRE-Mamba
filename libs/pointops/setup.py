import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

os.environ["CC"] = "gcc-6"
os.environ["CXX"]= "g++-6"

# 自动检测CUDA路径和版本
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if not os.path.exists(cuda_home):
    raise RuntimeError(f"CUDA not found at {cuda_home}. Please set the CUDA_HOME environment variable correctly.")

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

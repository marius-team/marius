import os
import platform
import re
import sys

import yaml
from importlib_metadata import version


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def get_os_info():
    os_info = {"platform": platform.platform()}
    return os_info


def get_cpu_info():
    cpu_info = {"num_cpus": "N/A", "total_memory": "N/A"}

    try:
        import psutil
    except ImportError:
        return cpu_info

    cpu_info["num_cpus"] = psutil.cpu_count()
    cpu_info["total_memory"] = "{}GB".format(psutil.virtual_memory().total >> 30)
    return cpu_info


def get_gpu_info():
    gpu_info = "N/A"

    try:
        import GPUtil
    except RuntimeError:
        return gpu_info

    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({"name": gpu.name, "memory": "{}GB".format(int(gpu.memoryTotal) >> 10)})

    return gpu_info


def get_python_info():
    py_deps = [
        "numpy",
        "pandas",
        "tox",
        "pytest",
        "torch",
        "omegaconf",
        "pyspark",
        "pip",
    ]
    py_deps_version = {}
    for dep in py_deps:
        try:
            imported_dep = __import__(dep)
            py_deps_version[dep + "_version"] = imported_dep.__version__
        except ModuleNotFoundError:
            py_deps_version[dep + "_version"] = "N/A"

    pytorch_info = {
        "version": sys.version,
        "deps": py_deps_version,
    }
    return pytorch_info


def get_cuda_info():
    cuda_info = {"version": "N/A"}
    try:
        import torch
    except ImportError:
        return cuda_info

    if torch.has_cuda:
        cuda_info["version"] = torch.version.cuda
    return cuda_info


def get_openmp_info():
    openmp_info = {"version": "N/A"}

    openmp_output = os.popen("echo | cpp -fopenmp -dM | grep -i open").read()
    version_pattern = re.compile(r"#define\s_OPENMP.*\s([0-9.]+)")
    openmp_version = re.search(version_pattern, openmp_output)
    if openmp_version is not None:
        openmp_info["version"] = openmp_version.group(1)

    return openmp_info


def get_pytorch_info():
    pytorch_info = {"version": "N/A", "install_path": "N/A"}
    try:
        import torch
    except ImportError:
        return pytorch_info

    pytorch_info["version"] = torch.__version__
    pytorch_info["install_path"] = os.path.dirname(torch.__file__)

    return pytorch_info


def get_marius_info():
    marius_info = {"version": "N/A", "install_path": "N/A", "bindings_installed": False}
    try:
        import marius

        marius_info["install_path"] = marius.__path__[0]
        marius_info["version"] = version("marius")
    except ImportError:
        return marius_info

    try:
        import marius.nn

        marius_info["bindings_installed"] = True
    except ImportError:
        pass

    return marius_info


def get_pybind_info():
    pybind_info = {"PYBIND11_COMPILER_TYPE": "N/A", "PYBIND11_STDLIB": "N/A", "PYBIND11_BUILD_ABI": "N/A"}

    try:
        import torch
    except ImportError:
        return pybind_info

    pybind_info["PYBIND11_COMPILER_TYPE"] = torch._C._PYBIND11_COMPILER_TYPE
    pybind_info["PYBIND11_STDLIB"] = torch._C._PYBIND11_STDLIB
    pybind_info["PYBIND11_BUILD_ABI"] = torch._C._PYBIND11_BUILD_ABI
    return pybind_info


def get_cmake_info():
    cmake_info = {"version": "N/A"}

    cmake_output = os.popen("cmake --version").read().split("\n")[0]
    version_pattern = re.compile(r".*\s([0-9.]+)")
    cmake_version = re.search(version_pattern, cmake_output)
    if cmake_version is not None:
        cmake_info["version"] = cmake_version.group(1)

    return cmake_info


def main():
    env_info = {}

    env_info["operating_system"] = get_os_info()
    env_info["cpu_info"] = get_cpu_info()
    env_info["gpu_info"] = get_gpu_info()
    env_info["python"] = get_python_info()
    env_info["pytorch"] = get_pytorch_info()
    env_info["cuda"] = get_cuda_info()
    env_info["marius"] = get_marius_info()
    env_info["pybind"] = get_pybind_info()
    env_info["cmake"] = get_cmake_info()
    env_info["openmp"] = get_openmp_info()

    print(yaml.dump(env_info, Dumper=MyDumper, default_flow_style=False))


if __name__ == "__main__":
    main()

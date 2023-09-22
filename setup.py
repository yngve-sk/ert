import os
import subprocess
import sys
from pathlib import Path

from setuptools import Command, find_packages
from setuptools.command.egg_info import egg_info
from skbuild import setup

# list of pair of .proto file and out directory
PROTOBUF_FILES = [("src/_ert_com_protocol/_schema.proto", "src/_ert_com_protocol")]


def compile_protocol_buffers():
    for proto, out_dir in PROTOBUF_FILES:
        proto_path = Path(proto).parent
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I",
                proto_path,
                f"--python_out={out_dir}",
                proto,
            ],
            check=True,
        )


class EggInfo(egg_info):
    """scikit-build uses the metadata of ert to determine what to include when building
    the project. This determination results in files being copied to a special build
    folder. If ert wants to compile e.g. protobuf files and have those included in the
    distribution, those files needs to be a part of the distribution metadata, i.e. it
    needs to happen in egg_info so that the compiled files are copied to the build
    folder."""

    def run(self):
        compile_protocol_buffers()
        egg_info.run(self)  # old style class, no super()


class CompileProtocolBuffers(Command):
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self):
        compile_protocol_buffers()


# Corporate networks tend to be behind a proxy server with their own non-public
# SSL certificates. Conan keeps its own certificates,
# whose path we can override
if "CONAN_CACERT_PATH" not in os.environ:
    # Look for a RHEL-compatible system-wide file
    for file_ in ("/etc/pki/tls/cert.pem",):
        if not os.path.isfile(file_):
            continue
        os.environ["CONAN_CACERT_PATH"] = file_
        break


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", "..", path, filename))
    return paths


args = dict(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "ert": package_files("src/ert/gui/resources/")
        + package_files("src/ert/shared/share/")
        + ["logging/logger.conf", "logging/storage_log.conf"]
        + [
            "job_queue/qstat_proxy.sh",
        ],
    },
    cmake_args=[
        "-DBUILD_TESTS=OFF",
        # we can safely pass OSX_DEPLOYMENT_TARGET as it's ignored on
        # everything not OS X. We depend on C++17, which makes our minimum
        # supported OS X release 10.15
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15",
    ],
    cmake_source_dir="src/clib/",
    cmdclass={
        "egg_info": EggInfo,
        "compile_protocol_buffers": CompileProtocolBuffers,
    },
)

setup(**args)

# workaround for https://github.com/scikit-build/scikit-build/issues/546 :
# This increases time taken to run `pip install -e .` somewhat until we
# have only one top level package at which point we can use the workaround
# in the issue
if sys.argv[1] == "develop":
    from setuptools import setup as setuptools_setup

    del args["cmake_args"]
    del args["cmake_source_dir"]
    setuptools_setup(**args)

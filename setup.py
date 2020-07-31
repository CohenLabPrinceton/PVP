from setuptools import setup, find_packages
import os
import subprocess

depend_links = []

# detect if on raspberry pi, and
# set location to wheel if we are
IS_RASPI = False
ret = subprocess.call(["grep", "-q", "BCM", "/proc/cpuinfo"])
if ret == 0:
    IS_RASPI = True
    os.system("sudo +x INSTALL")
    os.system("sudo ./INSTALL")

    # keeping this around for proper packaging later
    # get wheel name
    #external_files = os.listdir(os.path.join(os.getcwd(), "external"))
    #pyside_wheel = [whl for whl in external_files if whl.endswith(".whl") and whl.startswith("PySide2")][0]
    #depend_links.append(os.path.join(os.getcwd(), "external", pyside_wheel))


setup(
    name="pvp",
    author="vent team",
    author_email="vent@vents.com",
    description="some description of how we made a ventilator",
    keywords="vents ventilators etc",
    url="https://ventilator.readthedocs.io",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        # "PySide2",
        # "pyqtgraph>=0.11.0rc0",
        # "pytest-qt",
        # "pytest-timeout",
        "scipy",
        "pigpio",
        "tables"
    ],
    dependency_links=depend_links,
    python_requires="==3.7.*"
)

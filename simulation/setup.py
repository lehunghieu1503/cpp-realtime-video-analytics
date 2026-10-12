from setuptools import setup

package_name = "video_analytics_simulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hieu Hung Le",
    maintainer_email="hewlee789@gmail.com",
    description="Drives a CARLA vehicle from the detections and lane of a video-analytics device",
    license="MIT",
    entry_points={
        "console_scripts": [
            "simulator = video_analytics_simulation.simulator:main",
            "controller = video_analytics_simulation.controller:main",
        ],
    },
)

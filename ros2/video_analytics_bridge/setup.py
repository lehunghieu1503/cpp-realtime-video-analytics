from setuptools import setup

package_name = "video_analytics_bridge"

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
    description="Publishes the detections and events of a video-analytics device as ROS 2 topics",
    license="MIT",
    entry_points={
        "console_scripts": ["bridge = video_analytics_bridge.bridge:main"],
    },
)

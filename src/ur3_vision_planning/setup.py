from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur3_vision_planning'
package_dir = os.path.dirname(os.path.realpath(__file__))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yY][aA][mM][lL]*'))),
        (os.path.join('share', package_name, 'urdf'),
            glob(os.path.join('urdf', '*.xacro'))),
        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join('worlds', '*.xacro'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Le Hong Quang',
    maintainer_email='student@todo.todo',
    description='LLM control package for UR3 robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_precheck = ur3_vision_planning.motion_precheck:main',
            'natural_language_planner = ur3_vision_planning.natural_language_planner:main',
            'environment_manager_test = ur3_vision_planning.environment_manager:main',
            'camera_perception = ur3_vision_planning.camera_perception:main',
            'llm_planner = ur3_vision_planning.llm_planner:main',
            'skill_executor = ur3_vision_planning.skill_executor:main',
            'scene_publisher = ur3_vision_planning.scene_publisher:main',
        ],
    },
)

from setuptools import setup

package_name = 'udp_comm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='f1tenth udp_comm lab',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'udp_comm_node = udp_comm.udp_comm_node:main',
            'pp_node = udp_comm.pp_node:main',
        ],
    },
)

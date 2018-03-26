"""Setup script for object_detection."""
import logging
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install

class CustomCommands(install):
    """A setuptools Command class able to run arbitrary commands."""
    def RunCustomCommand(self, command_list):
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        logging.info('Log command output: %s', stdout_data)
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' %
                            (command_list, p.returncode))
       
    def run(self):
        self.RunCustomCommand(['apt-get', 'update'])
        self.RunCustomCommand(
            ['apt-get', 'install', '-y', 'python-tk'])
        install.run(self)
REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1']
setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
 cmdclass={
        'install': CustomCommands,
    }
)
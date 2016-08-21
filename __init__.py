#####################################################################
#                                                                   #
# __init__.py                                                       #
#                                                                   #
# Copyright 2013, Monash University                                 #
#                                                                   #
# This file is part of the labscript suite (see                     #
# http://labscriptsuite.org) and is licensed under the Simplified   #
# BSD License. See the license.txt file in the root of the project  #
# for the full license.                                             #
#                                                                   #
#####################################################################

__version__ = '2.3.0'


class VersionException(Exception):
    pass

    
def check_version(module_name, at_least, less_than=None, version=None):

    from distutils.version import LooseVersion

    if version is None:
        version = __import__(module_name).__version__
    # Require at_least version or later
    if less_than is None:
        at_least_version, installed_version = [LooseVersion(v) for v in [at_least, version]]
        if not at_least_version <= installed_version:
            raise VersionException(
                '{module_name} {version} found. {at_least} <= {module_name} required.'.format(**locals()))
    # Require at_least version but before less_than
    else:
        at_least_version, less_than_version, installed_version = [LooseVersion(v) for v in [at_least, less_than, version]]
        if not at_least_version <= installed_version < less_than_version:
            raise VersionException(
                '{module_name} {version} found. {at_least} <= {module_name} < {less_than} required.'.format(**locals()))

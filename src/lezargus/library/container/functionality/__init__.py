"""Functionality relating to the container classes are put in this module.

We put extra functionality that is not entirely related to the container
classes here. Although it is very common to have the containers have their
own interfaces to the functionality provided by this module. Moreover,
by offloading a lot of the logic into this module, the container class files
can be shorter and cleaner.

It is fine to have all of the functions in the same namespace. We separate
the files so it is easier to navigate however.
"""

# All of the container functions exist in all the same namespace, but we
# break it into different files for organization.
# ruff: noqa: F403

from lezargus.library.container.functionality.broadcast import *
from lezargus.library.container.functionality.convolution import *
from lezargus.library.container.functionality.transform import *

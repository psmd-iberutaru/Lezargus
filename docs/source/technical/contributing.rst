=================
Contributor Guide
=================

First, thank you for considering contributing to this project.

In this guide, we go over the basics of setting up a typical development 
environment for Lezargus, a few of the conventions, development procedures, 
and tooling available. If you find that something may be confusing, outdated, 
or missing in this guide, please submit an issue or let a developer know. 


Setup
=====

Prerequisites
-------------

This guide (and development in general) requires the following:

1. Install Git for your operating system. 
   See `<https://git-scm.com/downloads/>`_
2. Install Python 3.10+. The most recent version is recommended. 
   See `<https://www.python.org/downloads/>`_
3. Create (or have) a GitHub account. 
   See `<https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github>`_
4. (Optional) Install a full install of (La)TeX, usually a TeXLive install is 
   good enough. We only recommend this if you are the main developer of this 
   package and therefore need to build the LaTeX documentation. 
   See `<https://tug.org/texlive/>`_.

After Python has been installed, you can install other Python packages using 
``pip`` via:

.. code-block:: bash

    pip install <package>

Please ensure that you have the following minimum packages for development: 

- ``pip``

    - This should be installed by default. As ``pip`` is used to install all 
      other packages its installation is more involved.
    - Try: Using ``ensurepip`` you can bootstrap the installation: 
      see `<https://docs.python.org/3/library/ensurepip.html>`_
    - Try: On Linux, you may also try and install ``pip`` via your package 
      manager. 

- ``hatch``
- ``wheel``
- ``ruff``
- ``black``

Other dependencies (for both Lezargus itself and the development thereof 
will be installed on-the-fly).


Clone/Fork Repository
---------------------

To contribute, you need to either clone or fork the Lezargus repository. If 
you have been granted write access to the repository you only need to clone 
the repository, otherwise you will need to fork the repository first. You may 
contact the developers to gain write access to the main Lezargus repository; 
we discourage but do not disallow opening an issue for such a request. 

For more information on forking a repository for the sake of contributing to 
it, please see the 
`GitHub documentation <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_ 
on the matter. A great resource on Git can be found in 
`GitHub's Training Kit <https://githubtraining.github.io/training-manual>`_; 
a full tutorial on Git(Hub) itself is a little beyond this guide.

.. note::
    If you are a developer staff member with IRTF or similarly related, 
    by convention, we will/must give you write access to the main 
    Lezargus repository. Please contact the main developers for such 
    permission.

There are many ways to clone a repository, but the easiest way is via HTTPS:

.. code-block:: bash

   git clone https://github.com/psmd-iberutaru/Lezargus.git

This will clone the Lezargus repository to a directory called ``./Lezargus/``. 
We will refer to files from this "root" repository ``Lezargus/`` for 
convenience. Please enter the repository (i.e. ``cd Lezargus``). Moreover, 
all commands (``git``, ``hatch``, and other shell commands) should be run 
with ``Lezargus/`` being the present working directory.

Developing
==========

To properly develop, you will need to create a Git branch for your feature, 
implement the feature, ensure it passes testing, linting, and formatting 
checks, then submit a pull request to have your feature reviewed before adding. 

Create a Branch
---------------

The following command creates a branch with a given name. We suggest simple 
names relevant to the feature; use hyphens "-" as the word separator if needed.

.. code-block:: bash

    git branch <name>

Then enter the created branch to begin developing your feature:

.. code-block:: bash

    git switch <name>

From within this branch, you can start developing your feature.

.. _technical-contributing-implement-feature:

Implement Feature
-----------------

It is beyond this guide to tell you how to implement the feature you are 
adding to Lezargus. However, please keep the following in mind.

- Lezargus has a few coding and development conventions which we ask that you 
  follow.

    - See :ref:`technical-conventions`.

- Please utilize the Lezargus library ``lezargus.library`` as much as possible. 
  We list below some helpful modules which you will most likely need.

    - Logging and exception handling: ``lezargus.library.logging``, usually 
      aliased to ``logging``.
    - Math and uncertainty propagation: ``lezargus.library.math``.
    - Configuration: ``lezargus.library.config`` and the configuration file 
      ``Lezargus/src/lezargus/config.py``.
    - Extra data files: ``lezargus.library.data``.
    - Container structures for spectra, images, and cubes: ``lezargus.library.container``.
    - Type hinting: ``lezargus.library.hint`` usually aliased to ``hint``.

- Formatting, linting, and basic code cleanliness conventions are all handled 
  by automatic tools (described later). Therefore, there is no need to stress 
  about it, but consider the following:

    - Python source files are better as one word.
    - Please have proper docstrings for all your files, functions, and classes. 
      We require this but sometimes the automatic tools don't catch every case. 
      We (generally) follow the 
      `Numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ 
      style guide. (Note if you find a violation of the guide in our 
      documentation, feel free to fix it if it is minor or reach out and 
      submit an issue.)
    - Please use Python type hint decorations in your code.
    - If you are adding a completely new feature, please implement tests where 
      appropriate. This can nevertheless be deferred for a very important feature/bug.

We recommend implementing small portions of your feature as described in 
:ref:`technical-contributing-implement-feature` and committing it, checking it, 
and testing it as described :ref:`technical-contributing-check-feature`. 
Iterating like this encourages smaller changes which are easier to review. 
We also suggest that you add your changes and commit your changes often so 
you can take full advantage of version control:

.. code-block:: bash

    git add <files/pattern>
    git commit -m "<commit message>"

You can also push your commits to your created branch on the remote repository 
to ensure you do not lose your work:

.. code-block:: bash

    git push

It may be helpful to also merge your feature branch with an up-to-date master 
branch...

.. code-block:: bash

    git pull

... or another feature branch.

.. code-block:: bash

    git merge <other-branch>


If you have any questions with the development process, please feel free to 
contact your fellow developers.


.. _technical-contributing-check-feature:

Check Feature
-------------

To make sure the added feature works as intended, we suggest going through the 
development checks before opening a pull request.

1. The biggest check is to make sure the code builds into a package. You can 
   attempt to build the package using ``hatch``. We need to advance the version 
   to a new development version each time you build. The path to the wheel file 
   will be spat out after the build command. (We provide the convenience script 
   ``Lezargus/rebuild.ps1`` which does this as well.)

.. code-block:: bash

    hatch version dev
    hatch build
    pip install <path/to/wheel>

2. We follow the Python `Black code style <https://black.readthedocs.io/en/stable/>`_. 
   You can auto format your code (and all other code in the repository) 
   using the hatch job:

.. code-block:: bash

    hatch run format

3. Python linting is done by another job: ``check``. We use 
   `Ruff <https://docs.astral.sh/ruff/>`_ and 
   `Pylint <https://pylint.readthedocs.io/en/latest/>`_ for linting. When you 
   get linting errors, consult their documentation for more information. Ruff 
   can sometimes fix some of the linting problems that it catches, to utilize 
   this functionality use the ``lintfix`` job instead (however, this skips 
   Pylint). 

.. code-block:: bash

    hatch run check    (or checkfix)

4. You can test your code (or all of the repository code in general) against 
   our test suite using the ``test`` job. This leverages 
   `pytest <https://docs.pytest.org/en/>`_. Note that this only covers areas 
   where the test cases have been built; for the coverage see the next part. 

.. code-block:: bash

    hatch run test

5. Code coverage (for test cases) is checked and generated by the hatch job: ``cover``. 
   This job should not fail per-say, but it does give you information about 
   which parts of your feature code are checked in the currently implemented 
   test cases. (Note, you don't need 100% coverage, just something good 
   enough.)

.. code-block:: bash

    hatch run cover

6. Manually test the functions to make sure they work as intended. We 
   recommend adding the suite you use to the provided tests in general, but it 
   is still helpful for you to test your feature manually just in case. 

7. (Optional) You can also build the documentation using the hatch job ``docs``, 
   though this is generally not advised until the end. For more information on 
   how to document your feature and build the documentation, see 
   :ref:`technical-contributing-documenting`.



.. _technical-contributing-submit-pull-request:

Submit Pull Request
-------------------

To have your feature added to the master branch of the Lezargus repository, 
you will need to 
`open a pull request <https://github.com/psmd-iberutaru/Lezargus/pulls>`_ 
on the Lezargus GitHub page. A lot of information is present in the 
`GitHub pull requests documentation <https://docs.github.com/en/pull-requests>`_. 
We summarize it here.

You need to push your local changes to the remote branch on the remote 
repository, adding the files and creating one last commit:

.. code-block:: bash

    git add <file/pattern>
    git commit -m "Message"
    git push

Once pushed, on the Lezargus GitHub page, create a new pull request. Select 
the "base" branch as ``master`` (or your specific upstream branch of your 
feature) and the "compare" branch to be your feature branch. Then create the 
pull request and describe your changes using the template (if available). 

You typically will not be able to merge the changes on your own until the 
automatic checks are passed (like those found in 
:ref:`technical-contributing-check-feature`) and your changes have been 
reviewed. Once the checks are passed and the featured reviewed, it can be 
merged into the master branch. Congratulations!

.. _technical-contributing-documenting:

Documenting
===========

It is important to document your changes. As briefly touched on, we use 
Python docstrings to document our Python files, functions, modules, and 
classes. However, docstrings only document the code. We have three manuals 
for the three different types of people who interact with the Lezargus package. 
None of the manuals should really duplicate the information, they should 
instead cross reference each other.

Please document all your contributions in all three manuals where appropriate:

- For changes to the non-development user experience, please add your changes 
  to the User Manual: ``Lezargus/docs/source/user/``.
- For changes applicable to developers or advanced users of Lezargus, please 
  add your changes to the Technical Manual: ``Lezargus/docs/source/technical/``.
- All Python docstring documentation is automatically built and placed in the 
  Code Manual: ``Lezargus/docs/source/code/``. The generated files should not be 
  edited manually.

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to build our 
documentation. Apart from a few Markdown files relevant to GitHub 
repository documentation, we use 
`reStructuredText <https://docutils.sourceforge.io/rst.html>`_ to markup the 
documentation. Sphinx provides a good 
`primer to reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. 
We suggest looking at that and some existing documentation files to get a 
handle on the reStructuredText markup. 

The documentation files are built as described earlier, explained in more 
detail here. You can build the HTML version of the documentation files using 
the hatch job:

.. code-block:: bash

    hatch run docs

If you want to also build the LaTeX version of the documentation, you will 
need to have an installation of LaTeX and you will also need to uncomment out 
the LaTeX build line of the job in the ``Lezargus/pyproject.toml`` file. 
Normal developers do not typically need to worry about this as the main 
developer, from time to time, build the LaTeX documentation.

Please make sure that your documentation properly builds without any errors or 
warnings before submitting it via a pull request.

.. warning :: 
    
    We strongly advise against combining your feature changes with 
    documentation changes in the same commit. We suggest doing it at the very 
    end when opening a pull request or waiting until a new numbered release. 
    Running the documentation build job changes a lot of files and clutters the 
    Git history. You may build the documentation on your own machine to ensure 
    it builds properly and then reset it (or, at the very least, delete the 
    ``Lezargus/docs/build/`` and ``Lezargus/docs/source/code/`` directories 
    locally).  
    
    In general, building documentation files via the hatch job should be 
    after the feature has been committed and pushed to remote per 
    :ref:`technical-contributing-submit-pull-request`.

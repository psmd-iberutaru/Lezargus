# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# For the Python code itself.
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Lezargus"
copyright = "2023, Sparrow"
author = "Sparrow"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]
# Sphinx Napoleon autodoc config.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# We do not include the Python inter-sphinx mapping because native Python
# types should already be known and because the footnote links in the LaTeX
# file, get super out of hand.
intersphinx_mapping = {
    # "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}


templates_path = ["_templates"]
exclude_patterns = []

# Allow for figure numbers.
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://github.com/executablebooks/sphinx-book-theme

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 5,
}
# html_favicon = "./assets/pyukumuku_favicon.png"

html_static_path = ["_static"]


# -- Options for LaTeX output -------------------------------------------------
latex_engine = "lualatex"
latex_show_urls = "footnote"

latex_elements = {
    # Allow for nesting.
    "preamble": r"\usepackage{enumitem}",
    # A little bigger font to be more readable.
    "pointsize": "11pt",
    # Single column index.
    "makeindex": "\\usepackage[columns=1]{idxlayout}\\makeindex",
    # Strict figure placement so things do not get lost.
    "figure_align": "H",
}

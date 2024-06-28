# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SNNCutoff'
copyright = '2024-present, Dengyu Wu and Minghong Xu'
author = 'Dengyu Wu and Minghong Xu'
release = '0.0.0'

master_doc = 'index'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'SNNCutoff Documentation'
html_static_path = ['_static']
html_logo = "_static/light_mode.png"
html_favicon = "_static/snncutoff_favicon.png"

html_theme_options = {
    "repository_url": "https://github.com/Dengyu-Wu/snncutoff",
    "use_repository_button": True,
}

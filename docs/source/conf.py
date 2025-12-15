# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Energy-Exergy Analysis Engine'
copyright = '2025, betlab'
author = 'Habin Jo, Wonjun Choi'
release = '0.1.0'

# Force rebuild to update copyright

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_shibuya'
html_static_path = ['_static']
html_baseurl = 'https://bet-lab.github.io/enex_analysis_engine/'

# -- Options for sphinx-shibuya-theme ----------------------------------------

html_theme_options = {
    "github_url": "https://github.com/BET-lab/enex_analysis_engine",
    "nav_links": [
        {
            "title": "GitHub",
            "url": "https://github.com/BET-lab/enex_analysis_engine",
            "icon": "fa-brands fa-github",
        },
    ],
    "footer_links": [
        {
            "title": "GitHub",
            "url": "https://github.com/BET-lab/enex_analysis_engine",
            "icon": "fa-brands fa-github",
        },
    ],
    "page_sidebar_items": ["page-toc", "edit-this-page"],
}

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_mock_imports = ['dartwork_mpl', 'dartwork-mpl']

# -- Options for napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for type hints --------------------------------------------------

typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True


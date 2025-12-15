# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
# Add src directory to path for importing enex_analysis
src_path = os.path.abspath('../../src')
sys.path.insert(0, src_path)

# Try to import the package - if it fails, we'll mock it
try:
    import enex_analysis
    # Create alias for enex_analysis_engine
    import sys as _sys
    _sys.modules['enex_analysis_engine'] = enex_analysis
    _sys.modules['enex_analysis_engine.calc_util'] = enex_analysis.calc_util
    _sys.modules['enex_analysis_engine.enex_engine'] = enex_analysis.enex_engine
except ImportError:
    pass

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Energy-Exergy Analysis Engine'
copyright = '2025, betlab'
author = 'Habin Jo, Wonjun Choi'
release = '0.1.0'

# Force rebuild to update copyright and apply Shibuya theme

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',      # Copy button for code blocks
    'sphinx_design',         # Grid layouts and cards
    'myst_parser',           # Markdown support
    'sphinx_click',         # CLI documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store']

# Suppress warnings for myst cross-references (anchor links work in HTML output)
suppress_warnings = ['myst.xref_missing']

# -- Options for myst-parser ----------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_baseurl = 'https://bet-lab.github.io/enex_analysis_engine/'

# -- Options for sphinx-shibuya-theme ----------------------------------------

html_theme_options = {
    "github_url": "https://github.com/BET-lab/enex_analysis_engine",
    "accent_color": "indigo",
    "nav_links": [
        {
            "title": "Installation",
            "url": "installation",
        },
        {
            "title": "User Guide",
            "url": "user_guide",
        },
        {
            "title": "Examples",
            "url": "examples",
        },
        {
            "title": "API Reference",
            "url": "api",
        },
        {
            "title": "Theory",
            "url": "theory",
        },
        {
            "title": "GitHub",
            "url": "https://github.com/BET-lab/enex_analysis_engine",
            "icon": "fa-brands fa-github",
        },
    ],
    # 로고가 있으면 추가
    # "light_logo": "_static/logo-light.svg",
    # "dark_logo": "_static/logo-dark.svg",
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


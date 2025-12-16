# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add src directory to path for importing enex_analysis
src_path = os.path.abspath("../../src")
sys.path.insert(0, src_path)

# Try to import the package - if it fails, we'll mock it
try:
    import enex_analysis

    # Create alias for enex_analysis_engine
    import sys as _sys

    _sys.modules["enex_analysis_engine"] = enex_analysis
    _sys.modules["enex_analysis_engine.calc_util"] = enex_analysis.calc_util
    _sys.modules["enex_analysis_engine.enex_engine"] = enex_analysis.enex_engine
except ImportError:
    pass

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Energy-Exergy Analysis Engine"
copyright = "2025, betlab"
author = "Habin Jo, Wonjun Choi"
release = "0.1.0"

# Force rebuild to update copyright and apply Shibuya theme

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Grid layouts and cards
    "myst_parser",  # Markdown support
    "sphinx_click",  # CLI documentation
    "sphinx.ext.githubpages",  # Fix for GitHub Pages ignoring _static
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

# Suppress warnings for myst cross-references (anchor links work in HTML output)
suppress_warnings = ["myst.xref_missing"]

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

html_theme = "furo"
html_title = "Energy-Exergy Analysis Engine"
html_static_path = ["_static"]
html_baseurl = "https://bet-lab.github.io/enex_analysis_engine/"

# Custom CSS
html_css_files = [
    "css/custom.css",
]

# -- Options for Furo theme --------------------------------------------------
# https://pradyunsg.me/furo/customisation/

html_theme_options = {
    # Color scheme
    "light_css_variables": {
        "color-brand-primary": "#4051b5",  # Indigo
        "color-brand-content": "#4051b5",
        "color-sidebar-background": "#f8f9fa",
        "color-sidebar-background-border": "#e9ecef",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7c88cc",  # Lighter indigo for dark mode
        "color-brand-content": "#7c88cc",
    },
    # Sidebar
    "sidebar_hide_name": False,
    # Footer
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bet-lab/enex_analysis_engine",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    # Source code repository
    "source_repository": "https://github.com/bet-lab/enex_analysis_engine",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = ["dartwork_mpl", "dartwork-mpl"]

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

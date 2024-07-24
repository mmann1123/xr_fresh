# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = "xr_fresh"
copyright = "2024, Michael Mann"
author = "Michael Mann"

# The full version, including alpha/beta/rc tags
release = "0.2.0"


# -- General configuration ---------------------------------------------------
# html_context = {
#     "css_files": [
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/pygments.css",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/css/theme.css",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/graphviz.css",
#     ],
#     "script_files": [
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/jquery.js",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/_sphinx_javascript_frameworks_compat.js",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/documentation_options.js",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/doctools.js",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/sphinx_highlight.js",
#         "https://mmann1123.github.io/xr_fresh/build/html/_static/js/theme.js",
#     ],
# }

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"  #'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_baseurl = "https://mmann1123.github.io/xr_fresh"


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {'page_width': '80%',
#                      'fixed_sidebar': True,
#                      'logo': 'logo.png',
#                      'logo_name': False,
#                      'github_banner': False,
#                     'github_button': True,
#                     'github_user': 'mmann1123',
#                     'github_repo': 'xr_fresh',
#                     'anchor': '#d37a7a',
#                    'anchor_hover_bg': '#d37a7a',
#                    'anchor_hover_fg': '#d37a7a'}


html_theme_options = {
    "canonical_url": "https://mmann1123.github.io/xr_fresh/build/html/",
    "canonical_url": "",
    "analytics_id": "UA-XXXXXXX-1",  #  Provided by Google in your dashboard
    "logo_only": False,
    "logo": "logo.png",
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "github_button": True,
}

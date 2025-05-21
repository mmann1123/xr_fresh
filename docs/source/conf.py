# build with sphinx-build -b html source .
# maybe


import os
import sys

# Adjust path to include the xr_fresh directory
sys.path.insert(0, os.path.abspath("../../xr_fresh"))
# -- Project information -----------------------------------------------------
import requests

try:
    resp = requests.get(
        "https://api.github.com/repos/mmann1123/xr_fresh/releases/latest"
    )
    re = resp.json()["tag_name"]
except Exception:
    re = "0.2.1"

project = "xr_fresh"
author = "Michael Mann"
release = re

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "mmann1123",  # Username
    "github_repo": "xr_fresh",  # Repo name
    "github_version": "main",  # Branch
    "conf_py_path": "/docs/",  # Path in the repo to the docs root
}

# html_context = {
#     "css_files": [
#         "https://mmann1123.github.io/xr_fresh/_static/pygments.css",
#         "https://mmann1123.github.io/xr_fresh/_static/css/theme.css",
#         "https://mmann1123.github.io/xr_fresh/_static/graphviz.css",
#     ],
#     "script_files": [
#         "https://mmann1123.github.io/xr_fresh/_static/jquery.js",
#         "https://mmann1123.github.io/xr_fresh/_static/_sphinx_javascript_frameworks_compat.js",
#         "https://mmann1123.github.io/xr_fresh/_static/documentation_options.js",
#         "https://mmann1123.github.io/xr_fresh/_static/doctools.js",
#         "https://mmann1123.github.io/xr_fresh/_static/sphinx_highlight.js",
#         "https://mmann1123.github.io/xr_fresh/_static/js/theme.js",
#     ],
# }


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx_rtd_theme",
    "myst_nb",
]
autosummary_generate = True
jupyter_execute_notebooks = "off"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# skip install of the following packages
# autodoc_mock_imports = ["numpy", "geowombat", "gdal"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# pretend to install the following packages
# autodoc_mock_imports = ["xr_fresh", "numpy", "geowombat", "gdal"]
autodoc_mock_imports = [
    "rle",
    "xgboost",
    "xr_fresh.Sklearn_Methods",
    "xr_fresh.script_run_classifier_and_save_results",
    "xr_fresh.testbed",
]


# Skip specific members
def skip_member(app, what, name, obj, skip, options):
    # Skip the 'rle' module or any member containing 'rle'
    if name == "rle" or "rle" in name:
        return True
    return skip


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_baseurl = "https://mmann1123.github.io/xr_fresh/"

html_theme_options = {
    "canonical_url": "",
    "analytics_id": "UA-XXXXXXX-1",
    "logo_only": False,
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


## Copy all html files into the docs directory after build
import os
import shutil


def copy_html_files(app, exception):
    """
    Copies HTML files from the build directory to the docs directory after the build is finished.

    Args:
        app: The Sphinx application object.
        exception: An exception object if an error occurred, otherwise None.
    """
    if exception is not None:
        # If there was an exception during the build, do not proceed with copying
        return

    # Define the source and destination directories
    build_dir = os.path.join(
        app.outdir, "..", "html"
    )  # Typically, app.outdir is docs/_build/html
    destination_dir = os.path.abspath(
        os.path.join(app.srcdir, "..")
    )  # Typically, docs/

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate through all HTML files in the build directory
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            if file.endswith(".html"):
                src_file = os.path.join(root, file)
                # Compute the relative path to preserve the directory structure
                rel_path = os.path.relpath(root, build_dir)
                dest_path = os.path.join(destination_dir, rel_path)
                os.makedirs(dest_path, exist_ok=True)
                shutil.copy2(src_file, os.path.join(dest_path, file))


def setup(app):
    """
    Setup function to connect event hooks.

    Args:
        app: The Sphinx application object.
    """
    app.connect("autodoc-skip-member", skip_member)
    app.connect("build-finished", copy_html_files)

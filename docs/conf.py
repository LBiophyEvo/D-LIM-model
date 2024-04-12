# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'D-LIM'
copyright = '2024, Shuhui Wang, Vaitea Oupp'
author = 'Shuhui Wang, Vaitea Opuu'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# pip install nbsphinx 
extensions = [
    "nbsphinx",
    'IPython.sphinxext.ipython_console_highlighting',
]

exclude_patterns = ['_build', '**.ipynb_checkpoints']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# pip install sphinx-rtd-theme
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = '_static/dlim.png'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'canonical_url': 'https://dlim.readthedocs.io',
    #'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'none', #'bottom',
    'style_external_links': False,
    #'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}


html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "LBiophyEvo", # Username
    "github_repo": "D-LIM-model", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}


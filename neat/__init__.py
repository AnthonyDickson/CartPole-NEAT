"""Runs the neat algorithm with the default parameters.

Note: Typically I would leave the __init__.py file empty, but in order to
support profiling with PyCharm I had to call the main function from here. This
doesn't seem to affect normal execution via the command 'python -m neat'.
"""

from neat.__main__ import main

if __name__ == '__main__':
    main(debug_mode=True)

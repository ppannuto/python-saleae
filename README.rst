python-saleae
=============

This library implements the control protocol for the
`Saleae Logic Analyzer <https://www.saleae.com/>`__. It is based off of the
documentation and example here:
http://support.saleae.com/hc/en-us/articles/201104764-Socket-API-beta

Currently, this is basically a direct mapping of API calls with some small
sanity checking and conveniences. It has not been extensively tested beyond
my immediate needs, but it also should not have any known problems.

Issues, updates, pull requests, etc should be directed
`to github <https://github.com/ppannuto/python-saleae>`__.


Installation
------------

The easiest method is to simply use pip:

::

    (sudo) pip install saleae

Usage
-----

::

    import saleae
    s = saleae.SaleaeConnection()
    s.capture_to_file('~/test.logicdata')


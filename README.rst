python-saleae
=============

|travisci| |docs|

This library implements the control protocol for the
`Saleae Logic Analyzer <https://www.saleae.com/>`__. It is based off of the
documentation and example here:
https://github.com/saleae/SaleaeSocketApi

**IMPORTANT: You must enable the 'Remote Scripting Server' in Saleae.** Click
on "Options" in the top-right, the "Developer" tab, and check "Enable scripting
socket server". This should not require a restart.

This library requires Saleae Logic 1.2.x or greater. Unfortunately there is no
way to check the version of Logic running using the scripting protocol so this
is difficult to check at runtime.

  Note: Unfortunately, the new Logic2 software does not yet support remote
  access, so you will have to use the original Logic software.
  
  You can track Logic2 remote access progress on this thread from Saleae:
  https://discuss.saleae.com/t/scripting-socket-api/108/3
  
  **Update: July 2022:** Saleae is developing an official remote scripting
  inteface and library for Logic2. Check out the alpha/beta and give feedback:
  https://discuss.saleae.com/t/saleae-logic-2-automation-api/1685/13

Currently, this is basically a direct mapping of API calls with some small
sanity checking and conveniences. It has not been extensively tested beyond
my immediate needs, but it also should not have any known problems.

To get a feel for how the library works and what it can do, try the built-in demo:

::

    #!/usr/bin/env python3
    import saleae
    saleae.demo()


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
    s = saleae.Saleae()
    s.capture_to_file('/tmp/test.logicdata')


.. |docs| image:: https://readthedocs.org/projects/python-saleae/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://python-saleae.readthedocs.org/

.. |travisci| image:: https://travis-ci.org/ppannuto/python-saleae.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/ppannuto/python-saleae



License
-------

Licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

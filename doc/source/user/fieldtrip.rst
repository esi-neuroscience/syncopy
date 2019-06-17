Syncopy for Fieldtrip users
===========================

Syncopy is written in the `Python programming language
<https://www.python.org/>`_ using the `NumPy <https://www.numpy.org/>`_ and
`SciPy <https://scipy.org/>`_ libraries. However, it's call signatures and
parameter names are designed to mimick the `MATLAB <https://mathworks.com>`_
analysis toolbox `Fieldtrip <http://www.fieldtriptoolbox.org>`_. 

The scope of Syncopy is limited to only parts of Fieldtrip, in particular
spectral analysis of electrophysiology data. Therefore, M/EEG-specific routines
such as loading M/EEG file types, source localization, ..., are not part of
Syncopy. For a Python toolbox tailored to M/EEG data analysis, see for example
the `MNE Project <https://www.martinos.org/mne/>`_.


Translating MATLAB code Python
------------------------------

For translating code from MATLAB to Python there are several guides, e.g.

* the `Mathesaurus <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_
* `NumPy for Matlab users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_
* `MATLAB to Python - A Migration Guide by Enthought <https://www.enthought.com/white-paper-matlab-to-python>`_

Key differences between Python and MATLAB
-----------------------------------------

While the above links cover differences between Python and MATLAB to a great
extent, we highlight here what we think are the most important differences: 

* Python array indexing starts at 0
* The end of a range in Python is not included
* Data in Python is not necessarily copied and may be manipulated in-place.
* Namespaces


Translating Fieldtrip calls to Syncopy
--------------------------------------

Using a Fieldtrip function in MATLAB usually works via constructing a ``cfg``
``struct`` that contains all configured parameters, e.g. frequency and time
limits for spetral analysis:

.. code-block:: matlab

    ft_defaults
    cfg = [];
    cfg.option1 = 'yes';
    cfg.option2 = [10, 20];
    result = ft_something(cfg);

In Syncopy this struct is a Python dictionary that can automatically be filled
with the defaults of any function.

.. code-block:: python

    import syncopy as spy
    cfg = spy.get_defaults(spy.something)
    cfg.option1 = 'yes'
    # or
    cfg.option1 = True
    cfg.option2 = [10, 20]
    result = spy.something(cfg=cfg)


Key differences between Fieldtrip and Syncopy
---------------------------------------------

* Data is only loaded into memory on demand
* ...
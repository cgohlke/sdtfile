Read Becker & Hickl SDT files
=============================

Sdtfile is a Python library to read SDT files produced by Becker & Hickl
SPCM software. SDT files contain time correlated single photon counting
instrumentation parameters and measurement data. Currently only the
"Setup & Data", "DLL Data", and "FCS Data" formats are supported.

`Becker & Hickl GmbH <http://www.becker-hickl.de/>`_ is a manufacturer of
equipment for photon counting.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2023.9.28

Quickstart
----------

Install the sdtfile package and all dependencies from the
`Python Package Index <https://pypi.org/project/sdtfile/>`_::

    python -m pip install -U sdtfile

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/sdtfile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.5, 3.12rc
- `Numpy <https://pypi.org/project/numpy>`_ 1.25.2

Revisions
---------

2023.9.28

- Update structs to SPCM v.9.66 (breaking).
- Shorten MEASURE_INFO struct to meas_desc_block_length.

2023.8.30

- Fix linting issues.
- Add py.typed marker.
- Drop support for Python 3.8 and numpy < 1.22 (NEP29).

2022.9.28

- Convert docstrings to Google style with Sphinx directives.

2022.2.2

- Add type hints.
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.11.18

- Fix reading FLIM files created by Prairie View software (#5).

2021.3.21

- …

Refer to the CHANGES file for older revisions.

References
----------

1. W Becker. The bh TCSPC Handbook. 9th Edition. Becker & Hickl GmbH 2021.
   pp 879.
2. SPC_data_file_structure.h header file. Part of the Becker & Hickl
   SPCM software installation.

Examples
--------

Read image and metadata from a "SPC Setup & Data File":

>>> sdt = SdtFile('image.sdt')
>>> sdt.header.revision
588
>>> sdt.info.id[1:-1]
'SPC Setup & Data File'
>>> int(sdt.measure_info[0].scan_x[0])
128
>>> len(sdt.data)
1
>>> sdt.data[0].shape
(128, 128, 256)
>>> sdt.times[0].shape
(256,)

Read data and metadata from a "SPC Setup & Data File" with multiple data sets:

>>> sdt = SdtFile('fluorescein.sdt')
>>> len(sdt.data)
4
>>> sdt.data[3].shape
(1, 1024)
>>> sdt.times[3].shape
(1024,)

Read image data from a "SPC FCS Data File" as numpy array:

>>> sdt = SdtFile('fcs.sdt')
>>> sdt.info.id[1:-1]
'SPC FCS Data File'
>>> len(sdt.data)
1
>>> sdt.data[0].shape
(512, 512, 256)
>>> sdt.times[0].shape
(256,)

Read Becker & Hickl SDT files
=============================

Sdtfile is a Python library to read SDT files produced by Becker & Hickl
SPCM software. SDT files contain time correlated single photon counting
instrumentation parameters and measurement data. Currently only the
"Setup & Data", "DLL Data", and "FCS Data" formats are supported.

`Becker & Hickl GmbH <http://www.becker-hickl.de/>`_ is a manufacturer of
equipment for photon counting.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.7.28

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.13 <https://www.numpy.org>`_

Revisions
---------
2019.7.28
    Fix reading compressed, multi-channel data.
2018.9.22
    Use str, not bytes for ASCII data.
2018.8.29
    Move module into sdtfile package.
2018.2.7
    Bug fixes.
2016.3.30
    Support revision 15 files and compression.
2015.1.29
    Read SPC DLL data files.
2014.9.5
    Fixed reading multiple MEASURE_INFO records.

Notes
-----
The API is not stable yet and might change between revisions.

Python 2.7 and 3.4 are deprecated.

References
----------
(1) W Becker. The bh TCSPC Handbook. Third Edition. Becker & Hickl GmbH 2008.
    pp 401.
(2) SPC_data_file_structure.h header file. Part of the Becker & Hickl
    SPCM software.

Examples
--------

Read image and metadata from a SPC Setup & Data File:

>>> sdt = SdtFile('image.sdt')
>>> sdt.header.revision
588
>>> sdt.info.id[1:-1]
'SPC Setup & Data File'
>>> int(sdt.measure_info[0].scan_x)
128
>>> len(sdt.data)
1
>>> sdt.data[0].shape
(128, 128, 256)
>>> sdt.times[0].shape
(256,)

Read data and metadata from a SPC Setup & Data File with mutliple data sets:

>>> sdt = SdtFile('fluorescein.sdt')
>>> len(sdt.data)
4
>>> sdt.data[3].shape
(1, 1024)
>>> sdt.times[3].shape
(1024,)

Read data from a SPC FCS Data File as numpy array:

>>> sdt = SdtFile('fcs.sdt')
>>> sdt.info.id[1:-1]
'SPC FCS Data File'
>>> len(sdt.data)
1
>>> sdt.data[0].shape
(512, 512, 256)
>>> sdt.times[0].shape
(256,)

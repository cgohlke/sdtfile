# sdtfile.py

# Copyright (c) 2007-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read Becker & Hickl SDT files.

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

:License: BSD 3-Clause

:Version: 2020.12.10

Requirements
------------
* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_

Revisions
---------
2020.12.10
    Fix shape of non-square frames.
2020.8.3
    Fix integer overflow (#3).
    Support os.PathLike file names.
2020.1.1
    Fix reading MCS_BLOCK data.
    Remove support for Python 2.7 and 3.5.
    Update copyright.
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
    Fix reading multiple MEASURE_INFO records.

Notes
-----
The API is not stable yet and might change between revisions.

This module has been tested with a limited number of files only.

References
----------
1. W Becker. The bh TCSPC Handbook. Third Edition. Becker & Hickl GmbH 2008.
   pp 401.
2. SPC_data_file_structure.h header file. Part of the Becker & Hickl
   SPCM software.

Examples
--------

Read image and metadata from a "SPC Setup & Data File":

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

Read data and metadata from a "SPC Setup & Data File" with mutliple data sets:

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

"""

__version__ = '2020.12.10'

__all__ = (
    'SdtFile',
    'FileInfo',
    'SetupBlock',
    'BlockNo',
    'BlockType',
    'FileRevision',
)

import os
import io
import zipfile

import numpy


class SdtFile:
    """Becker & Hickl SDT file.

    Attributes
    ----------
    header : numpy.rec.array of FILE_HEADER structure
        General information about the location of the setup and measurement
        data within the file.
    info : FileInfo
        General information in ASCII format.
    setup : SetupBlock or None
        Setup block containing all system parameters, display parameters, etc.
    measure_info : numpy.rec.array of MEASURE_INFO structure
        Measurement description blocks.
    block_headers : list of numpy.rec.array of BLOCK_HEADER structure
        Data block headers.
    data : list of 2D numpy arrays
        Photon counts at each curve point.
    times : list of 1D numpy arrays
        Time axes for each data set.

    """

    def __init__(self, arg):
        """Initialize instance from file name or open file."""
        if hasattr(arg, 'seek'):
            self.filename = ''
            self._fromfile(arg)
        else:
            self.filename = os.fspath(arg)
            with open(arg, 'rb') as fh:
                self._fromfile(fh)

    def _fromfile(self, fh):
        """Initialize instance from open file."""
        # read file header
        self.header = numpy.rec.fromfile(
            fh, dtype=FILE_HEADER, shape=1, byteorder='<'
        )[0]
        if self.header.header_valid != 0x5555:
            raise ValueError('not a SDT file')
        if self.header.no_of_data_blocks == 0x7FFF:
            self.header.no_of_data_blocks = self.header.reserved1
        elif self.header.no_of_data_blocks > 0x7FFF:
            raise ValueError('')

        # read file info
        fh.seek(self.header.info_offset)
        info = fh.read(self.header.info_length).decode('windows-1250')
        info = info.replace('\r\n', '\n')
        self.info = FileInfo(info)
        try:
            if self.info.id not in (
                'SPC Setup & Data File',
                'SPC FCS Data File',
                'SPC DLL Data File',
            ):
                raise NotImplementedError('not supported:', self.info.id)
        except AttributeError as exc:
            raise ValueError('invalid SDT file info\n', self.info) from exc

        # read setup block
        if self.header.setup_length:
            fh.seek(self.header.setup_offs)
            self.setup = SetupBlock(fh.read(self.header.setup_length))
        else:
            # SPC DLL data file contain no setup, only data
            self.setup = None

        # read measurement description blocks
        self.measure_info = []
        dtype = numpy.dtype(MEASURE_INFO)
        if dtype.itemsize > self.header.meas_desc_block_length:
            # TODO: shorten MEASURE_INFO to meas_desc_block_length
            pass
        fh.seek(self.header.meas_desc_block_offset)
        for _ in range(self.header.no_of_meas_desc_blocks):
            self.measure_info.append(
                numpy.rec.fromfile(fh, dtype=dtype, shape=1, byteorder='<')
            )
            fh.seek(self.header.meas_desc_block_length - dtype.itemsize, 1)

        rev = FileRevision(self.header.revision)
        block_header_t = BLOCK_HEADER if rev.revision < 15 else BLOCK_HEADER_15

        self.times = []
        self.data = []
        self.block_headers = []

        offset = self.header.data_block_offset
        for _ in range(self.header.no_of_data_blocks):
            # read data block header
            fh.seek(offset)
            bh = numpy.rec.fromfile(
                fh, dtype=block_header_t, shape=1, byteorder='<'
            )[0]
            self.block_headers.append(bh)
            # read data block
            mi = self.measure_info[bh.meas_desc_block_no]
            bt = BlockType(bh.block_type)
            dtype = bt.dtype
            dsize = bh.block_length // dtype.itemsize
            fh.seek(bh.data_offs)
            if bt.compress:
                bio = io.BytesIO(fh.read(bh.next_block_offs - bh.data_offs))
                with zipfile.ZipFile(bio) as zf:
                    data = zf.read('data_block')
                del bio
                data = numpy.frombuffer(data, dtype=dtype, count=dsize)
            else:
                data = numpy.fromfile(fh, dtype=dtype, count=dsize)

            # TODO: support more block types
            # the following works with DECAY_BLOCK, IMG_BLOCK, and MCS_BLOCK
            adc_re = int(mi.adc_re)
            scan_x = int(mi.scan_x)
            scan_y = int(mi.scan_y)
            image_x = int(mi.image_x)
            image_y = int(mi.image_y)
            if dsize == scan_x * scan_y * adc_re:
                data = data.reshape(scan_y, scan_x, adc_re)
            elif dsize == image_x * image_y * adc_re:
                data = data.reshape(image_y, image_x, adc_re)
            elif dsize == mi.MeasHISTInfo.mcs_points[0]:
                data = data.reshape(-1, dsize)
            else:
                data = data.reshape(-1, adc_re)
            self.data.append(data)

            if bt.contents == 'MCS_BLOCK':
                t = numpy.arange(dsize, dtype=numpy.float64)
                t *= mi.MeasHISTInfo.mcs_time[0]
            else:
                # generate time axis
                t = numpy.arange(adc_re, dtype=numpy.float64)
                t *= mi.tac_r / (float(mi.tac_g) * adc_re)
            self.times.append(t)
            offset = bh.next_block_offs

    def block_measure_info(self, block):
        """Return measure_info record for data block."""
        return self.measure_info[self.block_headers[block].meas_desc_block_no]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __str__(self):
        """Return string containing all information about SdtFile."""
        # measure_info = '\n  '.join(str(i) for i in self.measure_info)
        # block_headers = '\n  '.join(
        #     indent(i, '  ') for i in self.block_headers)
        block_types = '\n '.join(
            indent(BlockType(i.block_type)) for i in self.block_headers
        )
        shapes = '\n  '.join(str(i.shape) for i in self.data)
        return '\n '.join(
            (
                self.__class__.__name__,
                os.path.normpath(os.path.normcase(self.filename)),
                indent(FileRevision(self.header.revision)),
                indent(self.info.strip()),
                # f'header: {self.header}',
                # f'measure_info:\n  {measure_info}',
                # f'block_headers:\n  {block_headers}',
                block_types,
                f'shapes:\n  {shapes}',
            )
        )


class FileInfo(str):
    """File info string and attributes."""

    def __init__(self, value):
        str.__init__(self)
        assert value.startswith('*IDENTIFICATION') and value.strip().endswith(
            '*END'
        )

        for line in value.splitlines()[1:-1]:
            try:
                key, val = line.split(':', 1)
            except Exception:
                pass
            else:
                setattr(self, key.strip().lower(), val.strip())


class SetupBlock:
    """Setup block ascii and binary data."""

    __slots__ = ('ascii', 'binary')

    def __init__(self, value):
        assert value.startswith(b'*SETUP') and value.strip().endswith(b'*END')
        i = value.find(b'BIN_PARA_BEGIN')
        if i:
            self.ascii = value[:i].decode('windows-1250')
            self.binary = bytes(value[i + 15 : -10])
            # TODO: parse binary data here
        else:
            self.ascii = value.decode('windows-1250')
            self.binary = None

    def __str__(self):
        return self.ascii


class BlockNo:
    """The lblock_no field of BLOCK_HEADER."""

    __slots__ = ('data', 'module')

    def __init__(self, value):
        self.data = (value & 0xFFFFFF00) >> 24
        self.module = value & 0x000000FF

    def __iter__(self):
        return iter((self.data, self.module))

    def __str__(self):
        return '\n '.join(
            (
                self.__class__.__name__,
                f'data number: {self.data}',
                f'module number: {self.module}',
            )
        )


class BlockType:
    """The block_type field of BLOCK_HEADER."""

    __slots__ = ('mode', 'contents', 'dtype', 'compress')

    def __init__(self, value):
        self.mode = BLOCK_CREATION[value & 0xF]
        self.contents = BLOCK_CONTENT[value & 0xF0]
        self.dtype = BLOCK_DTYPE[value & 0xF00]
        self.compress = bool(value & 0x1000)

    def __iter__(self):
        return iter((self.mode, self.contents, self.dtype))

    def __str__(self):
        return '\n '.join(
            (
                self.__class__.__name__,
                f'mode: {self.mode}',
                f'contents: {self.contents}',
                f'dtype: {self.dtype}',
                f'compress: {self.compress}',
            )
        )


class FileRevision:
    """The revision field of FILE_HEADER."""

    __slots__ = ('revision', 'module')

    def __init__(self, value):
        self.revision = value & 0b1111
        self.module = {
            0x20: 'SPC-130',
            0x21: 'SPC-600',
            0x22: 'SPC-630',
            0x23: 'SPC-700',
            0x24: 'SPC-730',
            0x25: 'SPC-830',
            0x26: 'SPC-140',
            0x27: 'SPC-930',
            0x28: 'SPC-150',
            0x29: 'DPC-230',
            0x2A: 'SPC-130EM',
        }.get((value & 0xFF0) >> 4, 'Unknown')

    def __str__(self):
        return '\n '.join(
            (
                self.__class__.__name__,
                f'revision: {self.revision}',
                f'module type: {self.module}',
            )
        )


FILE_HEADER = [
    ('revision', 'i2'),
    ('info_offset', 'i4'),
    ('info_length', 'i2'),
    ('setup_offs', 'i4'),
    ('setup_length', 'i2'),
    ('data_block_offset', 'i4'),
    ('no_of_data_blocks', 'i2'),
    ('data_block_length', 'i4'),
    ('meas_desc_block_offset', 'i4'),
    ('no_of_meas_desc_blocks', 'i2'),
    ('meas_desc_block_length', 'i2'),
    ('header_valid', 'u2'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2'),
    ('chksum', 'u2'),
]

SETUP_BIN_HDR = [
    ('soft_rev', 'u4'),
    ('para_length', 'u4'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2'),
]

# Info collected when measurement finished
MEASURE_STOP_INFO = [
    ('status', 'u2'),
    ('flags', 'u2'),
    ('stop_time', 'f4'),
    ('cur_step', 'i4'),
    ('cur_cycle', 'i4'),
    ('cur_page', 'i4'),
    ('min_sync_rate', 'f4'),
    ('min_cfd_rate', 'f4'),
    ('min_tac_rate', 'f4'),
    ('min_adc_rate', 'f4'),
    ('max_sync_rate', 'f4'),
    ('max_cfd_rate', 'f4'),
    ('max_tac_rate', 'f4'),
    ('max_adc_rate', 'f4'),
    ('reserved1', 'i4'),
    ('reserved2', 'f4'),
]

# Info collected when FIFO measurement finished
MEASURE_FCS_INFO = [
    ('chan', 'u2'),
    ('fcs_decay_calc', 'u2'),
    ('mt_resol', 'u4'),
    ('cortime', 'f4'),
    ('calc_photons', 'u4'),
    ('fcs_points', 'i4'),
    ('end_time', 'f4'),
    ('overruns', 'u2'),
    ('fcs_type', 'u2'),
    ('cross_chan', 'u2'),
    ('mod', 'u2'),
    ('cross_mod', 'u2'),
    ('cross_mt_resol', 'u4'),
]

# Extension of MeasFCSInfo for other histograms
HIST_INFO = [
    ('fida_time', 'f4'),
    ('filda_time', 'f4'),
    ('fida_points', 'i4'),
    ('filda_points', 'i4'),
    ('mcs_time', 'f4'),
    ('mcs_points', 'i4'),
]

# Measurement description blocks
MEASURE_INFO = [
    ('time', 'a9'),
    ('date', 'a11'),
    ('mod_ser_no', 'a16'),
    ('meas_mode', 'i2'),
    ('cfd_ll', 'f4'),
    ('cfd_lh', 'f4'),
    ('cfd_zc', 'f4'),
    ('cfd_hf', 'f4'),
    ('syn_zc', 'f4'),
    ('syn_fd', 'i2'),
    ('syn_hf', 'f4'),
    ('tac_r', 'f4'),
    ('tac_g', 'i2'),
    ('tac_of', 'f4'),
    ('tac_ll', 'f4'),
    ('tac_lh', 'f4'),
    ('adc_re', 'i2'),
    ('eal_de', 'i2'),
    ('ncx', 'i2'),
    ('ncy', 'i2'),
    ('page', 'u2'),
    ('col_t', 'f4'),
    ('rep_t', 'f4'),
    ('stopt', 'i2'),
    ('overfl', 'u1'),
    ('use_motor', 'i2'),
    ('steps', 'u2'),
    ('offset', 'f4'),
    ('dither', 'i2'),
    ('incr', 'i2'),
    ('mem_bank', 'i2'),
    ('mod', 'a16'),
    ('syn_th', 'f4'),
    ('dead_time_comp', 'i2'),
    ('polarity_l', 'i2'),
    ('polarity_f', 'i2'),
    ('polarity_p', 'i2'),
    ('linediv', 'i2'),
    ('accumulate', 'i2'),
    ('flbck_y', 'i4'),
    ('flbck_x', 'i4'),
    ('bord_u', 'i4'),
    ('bord_l', 'i4'),
    ('pix_time', 'f4'),
    ('pix_clk', 'i2'),
    ('trigger', 'i2'),
    ('scan_x', 'i4'),
    ('scan_y', 'i4'),
    ('scan_rx', 'i4'),
    ('scan_ry', 'i4'),
    ('fifo_typ', 'i2'),
    ('epx_div', 'i4'),
    ('mod_code', 'u2'),
    ('mod_fpga_ver', 'u2'),
    ('overflow_corr_factor', 'f4'),
    ('adc_zoom', 'i4'),
    ('cycles', 'i4'),
    ('StopInfo', MEASURE_STOP_INFO),
    ('FCSInfo', MEASURE_FCS_INFO),
    ('image_x', 'i4'),
    ('image_y', 'i4'),
    ('image_rx', 'i4'),
    ('image_ry', 'i4'),
    ('xy_gain', 'i2'),
    ('master_clock', 'i2'),
    ('adc_de', 'i2'),
    ('det', 'i2'),
    ('x_axis', 'i2'),
    ('MeasHISTInfo', HIST_INFO),
]

BLOCK_HEADER = [
    ('block_no', 'i2'),
    ('data_offs', 'i4'),
    ('next_block_offs', 'i4'),
    ('block_type', 'u2'),
    ('meas_desc_block_no', 'i2'),
    ('lblock_no', 'u4'),
    ('block_length', 'u4'),
]

BLOCK_HEADER_15 = [
    ('data_offs_ext', 'u1'),
    ('next_block_offs_ext', 'u1'),
    ('data_offs', 'u4'),
    ('next_block_offs', 'u4'),
    ('block_type', 'u2'),
    ('meas_desc_block_no', 'i2'),
    ('lblock_no', 'u4'),
    ('block_length', 'u4'),
]

# Mode of creation
BLOCK_CREATION = {
    0: 'NOT_USED',
    1: 'MEAS_DATA',
    2: 'FLOW_DATA',
    3: 'MEAS_DATA_FROM_FILE',
    4: 'CALC_DATA',
    5: 'SIM_DATA',
    8: 'FIFO_DATA',
    9: 'FIFO_DATA_FROM_FILE',
}

BLOCK_CONTENT = {
    0x0: 'DECAY_BLOCK',
    0x10: 'PAGE_BLOCK',
    0x20: 'FCS_BLOCK',
    0x30: 'FIDA_BLOCK',
    0x40: 'FILDA_BLOCK',
    0x50: 'MCS_BLOCK',
    0x60: 'IMG_BLOCK',
    0x70: 'MCSTA_BLOCK',
    0x80: 'IMG_MCS_BLOCK',
    0x90: 'MOM_BLOCK',
    0xA0: 'IMG_INT_BLOCK',
    0xB0: 'IMG_WF_BLOCK',
    0xC0: 'IMG_LIFE_BLOCK',
}

# Data type
BLOCK_DTYPE = {
    0x000: numpy.dtype('<u2'),
    0x100: numpy.dtype('<u4'),
    0x200: numpy.dtype('<f8'),
}

HEADER_VALID = {0x1111: False, 0x5555: True}

INFO_IDS = {
    'SPC Setup Script File': 'Setup script mode: setup only',
    'SPC Setup & Data File': 'Normal mode: setup + data',
    'SPC DLL Data File': 'DLL created: no setup, only data',
    'SPC Flow Data File': 'Continuous Flow mode: no setup, only data',
    'SPC FCS Data File': (
        'FIFO mode: setup, data blocks = Decay, FCS, FIDA, FILDA & MCS '
        'curves for each used routing channel'
    ),
}


def indent(string, pad=' '):
    """Return string with all but first line indented."""
    return f'\n{pad}'.join(str(string).splitlines())


if __name__ == '__main__':
    import doctest

    doctest.testmod()

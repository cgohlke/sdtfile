# sdtfile.py

# Copyright (c) 2007-2024, Christoph Gohlke
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

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.4.24
:DOI: `10.5281/zenodo.10125608 <https://doi.org/10.5281/zenodo.10125608>`_

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

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.9, 3.12.3
- `NumPy <https://pypi.org/project/numpy>`_ 1.26.4

Revisions
---------

2024.4.24

- Support NumPy 2.

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
>>> int(sdt.header.revision)
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

"""

from __future__ import annotations

__version__ = '2024.4.24'

__all__ = [
    'SdtFile',
    'FileInfo',
    'SetupBlock',
    'BlockNo',
    'BlockType',
    'FileRevision',
]

import io
import os
import zipfile
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from typing import Any, BinaryIO

    from numpy.typing import NDArray


class SdtFile:
    """Becker & Hickl SDT file.

    Parameters:
        arg: File name or open file.

    """

    filename: str
    """Name of file."""

    header: numpy.recarray
    """File header of type FILE_HEADER."""

    info: FileInfo
    """File info string and attributes."""

    setup: SetupBlock | None
    """Setup block ascii and binary data."""

    measure_info: list[numpy.recarray]
    """Measurement description blocks of type MEASURE_INFO."""

    block_headers: list[numpy.recarray]
    """Data block headers of type BLOCK_HEADER."""

    data: list[NDArray[Any]]
    """Photon counts at each curve point."""

    times: list[NDArray[Any]]
    """Time axes for each data set."""

    def __init__(self, arg: str | os.PathLike | BinaryIO, /) -> None:
        if isinstance(arg, (str, os.PathLike)):
            self.filename = os.fspath(arg)
            with open(arg, 'rb') as fh:
                self._fromfile(fh)
        else:
            assert hasattr(arg, 'seek')
            self.filename = ''
            self._fromfile(arg)

    def _fromfile(self, fh: BinaryIO, /) -> None:
        """Initialize instance from open file."""
        # read file header
        self.header = numpy.rec.fromfile(  # type: ignore
            fh, dtype=FILE_HEADER, shape=1, byteorder='<'
        )[0]
        if self.header.chksum != 0x55AA and self.header.header_valid != 0x5555:
            raise ValueError('not a SDT file')
        if self.header.no_of_data_blocks == 0x7FFF:
            self.header.no_of_data_blocks = self.header.reserved1
        elif self.header.no_of_data_blocks > 0x7FFF:
            raise ValueError('')

        # read file info
        fh.seek(self.header.info_offs)
        info = fh.read(self.header.info_length).decode('windows-1250')
        info = info.replace('\r\n', '\n')
        self.info = FileInfo(info)
        try:
            if self.info.id not in (
                'SPC Setup & Data File',
                'SPC FCS Data File',
                'SPC DLL Data File',
                'SPC Setup & Data File',  # corrupted?
            ):
                raise NotImplementedError(f'{self.info.id!r} not supported')
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
        dtype = struct_dtype(
            MEASURE_INFO, int(self.header.meas_desc_block_length)
        )
        fh.seek(self.header.meas_desc_block_offs)
        for _ in range(self.header.no_of_meas_desc_blocks):
            self.measure_info.append(
                numpy.rec.fromfile(  # type: ignore
                    fh, dtype=dtype, shape=1, byteorder='<'
                )
            )
            fh.seek(self.header.meas_desc_block_length - dtype.itemsize, 1)

        rev = FileRevision(self.header.revision)
        if rev.revision >= 15:
            block_header_t = BLOCK_HEADER
        else:
            block_header_t = BLOCK_HEADER_OLD

        self.times = []
        self.data = []
        self.block_headers = []

        offset = self.header.data_block_offs
        for _ in range(self.header.no_of_data_blocks):
            # read data block header
            fh.seek(offset)
            bh = numpy.rec.fromfile(  # type: ignore
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
                    databytes = zf.read(zf.filelist[0].filename)  # data_block
                del bio
                data = numpy.frombuffer(databytes, dtype=dtype, count=dsize)
            else:
                data = numpy.fromfile(fh, dtype=dtype, count=dsize)

            # TODO: support more block types
            # the following works with DECAY, IMG, MCS, PAGE

            # assume adc_re is always present
            adc_re = int(mi.adc_re[0])

            # the following fields may not be present
            try:
                scan_x = int(mi.scan_x[0])
                scan_y = int(mi.scan_y[0])
            except AttributeError:
                scan_x = 0
                scan_y = 0
            try:
                image_x = int(mi.image_x[0])
                image_y = int(mi.image_y[0])
            except AttributeError:
                image_x = 0
                image_y = 0
            try:
                mcs_points = mi.MeasHISTInfo.mcs_points[0]
            except AttributeError:
                mcs_points = -1
            try:
                mcs_time = mi.MeasHISTInfo.mcs_time[0]
            except AttributeError:
                mcs_time = 0

            if adc_re == 0:
                adc_re = 65536
            if dsize == scan_x * scan_y * adc_re:
                data = data.reshape(scan_y, scan_x, adc_re)
            elif dsize == image_x * image_y * adc_re:
                data = data.reshape(image_y, image_x, adc_re)
            elif dsize == mcs_points:
                data = data.reshape(-1, dsize)
            else:
                data = data.reshape(-1, adc_re)
            self.data.append(data)

            if bt.contents == 'MCS_BLOCK' and mcs_time != 0:
                time = numpy.arange(dsize, dtype=numpy.float64)
                time *= mcs_time
            else:
                # generate time axis
                time = numpy.arange(adc_re, dtype=numpy.float64)
                time *= mi.tac_r / (float(mi.tac_g[0]) * adc_re)
            self.times.append(time)
            offset = bh.next_block_offs

    def block_measure_info(self, block: int, /) -> numpy.recarray:
        """Return measure_info record for data block.

        Parameters:
            block: Block index.

        """
        return self.measure_info[self.block_headers[block].meas_desc_block_no]

    def __enter__(self) -> SdtFile:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __repr__(self) -> str:
        filename = os.path.split(self.filename)[-1]
        return f'{self.__class__.__name__}({filename!r})'

    def __str__(self) -> str:
        return indent(
            repr(self),
            # os.path.normpath(os.path.normcase(self.filename)),
            FileRevision(self.header.revision),
            indent('info:', self.info.strip()),
            # indent('header:', self.header),
            # indent('measure_info:', *self.measure_info),
            # indent('block_headers:', *self.block_headers),
            indent(
                'blocktypes:',
                *(BlockType(i.block_type) for i in self.block_headers),
            ),
            indent('shapes:', *(i.shape for i in self.data)),
        )


class FileInfo(str):
    """File info string and attributes.

    Parameters:
        value: File content from FILE_HEADER info_offs and info_length.

    """

    id: str
    """Identification."""

    def __init__(self, value: str, /) -> None:
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
    """Setup block ascii and binary data.

    Parameters:
        value: File content from FILE_HEADER setup_offs and setup_length.

    """

    __slots__ = ('ascii', 'binary')

    ascii: str
    """ASCII data."""

    binary: bytes | None
    """Binary data."""

    def __init__(self, value: bytes, /) -> None:
        assert value.startswith(b'*SETUP') and value.strip().endswith(b'*END')
        i = value.find(b'BIN_PARA_BEGIN')
        if i:
            self.ascii = value[:i].decode('windows-1250')
            self.binary = value[i:]  # [i + 15 : -10]
            # TODO: parse binary data here
        else:
            self.ascii = value.decode('windows-1250')
            self.binary = None

    def __str__(self) -> str:
        return self.ascii


class BlockNo:
    """BLOCK_HEADER.lblock_no field.

    Parameters:
        value: Value of BLOCK_HEADER.lblock_no.

    """

    __slots__ = ('data', 'module')

    data: int
    """Data."""

    module: int
    """Module."""

    def __init__(self, value: int, /) -> None:
        self.data = (value & 0xFFFFFF00) >> 24
        self.module = value & 0x000000FF

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.data} << 24 & {self.module})'


class BlockType:
    """BLOCK_HEADER.block_type field.

    Parameters:
        value: Value of BLOCK_HEADER.block_type.

    """

    __slots__ = ('mode', 'contents', 'dtype', 'compress')

    mode: str
    """BLOCK_CREATION."""

    contents: str
    """BLOCK_CONTENT."""

    dtype: numpy.dtype
    """BLOCK_DTYPE."""

    compress: bool
    """Data is compressed."""

    def __init__(self, value: int, /) -> None:
        self.mode = BLOCK_CREATION[value & 0xF]
        self.contents = BLOCK_CONTENT[value & 0xF0]
        self.dtype = BLOCK_DTYPE[value & 0xF00]
        self.compress = bool(value & 0x1000)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.mode} {self.contents}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            # f'mode: {self.mode}',
            # f'contents: {self.contents}',
            f'dtype: {self.dtype}',
            f'compress: {self.compress}',
        )


class FileRevision:
    """FILE_HEADER.revision field.

    Parameters:
        value: Value of FILE_HEADER.revision.

    """

    __slots__ = ('revision', 'module')

    revision: int
    """Revision."""

    module: str
    """Module."""

    def __init__(self, value: int, /) -> None:
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
            0x2B: 'SPC-160',
            0x2E: 'SPC-150N',
            0x80: 'SPC-150NX',
            0x81: 'SPC-160X',
            0x82: 'SPC-160PCIE',
            0x83: 'SPC-130EMN',
            0x84: 'SPC-180N',
            0x85: 'SPC-180NX',
            0x86: 'SPC-180NXX',
            0x87: 'SPC-180N-USB',
            0x88: 'SPC-130IN',
            0x89: 'SPC-130INX',
            0x8A: 'SPC-130INXX',
            0x8B: 'SPC-QC-104',
            0x8C: 'SPC-QC-004',
        }.get((value & 0xFF0) >> 4, 'Unknown')

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} {self.module!r} rev {self.revision}>'
        )


FILE_HEADER: list[tuple[str, str]] = [
    ('revision', 'i2'),
    ('info_offs', 'i4'),
    ('info_length', 'i2'),
    ('setup_offs', 'i4'),
    ('setup_length', 'u2'),
    ('data_block_offs', 'i4'),
    ('no_of_data_blocks', 'i2'),
    ('data_block_length', 'u4'),
    ('meas_desc_block_offs', 'i4'),
    ('no_of_meas_desc_blocks', 'i2'),
    ('meas_desc_block_length', 'i2'),
    ('header_valid', 'u2'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2'),
    ('chksum', 'u2'),
]

SETUP_BIN_HDR: list[tuple[str, str]] = [
    ('soft_rev', 'u4'),
    ('para_length', 'u4'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2'),
]

# Info collected when measurement finished
MEASURE_STOP_INFO: list[tuple[str, str]] = [
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
MEASURE_FCS_INFO: list[tuple[str, str]] = [
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
HIST_INFO: list[tuple[str, str]] = [
    ('fida_time', 'f4'),
    ('filda_time', 'f4'),
    ('fida_points', 'i4'),
    ('filda_points', 'i4'),
    ('mcs_time', 'f4'),
    ('mcs_points', 'i4'),
    ('cross_calc_phot', 'u4'),
    ('mcsta_points', 'u2'),
    ('mcsta_flags', 'u2'),
    ('mcsta_tpp', 'u4'),
    ('calc_markers', 'u4'),
    ('fcs_calc_phot', 'u4'),
    ('reserved3', 'u4'),
]

HIST_INFO_EXT: list[tuple[str, str]] = [
    ('first_frame_time', 'f4'),
    ('frame_time', 'f4'),
    ('line_time', 'f4'),
    ('pixel_time', 'f4'),
    ('scan_type', 'i2'),
    ('skip_2nd_line_clk', 'i2'),
    ('right_border', 'u4'),
    ('info', 'S40'),
]

MEASURE_INFO_EXT: list[tuple[str, str]] = [
    ('DCU_in_use', 'u4'),
    ('dcu_ser_no', '4S16'),
    ('axio_name', 'S32'),
    ('axio_lens_name', 'S64'),
    ('SIS_in_use', 'u4'),
    ('sis_ser_no', '4S16'),
    ('gvd_ser_no', 'S16'),
    ('gvd_zoom_factor', 'f4'),
    ('DCS_FOV_at_zoom_1', 'f4'),
    ('axio_connected', 'i2'),
    ('axio_lens_magnifier', 'f4'),
    ('axio_FOV', 'f4'),
    ('tdc_offset', '4f4'),
    ('tdc_control', 'u4'),
    ('reserve', 'S1250'),
]

# Measurement description blocks
MEASURE_INFO: list[tuple[str, str | list[tuple[str, str]]]] = [
    ('time', 'S9'),
    ('date', 'S11'),
    ('mod_ser_no', 'S16'),
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
    ('mod', 'S16'),
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
    ('HISTInfoExt', HIST_INFO_EXT),
    ('sync_delay', 'f4'),
    ('sdel_ser_no', 'u2'),
    ('sdel_input', 'i1'),
    ('mosaic_ctrl', 'i1'),
    ('mosaic_x', 'u1'),
    ('mosaic_y', 'u1'),
    ('frames_per_el', 'i2'),
    ('chan_per_el', 'i2'),
    ('mosaic_cycles_done', 'i4'),
    ('mla_ser_no', 'u2'),
    ('DCC_in_use', 'u1'),
    ('dcc_ser_no', 'S12'),
    ('TiSaLas_status', 'u2'),
    ('TiSaLas_wav', 'u2'),
    ('AOM_status', 'u1'),
    ('AOM_power', 'u1'),
    ('ddg_ser_no', 'S8'),
    ('prior_ser_no', 'i4'),
    ('mosaic_x_hi', 'u1'),
    ('mosaic_y_hi', 'u1'),
    ('reserve', 'S11'),
    ('extension_used', 'u1'),
    ('minfo_ext', MEASURE_INFO_EXT),
]

BLOCK_HEADER_OLD: list[tuple[str, str]] = [
    ('block_no', 'i2'),
    ('data_offs', 'i4'),
    ('next_block_offs', 'i4'),
    ('block_type', 'u2'),
    ('meas_desc_block_no', 'i2'),
    ('lblock_no', 'u4'),
    ('block_length', 'u4'),
]

BLOCK_HEADER: list[tuple[str, str]] = [
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
BLOCK_CREATION: dict[int, str] = {
    0: 'NOT_USED',
    1: 'MEAS_DATA',
    2: 'FLOW_DATA',
    3: 'MEAS_DATA_FROM_FILE',
    4: 'CALC_DATA',
    5: 'SIM_DATA',
    8: 'FIFO_DATA',
    9: 'FIFO_DATA_FROM_FILE',
}

BLOCK_CONTENT: dict[int, str] = {
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
BLOCK_DTYPE: dict[int, numpy.dtype] = {
    0x000: numpy.dtype('<u2'),
    0x100: numpy.dtype('<u4'),
    0x200: numpy.dtype('<f8'),
}

HEADER_VALID: dict[int, bool] = {0x1111: False, 0x5555: True}

INFO_IDS: dict[str, str] = {
    'SPC Setup Script File': 'Setup script mode: setup only',
    'SPC Setup & Data File': 'Normal mode: setup + data',
    'SPC DLL Data File': 'DLL created: no setup, only data',
    'SPC Flow Data File': 'Continuous Flow mode: no setup, only data',
    'SPC FCS Data File': (
        'FIFO mode: setup, data blocks = Decay, FCS, FIDA, FILDA & MCS '
        'curves for each used routing channel'
    ),
}


def struct_dtype(
    struct: list[tuple[str, str | list[tuple[str, str]]]], size: int, /
) -> numpy.dtype:
    """Return numpy dtype for struct not exceeding size bytes."""
    assert size > 0
    fields = len(struct)
    dtype = numpy.dtype(struct)
    while dtype.itemsize > size and fields > 0:
        # last_dtype = struct[-1][1]
        # if (
        #     isinstance(last_dtype, list)
        #     and dtype.itemsize - size < numpy.dtype(last_dtype).itemsize
        # ):
        #     struct = struct.copy()
        #     struct[-1] = (
        #         struct[-1][0],
        #         struct_dtype(last_dtype, dtype.itemsize - size).descr
        #     )
        #     dtype = numpy.dtype(struct)
        #     break
        fields -= 1
        dtype = numpy.dtype(struct[:fields])
    # if dtype.itemsize != size:
    #    log_warning(f'struct size {dtype.itemsize} != {size}')
    return dtype


def indent(*args) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def log_warning(msg: object, *args: object, **kwargs: Any) -> None:
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    assert numpy.dtype(FILE_HEADER).itemsize == 42  # BH_HDR_LENGTH
    assert numpy.dtype(MEASURE_INFO).itemsize == 2048
    assert numpy.dtype(MEASURE_INFO_EXT).itemsize == 1536
    assert numpy.dtype(HIST_INFO).itemsize == 48
    assert numpy.dtype(HIST_INFO_EXT).itemsize == 64
    assert numpy.dtype(MEASURE_FCS_INFO).itemsize == 38
    assert numpy.dtype(MEASURE_STOP_INFO).itemsize == 60
    assert numpy.dtype(SETUP_BIN_HDR).itemsize == 14

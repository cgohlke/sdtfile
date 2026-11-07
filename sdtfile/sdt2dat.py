#!/usr/bin/env python3
# sdtfile/sdt2dat.py

# Copyright (c) 2020-2026, Christoph Gohlke
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

"""Convert SPC decay curves in SDT files to Globals for Spectroscopy DAT files.

This Python script reads single photon counting decay curves from
Becker & Hickl SDT files, calculates and plots phasor coordinates,
and saves the decay curves in Globals for Spectroscopy DAT format.

For command line usage run::

    python -m sdtfile.sdt2dat --help

For example, to analyze the decay curve in `file.sdt` using a reference decay
in `reference.sdt` with lifetime 4.0 ns, after subtracting a background of
100 and extracting data in range 500 to 3900::

    sdt2dat -r reference.sdt -l 4.0 -b 100 -s 500 3900 file.sdt

The decay curves are saved to `file.sdt.dat`. Phasor coordinates are saved
to `sdtphasor.tsv`.

This script depends on Python >= 3.12 and the sdtfile, matplotlib, numpy, and
click libraries, which can be installed with::

    python -m pip install sdtfile matplotlib click

"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray


# TODO: reimplement this with PhasorPy <https://www.phasorpy.org>


def sdtread(filename: str) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return decay curve and time axis from Becker & Hickl SDT file."""
    from sdtfile import BlockType, SdtFile

    with SdtFile(filename) as sdt:
        for i in range(len(sdt.data)):
            bh = sdt.block_headers[i]
            if BlockType(bh.block_type).contents == 'DECAY_BLOCK':
                return sdt.data[i].squeeze(), sdt.times[i].squeeze()
        msg = 'SDT file does not contain any DECAY_BLOCK'
        raise TypeError(msg)


def tsvwrite(filename: str, labels: Sequence[str], *args: Any) -> None:
    """Write arrays to TSV file."""
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write('\t'.join(labels))
        fh.write('\n')
        for i in range(len(args[0])):
            fh.write('\t'.join(str(arg[i]) for arg in args))
            fh.write('\n')


def globalsdat_write(
    filename: str,
    decay: NDArray[Any],
    reference: NDArray[Any],
    frequency: float,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> None:
    """Write decay and reference arrays to Globals DAT file."""
    decay = numpy.squeeze(decay)
    reference = numpy.squeeze(reference)
    if decay.ndim != 1 or decay.shape != reference.shape:
        msg = 'decay and reference must be 1D arrays of same length'
        raise ValueError(msg)
    start, stop = slice(start, stop).indices(len(decay))[:2]
    step = 1e3 / len(decay) / frequency
    with open(filename, 'w', encoding='ascii') as fh:
        fh.write(f'{step:8.5f}    {start}     {stop}\n')
        fh.writelines(
            f'{int(r)}   {int(d)}\n'
            for d, r in zip(
                decay.astype(numpy.int64),
                reference.astype(numpy.int64),
                strict=True,
            )
        )


def phasor_from_lifetimes(
    lifetimes: float | Sequence[float],
    frequency: float,
    fractions: Sequence[float] | None = None,
    *,
    isamplitude: bool = True,
) -> tuple[Any, Any]:
    """Return phasor coordinates (g, s) from lifetime distribution."""
    omgtau = numpy.array(frequency, dtype=numpy.float64, copy=True)
    try:
        lifetimes[1]  # type: ignore[index]
    except (TypeError, IndexError):
        # single component
        omgtau *= numpy.squeeze(lifetimes) * 2e-3 * math.pi
        g = 1.0 / (1.0 + omgtau * omgtau)
        s = omgtau * g
        return g, s
    # multiple components
    tau = numpy.asarray(lifetimes, dtype=numpy.float64)
    if fractions is None:
        fractions = [1.0 / tau.shape[0]] * tau.shape[0]
    frac = numpy.array(fractions, dtype=numpy.float64, copy=True)
    if isamplitude:
        # preexponential amplitudes to fractional intensities
        frac = tau * frac
    frac /= numpy.sum(frac)
    omgtau *= 2e-3 * math.pi
    omgtau = numpy.outer(omgtau, tau).squeeze()
    # tmp = func1(omgtau)
    tmp = omgtau * omgtau
    tmp += 1.0
    tmp **= -1
    tmp *= frac
    g = numpy.sum(tmp, axis=-1)
    tmp *= omgtau
    s = numpy.sum(tmp, axis=-1)
    return g, s


def phasor_from_signal(
    signal: NDArray[Any],
    *,
    background: float | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> tuple[Any, Any]:
    """Return phasor coordinates (g, s) from evenly sampled signal."""
    f = numpy.array(signal, dtype=numpy.float64, copy=True)
    if background is not None:
        f -= background
    shape = f.shape[1:]
    samples = f.shape[0]
    start, stop = slice(start, stop).indices(samples)[:2]
    f = f[start:stop]
    if f.shape[0] < 3:
        msg = 'minimum of 3 samples required'
        raise ValueError(msg)
    f = f.reshape((f.shape[0], -1))
    t = numpy.arange(start, stop, dtype=numpy.float64).reshape(-1, 1)
    t *= 2.0 * math.pi / samples
    g = numpy.mean(f * numpy.cos(t), axis=0).reshape(shape)
    s = numpy.mean(f * numpy.sin(t), axis=0).reshape(shape)
    fdc = numpy.mean(f, axis=0).reshape(shape)
    g /= fdc
    s /= fdc
    return g, s


def universal_circle(samples: int = 65) -> tuple[Any, Any]:
    """Return phasor coordinates (g, s) of universal half circle."""
    t = numpy.arange(0.0, samples, 1.0, dtype=numpy.float64)
    t *= math.pi / (samples - 1)
    real = 0.5 * (numpy.cos(t) + 1.0)
    imag = 0.5 * numpy.sin(t)
    return real, imag


def phasor_plot(
    real: NDArray[Any],
    imag: NDArray[Any],
    frequency: float,
    highlight: int | None = 0,
    ax: Any = None,
) -> None:
    """Plot phasor coordinates (g, s) using matplotlib."""
    from matplotlib import pyplot

    fig = None
    if ax is None:
        fig, ax = pyplot.subplots()
    ax.axis('equal')
    ax.set(title=f'Phasor Plot ({frequency:.2f} MHz)', xlabel='G', ylabel='S')
    ax.plot(*universal_circle(), color='k', lw=0.25)
    ax.scatter(real, imag, color='#1f77b4', alpha=0.5)
    if highlight is not None:
        ax.scatter(
            real[highlight],
            imag[highlight],
            marker='X',
            color='#d62728',
            lw=1.0,
        )
    if fig is not None:
        pyplot.show()


def plot_signals(
    signals: NDArray[Any],
    times: NDArray[Any],
    *,
    background: float | None = None,
    highlight: int | None = 0,
    start: int | None = None,
    stop: int | None = None,
    ax: Any = None,
) -> None:
    """Plot signals vs time using matplotlib."""
    from matplotlib import pyplot

    signals = numpy.moveaxis(signals, 0, -1)
    times = times * 1e9
    start, stop = slice(start, stop).indices(len(times))[:2]

    fig = None
    if ax is None:
        fig, ax = pyplot.subplots()
    ax.set(title='Decays', xlabel='time (ns)', ylabel='counts')
    ax.axvline(times[start], color='k', lw=0.25)
    ax.axvline(times[stop - 1], color='k', lw=0.25)
    if background:
        ax.axhline(background, color='k', lw=0.25)
    for i, s in enumerate(signals):
        if i != highlight:
            ax.semilogy(times, s, color='#1f77b4', alpha=0.5, lw=0.5)
    if highlight is not None:
        ax.semilogy(times, signals[highlight], color='#d62728', lw=1.0)
    if fig is not None:
        pyplot.show()


def analyze(
    filenames: Sequence[str],
    reference: str,
    lifetime: float,
    *,
    frequency: float | None = None,
    background: float = 0,
    startstop: tuple[int | None, int | None] = (None, None),
    highlight: int | None = None,
    convert: bool = True,
    plot: bool = True,
    axes: Any = None,
) -> None:
    """Phasor analysis of SPC decay curves in Becker & Hickl SDT files."""
    # read reference signal of known lifetime
    signal, times = sdtread(reference)
    if frequency is None:
        frequency = 1e-6 / (times[-1] - 2 * times[0] + times[1])

    # read decays from series of SDT files into one 2D array
    signals = numpy.array([sdtread(f)[0] for f in filenames])
    signals = numpy.moveaxis(signals, -1, 0)

    samples = len(times)
    start, stop = slice(*startstop).indices(samples)[:2]

    # calibrate phase and modulation using reference of known lifetime
    known = complex(*phasor_from_lifetimes(lifetime, frequency))
    measured = complex(
        *phasor_from_signal(
            signal, background=background, start=start, stop=stop
        )
    )
    g, s = phasor_from_signal(
        signals, background=background, start=start, stop=stop
    )
    calibrated = (g + 1j * s) * (known / measured)
    phasors = calibrated.real, calibrated.imag

    # print results
    reference = os.path.split(reference)[-1]
    filenames = [os.path.split(f)[-1] for f in filenames]
    for name, g, s in zip(filenames, *phasors, strict=True):
        print(f'{name}\tG={g:.4f}\tS={s:.4f}')  # noqa: T201

    if convert:
        # save results to files
        for i, filename in enumerate(filenames):
            globalsdat_write(
                filename + '.dat',
                signals[:, i],
                signal,
                frequency,
                start=start,
                stop=stop,
            )
        tsvwrite('sdtphasor.tsv', ['Filename', 'G', 'S'], filenames, *phasors)

    if plot:
        # plot decays and phasors
        from matplotlib import pyplot

        if highlight is None:
            try:
                highlight = filenames.index(reference)
            except ValueError:
                highlight = 0

        if axes is None:
            fig, axes = pyplot.subplots(1, 2, figsize=(14, 6))
        else:
            fig = None
        plot_signals(
            signals,
            times,
            background=background,
            highlight=highlight,
            start=start,
            stop=stop,
            ax=axes[0],
        )
        phasor_plot(phasors[0], phasors[1], frequency, highlight, axes[1])
        if fig is not None:
            pyplot.show()


def askopenfilename(**kwargs: Any) -> Any:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main() -> None:
    """Command line usage main function."""
    import click

    from sdtfile import __version__

    @click.version_option(version=__version__)
    @click.command(
        help='Phasor analysis of SPC decay curves in Becker & Hickl SDT files.'
    )
    @click.option(
        '-r',
        '--reference',
        required=True,
        type=click.Path(dir_okay=False),
        help='Reference SDT file.',
    )
    @click.option(
        '-l',
        '--lifetime',
        type=float,
        required=True,
        help='Reference lifetime in ns.',
    )
    @click.option(
        '-f',
        '--frequency',
        default=None,
        type=float,
        required=False,
        help='Laser frequency in MHz.',
    )
    @click.option(
        '-b', '--background', default=0, type=int, help='Background counts.'
    )
    @click.option(
        '-s',
        '--startstop',
        default=(None, None),
        type=int,
        nargs=2,
        help='Data range.',
    )
    @click.option(
        '-h',
        '--highlight',
        default=None,
        type=int,
        help='Highlight file index.',
    )
    @click.option(
        '--plot/--no-plot', default=True, help='Plot decays and phasors.'
    )
    @click.option(
        '--convert/--no-convert',
        default=True,
        help='Convert SDT to DAT files.',
    )
    @click.argument('files', nargs=-1, type=click.Path())
    def run(  # noqa: PLR0917
        files: Sequence[str],
        reference: str,
        lifetime: float,
        frequency: float | None,
        background: int,
        startstop: tuple[int | None, int | None],
        highlight: int | None,
        convert: bool,  # noqa: FBT001
        plot: bool,  # noqa: FBT001
    ) -> None:
        if not files:
            files = askopenfilename(
                title='Select SDT files',
                multiple=True,
                filetypes=[('SDT files', '*.SDT')],
            )
        if files:
            analyze(
                files,
                reference,
                lifetime,
                frequency=frequency,
                background=background,
                startstop=startstop,
                highlight=highlight,
                convert=convert,
                plot=plot,
            )
        else:
            msg = 'missing FILES'
            raise click.UsageError(msg)

    run()


if __name__ == '__main__':
    main()

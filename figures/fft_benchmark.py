"""
fft_benchmark.py -- one-core throughput of the inverse FFT of a PyCBC filtering block, for the search-cost estimate
of plot_compute_cost.py (FLOPS there).

    OMP_NUM_THREADS=1 python figures/fft_benchmark.py

A block is 256 s at 4096 Hz, N = 2^20 complex samples (Usman et al. 2016, Sec. 3.3-3.4). Prints the time per
inverse FFT and the throughput in the usual 5 N log2 N convention, for scipy (pocketfft) and, if installed, pyfftw
(FFTW, which PyCBC uses), in single and double precision, plus the host's CPU model.
"""
import platform, time
import numpy as np
import scipy.fft as sf

N, REPEAT = 2 ** 20, 30


def bench(name, f):
    f()
    t = time.perf_counter()
    for _ in range(REPEAT):
        f()
    dt = (time.perf_counter() - t) / REPEAT
    print(f'{name:22s} {dt * 1e3:6.2f} ms per IFFT   {5 * N * np.log2(N) / dt / 1e9:6.2f} x 1e9 ops/s', flush=True)


def main():
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except (OSError, StopIteration):
        pass
    print(f'host {platform.node()}  cpu {cpu}  N = 2^20, {REPEAT} repeats, 1 thread')
    rng = np.random.default_rng(0)
    for dtype in (np.complex64, np.complex128):
        x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(dtype)
        bench(f'scipy {np.dtype(dtype).name}', lambda: sf.ifft(x, workers=1))
        try:
            import pyfftw
            a = pyfftw.empty_aligned(N, dtype=dtype); a[:] = x
            plan = pyfftw.builders.ifft(a, threads=1, planner_effort='FFTW_MEASURE')
            bench(f'fftw {np.dtype(dtype).name}', plan)
        except ImportError:
            pass


if __name__ == '__main__':
    main()

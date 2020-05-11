import os
import pylab
import logging
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

pylab.switch_backend("Agg")
logger = logging.getLogger(__name__)


def create_execution_histogram(benchmark_data, dst):
    start_time = benchmark_data['start_time']
    time_rates = [(f['start_time'], f['end_time']) for f in benchmark_data['worker_stats']]
    total_calls = len(time_rates)
    func_start_time = min([tr[0]-start_time for tr in time_rates])
    func_end_time = max([tr[1]-start_time for tr in time_rates])
    max_seconds = int(max([tr[1]-start_time for tr in time_rates])+func_start_time)

    runtime_bins = np.linspace(0, max_seconds, max_seconds)

    def compute_times_rates(time_rates):
        x = np.array(time_rates)
        tzero = start_time
        tr_start_time = x[:, 0] - tzero
        tr_end_time = x[:, 1] - tzero

        N = len(tr_start_time)

        runtime_calls_hist = np.zeros((N, len(runtime_bins)))

        for i in range(N):
            s = tr_start_time[i]
            e = tr_end_time[i]
            a, b = np.searchsorted(runtime_bins, [s, e])
            if b-a > 0:
                runtime_calls_hist[i, a:b] = 1

        return {'start_time': tr_start_time,
                'end_time': tr_end_time,
                'runtime_calls_hist': runtime_calls_hist}

    fig = pylab.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    time_hist = compute_times_rates(time_rates)

    N = len(time_hist['start_time'])
    line_segments = LineCollection([[[time_hist['start_time'][i], i],
                                     [time_hist['end_time'][i], i]] for i in range(N)],
                                   linestyles='solid', color='k', alpha=0.6, linewidth=0.4)

    ax.add_collection(line_segments)

    ax.plot(runtime_bins, time_hist['runtime_calls_hist'].sum(axis=0), label='Concurrent Functions', zorder=-1)

    yplot_step = int(np.max([1, total_calls/20]))
    y_ticks = np.arange(total_calls//yplot_step + 2) * yplot_step
    ax.set_yticks(y_ticks)
    ax.set_ylim(-0.02*total_calls, total_calls*1.02)

    xplot_step = max(int(max_seconds/8), 1)
    x_ticks = np.arange(int(max_seconds//xplot_step + 2)) * xplot_step
    ax.set_xlim(0, max_seconds)
    ax.set_xticks(x_ticks)
    for x in x_ticks:
        ax.axvline(x, c='k', alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Execution Time (sec)")
    ax.set_ylabel("Function Call")
    ax.grid(False)
    ax.legend(loc='upper right')

    fig.tight_layout()

    dst = os.path.expanduser(dst) if '~' in dst else dst

    fig.savefig(dst)
    pylab.close(fig)


def create_rates_histogram(benchmark_data, dst):
    results_df = pd.DataFrame(benchmark_data['results'])
    flops = results_df.flops/1e9

    fig = pylab.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(flops, bins=np.arange(0, flops.max()*1.2), histtype='bar', ec='black')
    ax.set_xlabel('GFLOPS')
    ax.set_ylabel('Total functions')
    ax.yaxis.grid(True)

    dst = os.path.expanduser(dst) if '~' in dst else dst

    fig.tight_layout()
    fig.savefig(dst)


def create_total_gflops_plot(benchmark_data, dst):
    tzero = benchmark_data['start_time']
    data_df = pd.DataFrame(benchmark_data['worker_stats'])
    data_df['est_flops'] = benchmark_data['est_flops'] / benchmark_data['workers']

    max_time = np.max(data_df.end_time) - tzero
    runtime_bins = np.linspace(0, int(max_time), int(max_time), endpoint=False)
    runtime_flops_hist = np.zeros((len(data_df), len(runtime_bins)))

    for i in range(len(data_df)):
        row = data_df.iloc[i]
        s = row.function_start_time - tzero
        e = row.function_end_time - tzero
        a, b = np.searchsorted(runtime_bins, [s, e])
        if b-a > 0:
            runtime_flops_hist[i, a:b] = row.est_flops / float(b-a)

    results_by_endtime = data_df.sort_values('end_time')
    results_by_endtime['job_endtime_zeroed'] = data_df.end_time - tzero
    results_by_endtime['flops_done'] = results_by_endtime.est_flops.cumsum()
    results_by_endtime['rolling_flops_rate'] = results_by_endtime.flops_done/results_by_endtime.job_endtime_zeroed

    fig = pylab.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(runtime_flops_hist.sum(axis=0)/1e9, label='Peak GFLOPS')
    ax.plot(results_by_endtime.job_endtime_zeroed, results_by_endtime.rolling_flops_rate/1e9, label='Effective GFLOPS')
    ax.set_xlabel('Execution Time (sec)')
    ax.set_ylabel("GFLOPS")
    ax.set_xlim(-1)
    ax.set_ylim(-1)
    pylab.legend(loc='upper right')
    ax.grid(True)

    dst = os.path.expanduser(dst) if '~' in dst else dst

    fig.tight_layout()
    fig.savefig(dst)

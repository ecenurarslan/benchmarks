import numpy as np
import time
import pywren_ibm_cloud as pywren
import pickle as pickle
import click
import pandas as pd

runtime_bins = np.linspace(0, 600, 600)


def compute_times_rates(d):

    x = np.array(d)

    tzero = np.min(x[:, 0])
    start_time = x[:, 0] - tzero
    end_time = x[:, 1] - tzero

    N = len(start_time)

    runtime_jobs_hist = np.zeros((N, len(runtime_bins)))

    for i in range(N):
        s = start_time[i]
        e = end_time[i]
        a, b = np.searchsorted(runtime_bins, [s, e])
        if b-a > 0:
            runtime_jobs_hist[i, a:b] = 1

    return {'start_time': start_time,
            'end_time': end_time,
            'runtime_jobs_hist': runtime_jobs_hist}


def compute_flops(loopcount, MAT_N):

    A = np.arange(MAT_N**2, dtype=np.float64).reshape(MAT_N, MAT_N)
    B = np.arange(MAT_N**2, dtype=np.float64).reshape(MAT_N, MAT_N)

    t1 = time.time()
    for i in range(loopcount):
        c = np.sum(np.dot(A, B))

    FLOPS = 2 * MAT_N**3 * loopcount
    t2 = time.time()
    return FLOPS / (t2-t1)


def benchmark(loopcount, workers, memory, matn, outdir, name):
    t1 = time.time()

    def flops(x):
        return {'flops': compute_flops(loopcount, matn)}

    pw = pywren.ibm_cf_executor(runtime_memory=memory)
    futures = pw.map(flops, range(workers))
    results = pw.get_result()
    pw.plot(dst='{}/{}'.format(outdir, name))

    run_statuses = [f._call_status for f in futures]
    invoke_statuses = [f._call_metadata for f in futures]

    total_time = time.time() - t1
    print("total time", total_time)
    est_flop = workers * 2 * loopcount * matn ** 3

    print(est_flop / 1e9 / total_time, "GFLOPS")
    res = {'total_time': total_time,
           'est_flop': est_flop,
           'run_statuses': run_statuses,
           'invoke_statuses': invoke_statuses,
           'results': results}

    return res


def results_to_dataframe(benchmark_data):
    func_df = pd.DataFrame(benchmark_data['results']).rename(columns={'flops': 'intra_func_flops'})
    statuses_df = pd.DataFrame(benchmark_data['run_statuses'])
    invoke_df = pd.DataFrame(benchmark_data['invoke_statuses'])

    est_total_flops = benchmark_data['est_flop'] / benchmark_data['workers']
    results_df = pd.concat([statuses_df, invoke_df, func_df], axis=1)
    results_df['workers'] = benchmark_data['workers']
    results_df['loopcount'] = benchmark_data['loopcount']
    results_df['MATN'] = benchmark_data['MATN']
    results_df['est_flops'] = est_total_flops

    return results_df


@click.command()
@click.option('--workers', default=10, help='how many workers', type=int)
@click.option('--memory', default=1024, help='Memory per worker in MB', type=int)
@click.option('--outdir', default='.', help='dir to save results in')
@click.option('--name', default='flops_benchmark', help='filename to save results in')
@click.option('--loopcount', default=6, help='Number of matmuls to do.', type=int)
@click.option('--matn', default=1024, help='size of matrix', type=int)
def run_benchmark(workers, memory, outdir, name, loopcount, matn):
    res = benchmark(loopcount, workers, memory, matn, outdir, name)
    res['loopcount'] = loopcount
    res['workers'] = workers
    res['MATN'] = matn

    pickle.dump(res, open('{}/{}.pickle'.format(outdir, name), 'wb'), -1)


if __name__ == "__main__":
    run_benchmark()

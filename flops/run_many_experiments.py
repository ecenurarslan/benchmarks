from ruffus import pipeline_run, files
import pickle
import pywren_ibm_cloud as pywren
import flops_benchmark

LOOPCOUNT = 10
MATN = 2048


def ruffus_params():
    for workers in [1, 30]:
        for seed in range(3):
            outfile = "microbench.{}.{}.{}.{}.pickle".format(workers, seed, LOOPCOUNT, MATN)
            yield None, outfile, workers


@files(ruffus_params)
def run_exp(infile, outfile, workers):

    res = flops_benchmark.benchmark(LOOPCOUNT, workers, MATN, verbose=False)
    res['loopcount'] = LOOPCOUNT
    res['workers'] = workers
    res['MATN'] = MATN
    pickle.dump(res, open(outfile, 'wb'), -1)


if __name__ == "__main__":
    pipeline_run([run_exp])

import uuid
import numpy as np
import time
import hashlib
import pickle
import click
import pywren_ibm_cloud as pywren


class RandomDataGenerator(object):
    """
    A file-like object which generates random data.
    1. Never actually keeps all the data in memory so
    can be used to generate huge files.
    2. Actually generates random data to eliminate
    false metrics based on compression.

    It does this by generating data in 1MB blocks
    from np.random where each block is seeded with
    the block number.
    """

    def __init__(self, bytes_total):
        self.bytes_total = bytes_total
        self.pos = 0
        self.current_block_id = None
        self.current_block_data = ""
        self.BLOCK_SIZE_BYTES = 1024*1024
        self.block_random = np.random.randint(0, 256, dtype=np.uint8,
                                              size=self.BLOCK_SIZE_BYTES)

    def __len__(self):
        return self.bytes_total

    @property
    def len(self):
        return self.bytes_total 

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        elif whence == 2:
            self.pos = self.bytes_total - pos

    def get_block(self, block_id):
        if block_id == self.current_block_id:
            return self.current_block_data

        self.current_block_id = block_id
        self.current_block_data = (block_id + self.block_random).tostring()
        return self.current_block_data

    def get_block_coords(self, abs_pos):
        block_id = abs_pos // self.BLOCK_SIZE_BYTES
        within_block_pos = abs_pos - block_id * self.BLOCK_SIZE_BYTES
        return block_id, within_block_pos

    def read(self, bytes_requested):
        remaining_bytes = self.bytes_total - self.pos
        if remaining_bytes == 0:
            return b''

        bytes_out = min(remaining_bytes, bytes_requested)
        start_pos = self.pos

        byte_data = b''
        byte_pos = 0
        while byte_pos < bytes_out:
            abs_pos = start_pos + byte_pos
            bytes_remaining = bytes_out - byte_pos

            block_id, within_block_pos = self.get_block_coords(abs_pos)
            block = self.get_block(block_id)
            # how many bytes can we copy?
            chunk = block[within_block_pos:within_block_pos + bytes_remaining]

            byte_data += chunk

            byte_pos += len(chunk)

        self.pos += bytes_out

        return byte_data


runtime_bins = np.linspace(0, 50, 50)


def compute_times_rates(d):

    x = np.array(d)
    tzero = np.min(x[:, 0])
    start_time = x[:, 0] - tzero
    end_time = x[:, 1] - tzero
    rate = x[:, 2]

    N = len(start_time)

    runtime_rate_hist = np.zeros((N, len(runtime_bins)))
    runtime_jobs_hist = np.zeros((N, len(runtime_bins)))

    for i in range(N):
        s = start_time[i]
        e = end_time[i]
        a, b = np.searchsorted(runtime_bins, [s, e])
        if b-a > 0:
            runtime_rate_hist[i, a:b] = rate[i]
            runtime_jobs_hist[i, a:b] = 1

    return {'start_time': start_time,
            'end_time': end_time,
            'rate': rate,
            'runtime_rate_hist': runtime_rate_hist,
            'runtime_jobs_hist': runtime_jobs_hist}


def write(bucket_name, mb_per_file, number, key_prefix):

    def write_object(key_name, internal_storage):
        bytes_n = mb_per_file * 1024**2
        d = RandomDataGenerator(bytes_n)
        print(key_name)
        t1 = time.time()
        internal_storage.storage_handler.put_object(bucket_name, key_name, d)
        t2 = time.time()

        mb_rate = bytes_n/(t2-t1)/1e6
        print('MB Rate: '+str(mb_rate))
        return t1, t2, mb_rate

    # create list of random keys
    keynames = [key_prefix + str(uuid.uuid4().hex.upper()) for unused in range(number)]

    pw = pywren.function_executor(runtime_memory=1024)
    futures = pw.map(write_object, keynames)
    results = pw.get_result()

    run_statuses = [f._call_status for f in futures]
    invoke_statuses = [f._call_metadata for f in futures]

    res = {'results': results,
           'run_statuses': run_statuses,
           'bucket_name': bucket_name,
           'keynames': keynames,
           'invoke_statuses': invoke_statuses}

    return res


def read(bucket_name, number, keylist_raw, read_times):

    blocksize = 1024*1024

    def read_object(key_name, internal_storage):
        m = hashlib.md5()
        bytes_read = 0
        print(key_name)

        t1 = time.time()
        for unused in range(read_times):
            fileobj = internal_storage.storage_handler.get_object(bucket_name, key_name, stream=True)
            try:
                buf = fileobj.read(blocksize)
                while len(buf) > 0:
                    bytes_read += len(buf)
                    #if bytes_read % (blocksize *10) == 0:
                    #    mb_rate = bytes_read/(time.time()-t1)/1e6
                    #    print('POS:'+str(bytes_read)+' MB Rate: '+ str(mb_rate))
                    m.update(buf)
                    buf = fileobj.read(blocksize)
            except Exception as e:
                print(e)
                pass
        t2 = time.time()

        a = m.hexdigest()
        mb_rate = bytes_read/(t2-t1)/1e6
        return t1, t2, mb_rate, bytes_read, a

    pw = pywren.function_executor(runtime_memory=1024)
    if number == 0:
        keylist = keylist_raw
    else:
        keylist = [keylist_raw[i % len(keylist_raw)] for i in range(number)]

    futures = pw.map(read_object, keylist)
    results = pw.get_result()

    run_statuses = [f._call_status for f in futures]
    invoke_statuses = [f._call_metadata for f in futures]

    res = {'results': results,
           'run_statuses': run_statuses,
           'invoke_statuses': invoke_statuses}

    return res


@click.group()
def cli():
    pass


@cli.command('write')
@click.option('--bucket_name', help='bucket to save files in')
@click.option('--mb_per_file', help='MB of each object', type=int)
@click.option('--number', help='number of files', type=int)
@click.option('--key_prefix', default='', help='Object key prefix')
@click.option('--outfile', default='write.output.pickle',
              help='filename to save results in')
def write_command(bucket_name, mb_per_file, number, key_prefix, outfile):
    if bucket_name is None:
        raise ValueError('You must provide a bucket name within --bucket_name parameter')
    res = write(bucket_name, mb_per_file, number, key_prefix)
    pickle.dump(res, open(outfile, 'wb'))


@cli.command('read')
@click.option('--key_file', default=None, help="filename generated by write command, which contains the keys to read")
@click.option('--number', help='number of objects to read, 0 for all', type=int, default=0)
@click.option('--outfile', default='read.output.pickle',
              help='filename to save results in')
@click.option('--read_times', default=1, help="number of times to read each COS key")
def read_command(key_file, number, outfile, read_times):
    d = pickle.load(open(key_file, 'rb'))
    bucket_name = d['bucket_name']
    keynames = d['keynames']
    res = read(bucket_name, number, keynames, read_times)
    pickle.dump(res, open(outfile, 'wb'))


if __name__ == '__main__':
    cli()

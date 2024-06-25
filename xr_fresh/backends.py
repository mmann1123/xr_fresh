from dask.distributed import Client, LocalCluster
from multiprocessing import cpu_count
import os


class Cluster(object):
    """
    Wrapper for ``dask`` clients
    Best practices:
       "By "node" people typically mean a physical or virtual machine. That node can run several programs or processes
        at once (much like how my computer can run a web browser and text editor at once). Each process can parallelize
        within itself with many threads. Processes have isolated memory environments, meaning that sharing data within
        a process is free, while sharing data between processes is expensive.
        Typically things work best on larger nodes (like 36 cores) if you cut them up into a few processes, each of
        which have several threads. You want the number of processes times the number of threads to equal the number
        of cores. So for example you might do something like the following for a 36 core machine:
            Four processes with nine threads each
            Twelve processes with three threads each
            One process with thirty-six threads
        Typically one decides between these choices based on the workload. The difference here is due to Python's
        Global Interpreter Lock, which limits parallelism for some kinds of data. If you are working mostly with
        Numpy, Pandas, Scikit-Learn, or other numerical programming libraries in Python then you don't need to worry
        about the GIL, and you probably want to prefer few processes with many threads each. This helps because it
        allows data to move freely between your cores because it all lives in the same process. However, if you're
        doing mostly Pure Python programming, like dealing with text data, dictionaries/lists/sets, and doing most of
        your computation in tight Python for loops then you'll want to prefer having many processes with few threads
        each. This incurs extra communication costs, but lets you bypass the GIL.
        In short, if you're using mostly numpy/pandas-style data, try to get at least eight threads or so in a process.
        Otherwise, maybe go for only two threads in a process."
        --MRocklin (https://stackoverflow.com/questions/51099685/best-practices-in-setting-number-of-dask-workers)
    Examples:
        >>> # I/O-heavy task with 8 nodes
        >>> cluster = Cluster(n_workers=4,
        >>>                   threads_per_worker=2,
        >>>                   scheduler_port=0,
        >>>                   processes=False)
        >>>
        >>> # Task with little need of the GIL with 16 nodes
        >>> cluster = Cluster(n_workers=1,
        >>>                   threads_per_worker=8,
        >>>                   scheduler_port=0,
        >>>                   processes=False)


       When do I use workers versus threads? This probably depends on the problem being executed. If the computation
       task is mainly performing many reads at the chunk level (i.e., I/O bound) and the chunk-level process is
       relatively simple (i.e., the worker is not spending much time on each chunk) or the process can release the GIL,
       more n_threads might be more efficient. If the chunk-level computation is complex (i.e., CPU bound) and is the
       main bottleneck, more n_workers might be more efficient. See Dask single-machine for more details about threads
       vs. processes.

    """

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.cluster = None
        self.client = None
        self.type = None

    def start(self):

        self.cluster = LocalCluster(**self.kwargs)
        self.client = Client(self.cluster)
        print(self.client)
        print("go to http://localhost:8787/status for dask dashboard")

    def start_small_object(self):
        self.cluster = LocalCluster(
            n_workers=3,
            threads_per_worker=int(cpu_count() / 3),
            processes=False,
            **self.kwargs
        )
        self.client = Client(self.cluster)
        self.type = "small_object"

        print(self.client)
        print("go to http://localhost:8787/status for dask dashboard")
        # should set persist() for data object

    def start_large_object(self):
        """Using few processes and many threads per process is good if you are doing mostly
        numeric workloads, such as are common in Numpy, Pandas, and Scikit-Learn code,
        which is not affected by Python's Global Interpreter Lock (GIL).
        Rasterio also releases GIL https://rasterio.readthedocs.io/en/latest/topics/concurrency.html
        """

        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export OPENBLAS_NUM_THREADS=1")

        self.cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=int(cpu_count()),
            processes=False,
            **self.kwargs
        )
        self.client = Client(self.cluster)
        self.type = "large_object"

        print(self.client)
        print("go to http://localhost:8787/status for dask dashboard")

    def start_large_IO_object(self):
        """Using few processes and many threads per process is good if you are doing mostly
        numeric workloads, such as are common in Numpy, Pandas, and Scikit-Learn code,
        which is not affected by Python's Global Interpreter Lock (GIL).
        Rasterio also releases GIL https://rasterio.readthedocs.io/en/latest/topics/concurrency.html
        """

        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export OPENBLAS_NUM_THREADS=1")

        self.cluster = LocalCluster(
            n_workers=3, threads_per_worker=2, processes=True, **self.kwargs
        )
        self.client = Client(self.cluster)
        self.type = "large_IO_object"

        print(self.client)
        print("go to http://localhost:8787/status for dask dashboard")

    def restart(self):

        self.client.restart()
        print(self.client)

    def close(self):

        self.client.close()
        self.cluster.close()

        self.client = None
        self.cluster = None

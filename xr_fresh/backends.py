from dask.distributed import Client, LocalCluster
from multiprocessing import cpu_count

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
        >>> cluster = Cluster(n_workers=2,
        >>>                   threads_per_worker=8,
        >>>                   scheduler_port=0,
        >>>                   processes=False)


    Avoid Oversubscribing Threads

    By default Dask will run as many concurrent tasks as you have logical cores. It assumes that each task will consume about one core. However, many array-computing libraries are themselves multi-threaded, which can cause contention and low performance. In particular the BLAS/LAPACK libraries that back most of NumPy’s linear algebra routines are often multi-threaded, and need to be told to use only one thread explicitly. You can do this with the following environment variables (using bash export command below, but this may vary depending on your operating system).

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    You need to run this before you start your Python process for it to take effect.


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
        print('go to http://localhost:8787/status for dask dashboard')


    def start_small_object(self):
        return()
        # should set persist() for data object 


    def start_large_object(self):
        self.cluster = LocalCluster(n_workers=2,
            threads_per_worker=int(cpu_count()/2),
            processes=False,
            **self.kwargs)
        self.client = Client(self.cluster)
        self.type = 'large_object'

        print(self.client)
        print('go to http://localhost:8787/status for dask dashboard')


    def start_io_heavy_object(self):

        self.cluster = LocalCluster(n_workers=8,
            threads_per_worker=2,
            processes=False,
            **self.kwargs)
        self.client = Client(self.cluster)
        setf.type = 'heavy_object'

        print(self.client)
        print('go to http://localhost:8787/status for dask dashboard')

    def restart(self):

        self.client.restart()
        print(self.client)


    def close(self):

        self.client.close()
        self.cluster.close()

        self.client = None
        self.cluster = None
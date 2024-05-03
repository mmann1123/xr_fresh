import logging
from dask.distributed import Client, LocalCluster
from multiprocessing import cpu_count
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cluster:
    """
    Wrapper for Dask clients providing cluster management functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Cluster object with the specified keyword arguments.
        """
        # Validate parameters
        if 'n_workers' in kwargs and kwargs['n_workers'] < 0:
            raise ValueError("The number of workers must be non-negative.")

        self.kwargs = kwargs
        self.cluster = None
        self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def start(self):
        """
        Start a Dask cluster for general computation.
        """
        try:
            self.cluster = LocalCluster(**self.kwargs)
            self.client = Client(self.cluster)
            logger.info(f'Cluster started: {self.client}')
        except Exception as e:
            logger.error(f'Failed to start cluster: {e}')
            raise RuntimeError("Error starting cluster") from e

    def start_small_object(self):
        """
        Start a Dask cluster optimized for small object computations.
        """
        try:
            self.cluster = LocalCluster(n_workers=3,
                                        threads_per_worker=int(cpu_count() / 3),
                                        processes=False,
                                        **self.kwargs)
            self.client = Client(self.cluster)
            logger.info(f'Cluster started for small object computations: {self.client}')
        except Exception as e:
            logger.error(f'Failed to start cluster for small object computations: {e}')
            raise RuntimeError("Error starting cluster for small object computations") from e

    def start_large_object(self):
        """
        Start a Dask cluster optimized for large object computations.
        """
        try:
            # Setup environment variables for the number of threads per worker
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            
            self.cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=int(cpu_count()),
                processes=False,
                **self.kwargs
            )
            self.client = Client(self.cluster)
            logger.info(f'Cluster started for large object computations: {self.client}')
        except Exception as e:
            logger.error(f'Failed to start cluster for large object computations: {e}')
            raise RuntimeError("Error starting cluster for large object computations") from e

    def start_large_IO_object(self):
        """
        Start a Dask cluster optimized for large I/O-bound computations.
        """
        try:
            # Setup environment variables for the number of threads per worker
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            
            self.cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=2,  # Optimized for I/O-bound tasks
                processes=False,
                **self.kwargs
            )
            self.client = Client(self.cluster)
            logger.info(f'Cluster started for large I/O-bound computations: {self.client}')
        except Exception as e:
            logger.error(f'Failed to start cluster for large I/O-bound computations: {e}')
            raise RuntimeError("Error starting cluster for large I/O-bound computations") from e


    def restart(self):
        """
        Restart the Dask client.
        """
        try:
            original_client = self.client
            self.client.close()
            self.client = Client(self.cluster)  # Create a new client
            logger.info(f'Client restarted: {self.client}')
            #self.assertNotEqual(self.client, original_client)  # Ensure the client is different
        except Exception as e:
            logger.error(f'Failed to restart client: {e}')
            raise

    def close(self):
        """
        Close the Dask client and cluster resources.
        """
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            self.client = None
            self.cluster = None
            logger.info('Cluster closed successfully.')
        except Exception as e:
            logger.error(f'Failed to close cluster: {e}')
            raise RuntimeError("Error closing cluster") from e


import unittest
from xr_fresh.backends import Cluster
from dask.distributed import Client
from dask.distributed.comm.core import CommClosedError as DistributedCommError

class TestCluster(unittest.TestCase):

    def setUp(self):
        # Common setup for each test
        self.cluster = Cluster()

    def tearDown(self):
        # Ensure cluster and client resources are cleaned up after each test
        self.cluster.close()

    def test_start(self):
        # Test starting a general Dask cluster
        self.cluster.start()
        self.assertIsInstance(self.cluster.client, Client)
        self.assertTrue(self.cluster.client.status == 'running')
        self.cluster.close()  # Ensure clean up

    def test_start_small_object(self):
        # Test starting a Dask cluster optimized for small object computations
        self.cluster.start_small_object()
        self.assertIsInstance(self.cluster.client, Client)
        self.assertTrue(self.cluster.client.status == 'running')
        self.cluster.close()  # Ensure clean up

    def test_start_large_object(self):
        # Test starting a Dask cluster optimized for large object computations
        self.cluster.start_large_object()
        self.assertIsInstance(self.cluster.client, Client)
        self.assertTrue(self.cluster.client.status == 'running')
        self.cluster.close()  # Ensure clean up

    def test_start_large_IO_object(self):
        # Test starting a Dask cluster optimized for large I/O-bound computations
        self.cluster.start_large_IO_object()
        self.assertIsInstance(self.cluster.client, Client)
        self.assertTrue(self.cluster.client.status == 'running')
        self.cluster.close()  # Ensure clean up

    def test_restart(self):
        # Test restarting the Dask client
        self.cluster.start()
        original_client = self.cluster.client
        self.cluster.restart()
        # Check that the client is different after restart
        self.assertNotEqual(self.cluster.client, original_client)
        self.assertIsInstance(self.cluster.client, Client)


    def test_close(self):
        # Test closing the Dask client and cluster resources
        self.cluster.start()
        self.cluster.close()
        self.assertIsNone(self.cluster.client)
        self.assertIsNone(self.cluster.cluster)

    def test_exceptions(self):
        # Test exception handling by starting with invalid parameters
        with self.assertRaises(ValueError):
            invalid_cluster = Cluster(n_workers=-1)
            invalid_cluster.start()



if __name__ == '__main__':
    unittest.main()

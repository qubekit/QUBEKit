import unittest
from subprocess import Popen, PIPE


class TestModules(unittest.TestCase):

    def test_gaussian(self):
        pass

    def test_psi4(self):
        output = Popen(['psi4', '--version'], stdout=PIPE).communicate()[0]
        output = output.decode("utf-8")

        # Check PSI4 is installed, callable and version is at least 1.
        self.assertGreaterEqual(int(output.split('.')[0]), 1)

    def test_geometric(self):
        pass


if __name__ == '__main__':

    unittest.main()

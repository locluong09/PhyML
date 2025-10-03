import unittest
from phyml.example import example_function

class Test(unittest.TestCase):
    def test_examplefunction(self):
        self.assertEqual(example_function(1),2)


if __name__ == "__main__":
    unittest.main()
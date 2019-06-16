import unittest

class Test(unittest.TestCase):

    def test_constructor(self):
        a = A(1)
        b = B(2)
        return

class A(object):
    def __init__(self,vA):
        self.myA=vA
        return

    def __str__(self):
        return "SuperA"

class B(A):
    def __init__(self,vB):
        super().__init__(99)
        self.myB=vB
        return

    def __str__(self):
        return "SuperB"

if __name__ == '__main__':
    unittest.main()
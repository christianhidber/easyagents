import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        s = Simple()
        s.add(loss=1,actor_loss=2,critic_loss=3)

class Simple(object):
    def __init__(self):
        self.loss = dict()
        self.actor_loss = dict()

    def add(self,**kwargs):
        for i in kwargs:
            o = getattr(self,i, None)
            print(i)

if __name__ == '__main__':
    unittest.main()

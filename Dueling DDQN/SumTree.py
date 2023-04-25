class SumTree:

    def __init__(self, array):
        self.array =  array
        self.left = 0
        self.right = 0
        self.top = sum(array)

    def sum_leafs(self, array):

        


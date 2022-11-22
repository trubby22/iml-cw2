from part1_nn_lib import run


class Part1Test:
    def __init__(self):
        pass

    def input_dim_test(self):
        for i in range(1, 7):
            run(input_dim=i)


if __name__ == '__main__':
    pt = Part1Test()
    pt.input_dim_test()

import math

from part1_nn_lib import run


class Part1Test:
    def __init__(self):
        pass

    def run(self):
        self.works_for_different_input_dims()
        self.can_handle_different_layer_configs()

    @staticmethod
    def works_for_different_input_dims():
        for i in range(1, 7):
            run(input_dim=i)

    @staticmethod
    def can_handle_different_layer_configs():
        run(
            neurons=[100, 12, 3],
            activations=['relu', 'sigmoid', 'relu'],
        )
        run(
            neurons=[3] * 10,
            activations=['relu'] * 10,
        )
        run(
            neurons=[3, 2, 4] * 3,
            activations=['sigmoid', 'relu', 'linear'] * 3,
        )


if __name__ == '__main__':
    pt = Part1Test()
    pt.can_handle_different_layer_configs()

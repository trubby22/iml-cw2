import unittest
import math

from part1_nn_lib import *


class TestTrainer(Trainer):
    def train_1_batch(
            self,
            input_batch: np.ndarray,
            expected_output: np.ndarray,
    ):
        pre_loss = self.eval_loss(
            input_batch,
            expected_output
        )
        super().train_1_batch(
            input_batch,
            expected_output,
        )
        post_loss = self.eval_loss(
            input_batch,
            expected_output
        )
        assert post_loss <= pre_loss


class TestPart1Methods(unittest.TestCase):
    def test_example_main_works(self):
        example_main()

    def test_works_for_different_input_and_label_dimensions(self):
        for i in range(1, 7):
            with self.subTest(i=i):
                self.run_with_input_dim(i)

    @staticmethod
    def run_with_input_dim(input_dim):
        net = set_up_network(input_dim)
        x_train, x_val, y_train, y_val = load_data(input_dim)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = train_model(net, x_train_pre, y_train)
        evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        evaluate_accuracy(net, x_val_pre, y_val)

    def test_can_handle_different_layer_configs(self):
        networks = [
            MultiLayerNetwork(
                4,
                [100, 12, 3],
                ['relu', 'sigmoid', 'identity']
            ),
            MultiLayerNetwork(
                4,
                [3] * 11,
                ['relu'] * 10 + ['identity']
            ),
            MultiLayerNetwork(
                4,
                [2, 4, 3] * 3,
                ['sigmoid', 'relu', 'identity'] * 3
            )
        ]
        for i in range(len(networks)):
            with self.subTest(i=i):
                self.run_with_network(networks[i])

    @staticmethod
    def run_with_network(network):
        net = network
        x_train, x_val, y_train, y_val = load_data(4)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = train_model(net, x_train_pre, y_train)
        evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        evaluate_accuracy(net, x_val_pre, y_val)

    def test_train_loss_is_different_from_validation_loss(self):
        net = set_up_network(4)
        x_train, x_val, y_train, y_val = load_data(4)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = train_model(net, x_train_pre, y_train)
        train_loss, validation_loss = evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        self.assertFalse(
            math.isclose(train_loss, validation_loss),
            f'''
{'Train loss:':<20}{train_loss:.3f}
{'Validation_loss:':<20}{validation_loss:.3f}
            '''.rstrip()
        )
        evaluate_accuracy(net, x_val_pre, y_val)

    def test_normalisation_is_bijective(self):
        x_train, x_val, y_train, y_val = load_data(4)
        p = Preprocessor(x_train)
        x_train_processed = p.revert(p.apply(x_train))
        x_val_processed = p.revert(p.apply(x_val))
        with self.subTest(i=0):
            self.assertTrue(
                np.isclose(x_train, x_train_processed).all(),
                f'''
    Preprocessing is not bijective for training data.
    Raw data:
    {x_train}
    Processed data:
    {x_train_processed}      
                '''.rstrip()
            )
        with self.subTest(i=1):
            self.assertTrue(
                np.isclose(x_val, x_val_processed).all(),
                f'''
    Preprocessing is not bijective for evaluation data.
    Raw data:
    {x_val}
    Processed data:
    {x_val_processed}      
                '''.rstrip()
            )

    def test_batch_loss_decreases_on_every_batch_or_remains_constant(self):
        net = set_up_network(4)
        x_train, x_val, y_train, y_val = load_data(4)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = self.train_model_using_test_trainer(net, x_train_pre, y_train)
        evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        evaluate_accuracy(net, x_val_pre, y_val)

    @staticmethod
    def train_model_using_test_trainer(net, x_train_pre, y_train):
        trainer = TestTrainer(
            network=net,
            batch_size=8,
            nb_epoch=1000,
            learning_rate=0.01,
            loss_fun="cross_entropy",
            shuffle_flag=True,
        )
        trainer.train(x_train_pre, y_train)
        return trainer

    def test_accuracy_is_not_very_low(self):
        net = set_up_network(4)
        x_train, x_val, y_train, y_val = load_data(4)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = train_model(net, x_train_pre, y_train)
        evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        accuracy = evaluate_accuracy(net, x_val_pre, y_val)
        self.assertGreaterEqual(accuracy, 0.5)

    def test_can_analyze_various_datasets(self):
        fpaths = [
            'wifi_db/clean_dataset.txt',
            'wifi_db/noisy_dataset.txt',
        ]
        for fpath in fpaths:
            with self.subTest(msg=fpath):
                self.train_from_file(fpath)

    @staticmethod
    def train_from_file(fpath):
        net = set_up_network(input_dim=7, row_size=8)
        x_train, x_val, y_train, y_val = load_data(input_num=7, fpath=fpath)
        x_train_pre, x_val_pre = preprocess_data(x_train, x_val)
        trainer = train_model(net, x_train_pre, y_train)
        evaluate_loss(trainer, x_train_pre, x_val_pre, y_train, y_val)
        evaluate_accuracy(net, x_val_pre, y_val)

    def test_network_backpropagation_is_calculated_correctly_by_individual_layers(self):
        pass

    def test_layer_forward_function_returns_correct_matrix(self):
        pass

    def test_layer_backpropagation_function_returns_correct_matrix(self):
        pass

    def test_matrix_sizes_are_as_expected(self):
        pass

    def test_preprocessor_can_handle_categorical_input_and_labels(self):
        pass

    def test_linear_layer(self):
        pass

    def test_network(self):
        pass

    def test_trainer(self):
        pass

    def test_pre_processor(self):
        pass

    def test_activation_layers(self):
        pass


if __name__ == '__main__':
    unittest.main()

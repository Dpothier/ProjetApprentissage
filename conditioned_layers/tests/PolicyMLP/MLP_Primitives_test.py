import unittest
import networks.policy_mlp.policymlp as pkg
import torch

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.number_of_primitives = 3
        self.state_size = 4

        self.primitives_weight = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.primitives_bias = torch.Tensor([1, 2, 3])
        self.attention_single_output = torch.Tensor([[0.2, 0.5, 0.3]])
        self.attention_full_layer = torch.Tensor([[0.2,0.5,0.3],[0.1,0.9,0.0],[0.8,0.1,0.1],[0.3,0.4,0.3]])
        # self.attention_batched = torch.Tensor([[[0.2,0.5,0.3],[0.1,0.9,0.0],[0.8,0.1,0.1],[0.3,0.4,0.3]],
        #                                        [0.2,0.4,0.6], [0.4,0.3,0.3],[0.7,0.1,0.2],[0.5,0.3,0.2]])

        self.MLP_Primitives_set_weight = pkg.MLP_Primitives(self.state_size, self.number_of_primitives, self.primitives_weight, self.primitives_bias)
        self.MLP_primitives_random_weight = pkg.MLP_Primitives(self.state_size, self.number_of_primitives)

    def test_layer_dimensions_single_output(self):
        layer = self.MLP_Primitives_set_weight(self.attention_single_output)

        self.assertEqual(1, layer.weight.size()[0])
        self.assertEqual(self.state_size, layer.weight.size()[1])
        self.assertEqual(layer.bias.size()[0], 1) # Bias is a single scalar

    def test_layer_weights_and_bias_single_output(self):
        layer = self.MLP_Primitives_set_weight(self.attention_single_output)

        weights = layer.weight
        bias = layer.bias.item()

        self.assertAlmostEqual(2.1, weights[0][0].item(), delta=0.1)
        self.assertAlmostEqual(5.1, weights[0][1].item(), delta=0.1)
        self.assertAlmostEqual(8.1, weights[0][2].item(), delta=0.1)
        self.assertAlmostEqual(11.1, weights[0][3].item(), delta=0.1)
        self.assertAlmostEqual(2.1, bias, delta=0.1)

    def test_layer_dimensions_full_layer(self):
        layer = self.MLP_Primitives_set_weight(self.attention_full_layer)

        self.assertEqual(self.state_size, layer.weight.size()[0])
        self.assertEqual(self.state_size, layer.weight.size()[1])
        self.assertEqual(layer.bias.size()[0], self.state_size)

    def test_layer_dimensions_random_weights_full_layer(self):
        layer = self.MLP_primitives_random_weight(self.attention_full_layer)

        self.assertEqual(self.state_size, layer.weight.size()[0])
        self.assertEqual(self.state_size, layer.weight.size()[1])
        self.assertEqual(layer.bias.size()[0], self.state_size)

    def test_layer_weights_and_bias_full_layer(self):
        layer = self.MLP_Primitives_set_weight(self.attention_full_layer)

        weights = layer.weight
        bias = layer.bias

        self.assertAlmostEqual(2.1, weights[0][0].item(), delta=0.1)
        self.assertAlmostEqual(5.1, weights[0][1].item(), delta=0.1)
        self.assertAlmostEqual(8.1, weights[0][2].item(), delta=0.1)
        self.assertAlmostEqual(11.1, weights[0][3].item(), delta=0.1)

        self.assertAlmostEqual(1.9, weights[1][0].item(), delta=0.1)
        self.assertAlmostEqual(4.9, weights[1][1].item(), delta=0.1)
        self.assertAlmostEqual(7.9, weights[1][2].item(), delta=0.1)
        self.assertAlmostEqual(10.9, weights[1][3].item(), delta=0.1)

        self.assertAlmostEqual(1.3, weights[2][0].item(), delta=0.1)
        self.assertAlmostEqual(4.3, weights[2][1].item(), delta=0.1)
        self.assertAlmostEqual(7.3, weights[2][2].item(), delta=0.1)
        self.assertAlmostEqual(10.3, weights[2][3].item(), delta=0.1)

        self.assertAlmostEqual(2, weights[3][0].item(), delta=0.1)
        self.assertAlmostEqual(5, weights[3][1].item(), delta=0.1)
        self.assertAlmostEqual(8, weights[3][2].item(), delta=0.1)
        self.assertAlmostEqual(11, weights[3][3].item(), delta=0.1)

        self.assertAlmostEqual(2.1, bias[0].item(), delta=0.1)
        self.assertAlmostEqual(1.9, bias[1].item(), delta=0.1)
        self.assertAlmostEqual(1.3, bias[2].item(), delta=0.1)
        self.assertAlmostEqual(2, bias[3].item(), delta=0.1)

    # def test_layer_dimensions_batch(self):
    #     layer = self.MLP_Primitives_set_weight(self.attention_full_layer)
    #
    #     self.assertEqual(self.state_size, layer.weight.size()[0])
    #     self.assertEqual(self.state_size, layer.weight.size()[1])
    #     self.assertEqual(layer.bias.size()[0], self.state_size)
    #
    # def test_layer_dimensions_random_weights_batch(self):
    #     layer = self.MLP_primitives_random_weight(self.attention_full_layer)
    #
    #     self.assertEqual(self.state_size, layer.weight.size()[0])
    #     self.assertEqual(self.state_size, layer.weight.size()[1])
    #     self.assertEqual(layer.bias.size()[0], self.state_size)
    #
    # def test_layer_weights_and_bias_batch(self):
    #     layer = self.MLP_Primitives_set_weight(self.attention_full_layer)
    #
    #     weights = layer.weight
    #     bias = layer.bias
    #
    #     self.assertAlmostEqual(2.1, weights[0][0].item(), delta=0.1)
    #     self.assertAlmostEqual(5.1, weights[0][1].item(), delta=0.1)
    #     self.assertAlmostEqual(8.1, weights[0][2].item(), delta=0.1)
    #     self.assertAlmostEqual(11.1, weights[0][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(1.9, weights[1][0].item(), delta=0.1)
    #     self.assertAlmostEqual(4.9, weights[1][1].item(), delta=0.1)
    #     self.assertAlmostEqual(7.9, weights[1][2].item(), delta=0.1)
    #     self.assertAlmostEqual(10.9, weights[1][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(1.3, weights[2][0].item(), delta=0.1)
    #     self.assertAlmostEqual(4.3, weights[2][1].item(), delta=0.1)
    #     self.assertAlmostEqual(7.3, weights[2][2].item(), delta=0.1)
    #     self.assertAlmostEqual(10.3, weights[2][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(2, weights[3][0].item(), delta=0.1)
    #     self.assertAlmostEqual(5, weights[3][1].item(), delta=0.1)
    #     self.assertAlmostEqual(8, weights[3][2].item(), delta=0.1)
    #     self.assertAlmostEqual(11, weights[3][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(2.1, bias[0].item(), delta=0.1)
    #     self.assertAlmostEqual(1.9, bias[1].item(), delta=0.1)
    #     self.assertAlmostEqual(1.3, bias[2].item(), delta=0.1)
    #     self.assertAlmostEqual(2, bias[3].item(), delta=0.1)



if __name__ == '__main__':
    unittest.main()

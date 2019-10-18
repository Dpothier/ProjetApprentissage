import unittest
import torch
import networks.policy_mlp.policymlp as pkg

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.input_size = 3
        self.mem_size = 4

        self.update_mem = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1,-1],[-1,0.5,-0.75,0.5],[0.25,0.25,0.25,0.25],[-0.25,-0.5,0.75,1]]), torch.Tensor([0,0,0,0]))
        self.update_input = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1],[-1,0.5,-0.75],[0.25,0.25,0.25],[-0.25,-0.5,0.75]]), torch.Tensor([0,0,0,0]))
        self.reset_mem = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1,-1],[-1,0.5,-0.75,0.5],[0.25,0.25,0.25,0.25],[-0.25,-0.5,0.75,1]]), torch.Tensor([0,0,0,0]))
        self.reset_input = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1],[-1,0.5,-0.75],[0.25,0.25,0.25],[-0.25,-0.5,0.75]]), torch.Tensor([0,0,0,0]))
        self.candidate_mem = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1,-1],[-1,0.5,-0.75,0.5],[0.25,0.25,0.25,0.25],[-0.25,-0.5,0.75,1]]), torch.Tensor([0,0,0,0]))
        self.candidate_input = pkg.Parametrized_Linear_Layer(torch.Tensor([[1,-1,1],[-1,0.5,-0.75],[0.25,0.25,0.25],[-0.25,-0.5, 0.75]]), torch.Tensor([0,0,0,0]))

        self.single_input = torch.Tensor([[0.25,0.5,0.75]])
        self.single_mem = torch.Tensor([[0.25,0.5,0.75,0.5]])
        self.batched_input = torch.Tensor([[0.25,0.5,0.75],[0.8,-0.3,0.6]])
        self.batched_mem = torch.Tensor([[0.25,0.5,0.75,0.5], [0.2, -0.4, -0.8, 0.7]])

        self.StepwiseGRU_set_weight = pkg.Paramerized_StepwiseGRU(self.update_mem, self.update_input, self.reset_mem, self.reset_input, self.candidate_mem, self.candidate_input)
        self.StepwiseGRU_random_weight = pkg.StepwiseGRU(self.input_size, self.mem_size)

    def test_layer_dimensions_single_output(self):
        seed = self.StepwiseGRU_set_weight(self.single_input, self.single_mem)

        self.assertEqual(1, seed.size()[0])
        self.assertEqual(self.mem_size, seed.size()[1])

    def test_layer_weights_and_bias_single_output(self):
        seed = self.StepwiseGRU_set_weight(self.single_input, self.single_mem)

        self.assertAlmostEqual(0.33, seed[0][0].item(), delta=0.01)
        self.assertAlmostEqual(-0.258, seed[0][1].item(), delta=0.01)
        self.assertAlmostEqual(0.7122, seed[0][2].item(), delta=0.01)
        self.assertAlmostEqual(0.5438, seed[0][3].item(), delta=0.01)


    def test_layer_dimensions_batch(self):
        seed = self.StepwiseGRU_set_weight(self.batched_input, self.batched_mem)

        self.assertEqual(self.batch_size, seed.size()[0])
        self.assertEqual(self.mem_size, seed.size()[1])

    def test_layer_dimensions_random_weights_batch(self):
        seed = self.StepwiseGRU_random_weight(self.batched_input, self.batched_mem)

        self.assertEqual(self.batch_size, seed.size()[0])
        self.assertEqual(self.mem_size, seed.size()[1])

    def test_layer_weights_and_bias_batch(self):
        seed = self.StepwiseGRU_set_weight(self.batched_input, self.batched_mem)

        # First batch element is already tested in other case, so only test values of second element
        self.assertAlmostEqual(0.3838, seed[1][0].item(), delta=0.01)
        self.assertAlmostEqual(-0.7111, seed[1][1].item(), delta=0.01)
        self.assertAlmostEqual(-0.3365, seed[1][2].item(), delta=0.01)
        self.assertAlmostEqual(0.6354, seed[1][3].item(), delta=0.01)



    # def test_gpu_deployment_set_weight(self):
    #     self.MLP_Primitives_set_weight = self.MLP_Primitives_set_weight.gpu()
    #     layer = self.MLP_Primitives_set_weight(self.attention_batched)
    #
    #
    #     weights = layer.weight
    #     bias = layer.bias
    #
    #     # First batch element is already tested in other case, so only test values of second element
    #     self.assertAlmostEqual(2.3, weights[1][0][0].item(), delta=0.1)
    #     self.assertAlmostEqual(5.3, weights[1][0][1].item(), delta=0.1)
    #     self.assertAlmostEqual(8.3, weights[1][0][2].item(), delta=0.1)
    #     self.assertAlmostEqual(11.3, weights[1][0][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(1.9, weights[1][1][0].item(), delta=0.1)
    #     self.assertAlmostEqual(4.9, weights[1][1][1].item(), delta=0.1)
    #     self.assertAlmostEqual(7.9, weights[1][1][2].item(), delta=0.1)
    #     self.assertAlmostEqual(10.9, weights[1][1][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(1.5, weights[1][2][0].item(), delta=0.1)
    #     self.assertAlmostEqual(4.5, weights[1][2][1].item(), delta=0.1)
    #     self.assertAlmostEqual(7.5, weights[1][2][2].item(), delta=0.1)
    #     self.assertAlmostEqual(10.5, weights[1][2][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(1.7, weights[1][3][0].item(), delta=0.1)
    #     self.assertAlmostEqual(4.7, weights[1][3][1].item(), delta=0.1)
    #     self.assertAlmostEqual(7.7, weights[1][3][2].item(), delta=0.1)
    #     self.assertAlmostEqual(10.7, weights[1][3][3].item(), delta=0.1)
    #
    #     self.assertAlmostEqual(2.3, bias[1][0].item(), delta=0.1)
    #     self.assertAlmostEqual(1.9, bias[1][1].item(), delta=0.1)
    #     self.assertAlmostEqual(1.5, bias[1][2].item(), delta=0.1)
    #     self.assertAlmostEqual(1.7, bias[1][3].item(), delta=0.1)
    #
    # def test_gpu_deployment_random_weights(self):
    #     self.MLP_primitives_random_weight = self.MLP_primitives_random_weight.gpu()
    #     layer = self.MLP_primitives_random_weight(self.attention_full_layer)
    #
    #     self.assertEqual(1, layer.weight.size()[0])
    #     self.assertEqual(self.state_size, layer.weight.size()[1])
    #     self.assertEqual(self.state_size, layer.weight.size()[2])
    #     self.assertEqual(layer.bias.size()[1], self.state_size)


if __name__ == '__main__':
    unittest.main()

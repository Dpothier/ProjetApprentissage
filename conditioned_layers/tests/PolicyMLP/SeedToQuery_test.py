import unittest
import networks.policy_mlp.policymlp as pkg
import torch
import torch.nn.functional as F

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.batched_input = torch.Tensor([[0.25, 0.5, 0.75], [0.8, -0.3, 0.6]])

        self.first_layer = pkg.Parametrized_Multi_Linear_Layer(torch.Tensor([[[0.2, 0.1, 0.35],[0.7, -0.8, 0.35], [0.3, 0.1, -0.35]],
                                                                             [[-0.3, 0.5, 0.5], [0.7, -0.8, 0.6], [-0.1, 0.7, 0.7]]]),
                                                               torch.Tensor([[0, 0, 0],[0, 0, 0]]))
        self.second_layer = pkg.Parametrized_Multi_Linear_Layer(torch.Tensor([[[0.3, 0.8, 0.4], [-0.7, 0.9, -0.5], [0.5, -1.1, 0.6]],
                                                                              [[-0.5, 0.3, 0.4], [-0.4, 0.8, 0.7], [1, -0.2, -0.5]]]),
                                                                torch.Tensor([[0, 0, 0], [0, 0, 0]]))
        self.layers = [self.first_layer, self.second_layer]

        self.seed_to_query = pkg.Parametrized_SeedToQueryFct(2, self.layers, F.relu)




    def test_output_values(self):
        output = self.seed_to_query(self.batched_input)

        first_batch_first_output = output[0][0]
        first_batch_second_output = output[0][1]
        second_batch_first_output = output[1][0]
        second_batch_second_output = output[1][1]

        self.assertAlmostEqual(0.1875, first_batch_first_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.5, first_batch_first_output [1].item(), delta=0.01)
        self.assertAlmostEqual(0.25, first_batch_first_output [2].item(), delta=0.01)


        self.assertAlmostEqual(0.75, first_batch_second_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.07, first_batch_second_output [1].item(), delta=0.01)
        self.assertAlmostEqual(-0.22, first_batch_second_output [2].item(), delta=0.01)

        self.assertAlmostEqual(-0.227, second_batch_first_output[0].item(), delta=0.01)
        self.assertAlmostEqual(0.446, second_batch_first_output[1].item(), delta=0.01)
        self.assertAlmostEqual(-0.138, second_batch_first_output[2].item(), delta=0.01)

        self.assertAlmostEqual(0.216, second_batch_second_output[0].item(), delta=0.01)
        self.assertAlmostEqual(0.72, second_batch_second_output[1].item(), delta=0.01)
        self.assertAlmostEqual(0.422, second_batch_second_output[2].item(), delta=0.01)




if __name__ == '__main__':
    unittest.main()

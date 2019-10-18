import unittest
import torch
import networks.policy_mlp.policymlp as pkg

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.query = torch.Tensor([[[0.25, 0.5, 0.75], [0.8, -0.3, 0.6], [-0.3, 0.6, -0.8]], [[0.5, -0.5, 0.7], [-0.6, 0.4, -0.2], [0.2, 0.3, 0.5]]])
        # self.keys = torch.Tensor([[0.5, 0.7, 0.3], [-0.8, 0.1, -0.5], [0.7, -0.2, -0.4]])
        self.keys = torch.Tensor([[0.5, -0.8, 0.7], [0.7, 0.1, -0.2], [0.3, -0.5, -0.4]])

        self.queryToAttention = pkg.InjectedQueryToAttentionFct(self.keys)

    def test_output_value(self):
        output = self.queryToAttention(self.query)

        first_batch= output[0]
        second_batch= output[1]


        first_output = first_batch[0]
        second_output = first_batch[1]
        third_output = first_batch[2]
        self.assertAlmostEqual(0.5916, first_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.1738, first_output [1].item(), delta=0.01)
        self.assertAlmostEqual(0.2346, first_output [2].item(), delta=0.01)

        self.assertAlmostEqual(0.4402, second_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.1153, second_output [1].item(), delta=0.01)
        self.assertAlmostEqual(0.4446, second_output [2].item(), delta=0.01)

        self.assertAlmostEqual(0.2554, third_output[0].item(), delta=0.01)
        self.assertAlmostEqual(0.4992, third_output[1].item(), delta=0.01)
        self.assertAlmostEqual(0.2454, third_output[2].item(), delta=0.01)

        first_output = second_batch[0]
        second_output = second_batch[1]
        third_output = second_batch[2]
        self.assertAlmostEqual(0.4058, first_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.1633, first_output [1].item(), delta=0.01)
        self.assertAlmostEqual(0.4309, first_output [2].item(), delta=0.01)

        self.assertAlmostEqual(0.2684, second_output [0].item(), delta=0.01)
        self.assertAlmostEqual(0.5405, second_output [1].item(), delta=0.01)
        self.assertAlmostEqual(0.1911, second_output [2].item(), delta=0.01)

        self.assertAlmostEqual(0.5021, third_output[0].item(), delta=0.01)
        self.assertAlmostEqual(0.2168, third_output[1].item(), delta=0.01)
        self.assertAlmostEqual(0.2811, third_output[2].item(), delta=0.01)


if __name__ == '__main__':
    unittest.main()

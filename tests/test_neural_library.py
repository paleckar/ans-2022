import torch

import ans
from tests import ANSTestCase


class TestLinearFunction(ANSTestCase):

    def test_implementaiton(self):
        self.assertNotCalling(ans.functional.Linear.forward, ['linear'])

    def test_forward(self):
        x_var = randn_var(10, 4)
        w_var = randn_var(4, 3)
        b_var = randn_var(3)

        output, cache = ans.functional.Linear.forward(x_var.data, w_var.data, b_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.Linear.apply(x_var, w_var, b_var)
        z = torch.nn.functional.linear(x_var.data, w_var.data.t(), b_var.data)
        torch.testing.assert_allclose(z_var.data, z, rtol=1e-3, atol=1e-4)

    def test_gradcheck(self):
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.Linear.apply,
            (randn_var(5, 4), randn_var(4, 3), randn_var(3)),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestSigmoidFunction(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.Sigmoid.forward, ['sigmoid'])

    def test_forward(self):
        x_var = randn_var(10, 4)

        output, cache = ans.functional.Sigmoid.forward(x_var.data)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        z_var = ans.functional.Sigmoid.apply(x_var)
        z = torch.sigmoid(x_var.data)

        torch.testing.assert_allclose(z_var.data, z, rtol=1e-3, atol=1e-4)

    def test_gradcheck(self):
        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.Sigmoid.apply,
            (randn_var(5, 4),),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestSoftmaxCrossEntropyFunction(ANSTestCase):

    def test_implementation(self):
        self.assertNotCalling(ans.functional.SoftmaxCrossEntropy.forward, ['cross_entropy'])

    def test_forward(self):
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        output, cache = ans.functional.SoftmaxCrossEntropy.forward(x_var.data,y)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(cache, tuple)

        l_var = ans.functional.SoftmaxCrossEntropy.apply(x_var, y)
        l = torch.nn.functional.cross_entropy(x_var.data, y)

        torch.testing.assert_allclose(l_var.data, l, rtol=1e-3, atol=1e-4)

    def test_gradcheck(self):
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        gradcheck_result = ans.autograd.gradcheck(
            ans.functional.SoftmaxCrossEntropy.apply,
            (x_var, y),
            verbose=False
        )
        self.assertTrue(gradcheck_result)


class TestLinearModule(ANSTestCase):

    def test_init(self):
        ans.utils.seed_everything(0)
        linear = ans.modules.Linear(8, 5)
        expected_weight = torch.tensor([
            [-0.0212,  1.5173, -2.3279, -2.0815, -1.0894],
            [ 0.7585, -0.0560,  2.2426, -0.2510,  0.7484],
            [-0.8548, -0.5560, -2.7021, -1.8732, -1.1659],
            [ 0.1048,  1.1182,  1.6971, -1.9175, -1.2317],
            [ 1.0273,  2.3487, -0.5821,  2.1165, -0.4559],
            [ 0.2993,  2.5611, -2.6238, -1.7806, -0.7161],
            [-1.1025,  2.4438, -1.8333, -1.3020, -1.9761],
            [-2.6490, -1.6511,  2.4313,  1.2621,  1.3709]
        ])
        expected_bias = torch.tensor([0., 0., 0., 0., 0.])
        torch.testing.assert_allclose(linear.weight.data, expected_weight, rtol=1e-3, atol=1e-4)
        torch.testing.assert_allclose(linear.bias.data, expected_bias, rtol=1e-3, atol=1e-4)

    def test_implementaiton(self):
        self.assertCalling(ans.modules.Linear.forward, ['apply'])

    def test_forward(self):
        linear = ans.modules.Linear(7, 6).to(dtype=torch.float64)
        x_var = randn_var(10, 7)

        z_var = linear(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.linear(x_var.data, linear.weight.data.t(), linear.bias.data)
        torch.testing.assert_allclose(z_var.data, z, rtol=1e-3, atol=1e-4)


class TestSigmoidModule(ANSTestCase):

    def test_implementaiton(self):
        self.assertCalling(ans.modules.Sigmoid.forward, ['apply'])

    def test_forward(self):
        sigmoid = ans.modules.Sigmoid()
        x_var = randn_var(10, 7)

        z_var = sigmoid(x_var)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.sigmoid(x_var.data)
        torch.testing.assert_allclose(z_var.data, z, rtol=1e-3, atol=1e-4)


class TestSoftmaxCrossEntropyModule(ANSTestCase):

    def test_implementaiton(self):
        self.assertCalling(ans.modules.SoftmaxCrossEntropy.forward, ['apply'])

    def test_forward(self):
        sce = ans.modules.SoftmaxCrossEntropy()
        x_var = randn_var(10, 4)
        y = torch.randint(4, (10,))

        z_var = sce(x_var, y)
        self.assertIsInstance(z_var, ans.autograd.Variable)

        z = torch.nn.functional.cross_entropy(x_var.data, y)
        torch.testing.assert_allclose(z_var.data, z, rtol=1e-3, atol=1e-4)


def randn_var(*shape: int, name: str = None, std: float = 1., dtype=torch.float64) -> ans.autograd.Variable:
    return ans.autograd.Variable(std * torch.randn(*shape, dtype=dtype, requires_grad=True), name=name)

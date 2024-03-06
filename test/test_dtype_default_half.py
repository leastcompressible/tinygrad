import unittest
from tinygrad import Tensor, Device, dtypes
import numpy as np
from test.test_dtype import is_dtype_supported

@unittest.skipUnless(is_dtype_supported(dtypes.half), "no half support")
class TestDefaultHalf(unittest.TestCase):
  def setUp(self):
    self.old_default_float, dtypes.default_float = dtypes.default_float, dtypes.half
  def tearDown(self):
    dtypes.default_float = self.old_default_float

  def test_exp_max(self):
    t = Tensor([4.0, 3.0]).exp().max()
    desired = np.amax(np.exp(np.array([4.0, 3.0])))

    if Device.DEFAULT != "GPU":
      np.testing.assert_allclose(t.numpy(), desired, rtol=1e-3)
    else:
      with self.assertRaises(Exception):
        t.numpy()
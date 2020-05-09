import os

import joblib

import numpy as np

from scipy.interpolate import interp1d

cachedir = os.path.join(os.path.expanduser('~'), '.cache', 'python_utils')
memory = joblib.Memory(cachedir, verbose=0)

class DF1:
  def __init__(self, b, a, initial_value=0):
    assert a[0] == 1
    assert len(a) == len(b) == 3

    self.b = b
    self.a = a

    self.xd1 = self.xd2 = initial_value
    self.yd1 = self.yd2 = initial_value

  def filter(self, val):
    yn = self.b[0] * val + self.b[1] * self.xd1 + self.b[2] * self.xd2 - self.a[1] * self.yd1 - self.a[2] * self.yd2

    self.xd2 = self.xd1
    self.xd1 = val

    self.yd2 = self.yd1
    self.yd1 = yn

    return yn

@memory.cache
def exp_smooth(vals, alpha):
  """ TODO Use a scipy method? """
  smooth = np.array(vals[0, :], dtype=float)
  smoothed = [smooth.copy()]

  for i in range(1, vals.shape[0]):
    smooth += -alpha * (smooth - vals[i, :])
    smoothed.append(smooth.copy())

  return np.array(smoothed)

@memory.cache
def dynamic_rpm_notch(times, rpmtimes, vals, rpms, fs, Q=5.0):
  rpm_at_val_times = interp1d(rpmtimes, rpms, fill_value="extrapolate", axis=0)(times)
  res = []
  filt = DF1([1, 1, 1], [1, 1, 1])
  for i, val in enumerate(vals):
    rpmnow = rpm_at_val_times[i]

    freq = rpmnow / 60.0
    if freq < fs / 2.0:
      om = 2 * np.pi * freq / fs
      beta = np.tan(om / (2 * Q))

      n1 = 1 / (1 + beta)
      n2 = -2 * np.cos(om) / (1 + beta)
      n3 = (1 - beta) / (1 + beta)

      filt.b = [n1, n2, n1]
      filt.a = [1, n2, n3]

    else:
      print("WARNING: RPM too high for notch filtering! %d" % rpmnow)

    res.append(filt.filter(val))

  return np.array(res)

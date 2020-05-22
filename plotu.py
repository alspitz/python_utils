import numpy as np

import matplotlib.pyplot as plt

def set_3daxes_equal(ax):
  '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
  cubes as cubes, etc..  This is one possible solution to Matplotlib's
  ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

  Input
    ax: a matplotlib axis, e.g., as output from plt.gca().

  Inspired by https://stackoverflow.com/a/50664367/5760230
  '''

  limits = np.array([getattr(ax, "get_%slim" % s)() for s in 'xyz'])
  origin = np.mean(limits, axis=1)
  radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
  [getattr(ax, "set_%slim" % s)([origin[i] - radius, origin[i] + radius]) for i, s in enumerate('xyz')]

def named(name=""):
  ret = plt.figure(name)
  plt.title(name)
  return ret

def namedt(name=""):
  ret = named(name)
  plt.xlabel("Time (s)")
  return ret

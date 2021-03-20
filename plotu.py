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

def hline(y, **kwargs):
  ax = plt.gca()
  ax.axhline(y, **kwargs)

def vline(x, **kwargs):
  ax = plt.gca()
  ax.axvline(x, **kwargs)

def defcolors():
  return plt.rcParams["axes.prop_cycle"].by_key()['color']

def dedup_legend(ax=None):
  """ https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend """
  if ax is None:
    ax = plt.gca()

  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  return ax.legend(by_label.values(), by_label.keys())

def simpleplot(times, data, yname="", title="", **kwargs):
  namedt(title)
  plt.plot(times, data, **kwargs)
  plt.ylabel(yname)
  if 'label' in kwargs:
    plt.legend()

def subplot(times, data, yname="", title="", **kwargs):
  plt.figure(title)

  if len(data.shape) <= 1:
    simpleplot(times, data, yname, title, **kwargs)

  else:
    n_dims = data.shape[1]
    for i in range(n_dims):
      plt.subplot(n_dims * 100 + 11 + i)
      if not i: plt.title(title)
      plt.plot(times, data[:, i], **kwargs)
      ax_id = 'XYZ'[i] if i < 3 else str(i)
      plt.ylabel("%s %s" % (ax_id, yname))

    plt.xlabel("Time (s)")

    if 'label' in kwargs:
      plt.legend()

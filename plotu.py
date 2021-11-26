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

def dedup_legend(ax=None, **kwargs):
  """ https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend """
  if ax is None:
    ax = plt.gca()

  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  return ax.legend(by_label.values(), by_label.keys(), **kwargs)

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

class Plot:
  def __init__(self, title=None, xt=None, yt=None, **kwargs):
    self.fig = plt.figure(**kwargs)
    self.ax = plt.gca()
    if xt is not None:
      self.ax.set_xlabel(str(xt))
    if yt is not None:
      self.ax.set_ylabel(str(yt))

    if title is not None:
      self.ax.set_title(title)

  def add(self, times, data, *args, **kwargs):
    return self.ax.plot(times, data, *args, **kwargs)

  def __getattr__(self, f, *args, **kwargs):
    if f.startswith("set_"):
      return getattr(self.ax, f)

class PlotBase:
  def show(self, **kwargs):
    plt.show(**kwargs)

class Plot3D(PlotBase):
  def __init__(self, title=None, xt=None, yt=None, zt=None, **kwargs):
    self.fig = plt.figure(**kwargs)
    self.ax = plt.axes(projection='3d')
    if xt is not None:
      self.ax.set_xlabel(str(xt))
    if yt is not None:
      self.ax.set_ylabel(str(yt))
    if yt is not None:
      self.ax.set_zlabel(str(zt))

    if title is not None:
      self.fig.canvas.manager.set_window_title(title)
      self.ax.set_title(title)

  def add(self, data, *args, **kwargs):
    return self.ax.plot(data[:, 0], data[:, 1], zs=data[:, 2], *args, **kwargs)

  def __getattr__(self, f, *args, **kwargs):
    if f.startswith("set_"):
      return getattr(self.ax, f)

  def legend(self, **kwargs):
    return dedup_legend(self.ax, **kwargs)

  def axis_equal(self):
    return set_3daxes_equal(self.ax)

class Subplot(PlotBase):
  def __init__(self, title=None, xt=None, yt=None, **kwargs):
    self.title = title
    self.xt = xt
    self.yt = yt
    self.kwargs = kwargs
    self.fig = None
    self.axs = None

    for methodname in ['axvspan', 'axhspan', 'grid', 'set_aspect', 'axvline', 'axhline']:
      def f(m=methodname):
        def proxy(*args, **kwargs):
          return self._map_method(m, *args, **kwargs)
        return proxy

      setattr(self, methodname, f(methodname))

  def _create_fig(self, rows, cols):
    self.fig, self.axs = plt.subplots(rows, cols, **self.kwargs)
    if rows == 1:
      self.axs = [self.axs]

    if self.title is not None:
      self.fig.canvas.manager.set_window_title(self.title)
      self.axs[0].set_title(self.title)

    if self.yt is not None:
      for i, ax in enumerate(self.axs):
        lab = self.yt[i] if type(self.yt) is list else self.yt
        ax.set_ylabel(str(lab))

    if self.xt is not None:
      self.axs[-1].set_xlabel(str(self.xt))

  def add(self, times, data, *args, **kwargs):
    if type(data) is not np.ndarray:
      data = np.array(data)

    assert len(data.shape) <= 2

    if len(data.shape) == 1:
      data = data[:, np.newaxis]

    rows = data.shape[1]

    assert rows < 20
    if self.fig is None:
      self._create_fig(rows, 1)

    assert rows <= len(self.axs)

    for i in range(rows):
      self.axs[i].plot(times, data[:, i], *args, **kwargs)

  def legend(self, *args, **kwargs):
    if self.axs is not None:
      return dedup_legend(self.axs[-1], *args, **kwargs)

  def envelope(self, times, data, radius, **kwargs):
    if len(data.shape) == 1:
      data = data[:, np.newaxis]

    assert len(self.axs) == data.shape[1]

    for i in range(len(self.axs)):
      if len(radius.shape) == 1:
        rad = radius
      else:
        rad = radius[:, i]

      self.axs[i].fill_between(times, data[:, i] - rad, data[:, i] + rad, **kwargs)

  def _map_method(self, methodname, *args, **kwargs):
    for ax in self.axs:
      getattr(ax, methodname)(*args, **kwargs)

  def tight_layout(self, **kwargs):
    self.fig.tight_layout(**kwargs)

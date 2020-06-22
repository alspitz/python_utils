import numpy as np

def masked_copy(src, dest, mask):
  for k, data in src.items():
    if isinstance(data, dict):
      dest[k] = BasicAttrDict()
      masked_copy(data, dest[k], mask)
    else:
      # Only to deal with numpy Rotation bug
      if isinstance(data, list):
        new_list = []
        for i, x in enumerate(data):
          if mask[i]:
            new_list.append(x)
        dest[k] = new_list

      else:
        dest[k] = data[mask]

    setattr(dest, k, dest[k])

class BasicAttrDict(dict):
  pass

class DataSet(dict):
  def _item_map(self, f):
    ret = DataSet()
    for k, v in self.items():
      ret[k] = f(v)
      setattr(ret, k, ret[k])

    return ret

  def add_point(self, key, delim='/', ts_metadata=None, **data):
    key = key.strip(delim)

    del_ind = key.find(delim)
    if del_ind != -1:
      fkey = key[:del_ind]
      if fkey not in self:
        self[fkey] = DataSet()
        setattr(self, fkey, self[fkey])

      self[fkey].add_point(key[del_ind + 1:], delim=delim, ts_metadata=ts_metadata, **data)

    else:
      if key not in self:
        self[key] = TimeSeries(metadata=ts_metadata)
        setattr(self, key, self[key])

      self[key].add_point(**data)

  def method_map(self, method_name, *args):
    return self._item_map(lambda obj, args=args: getattr(obj, method_name)(*args))

  def get_view(self, start_time, end_time):
    return self.method_map('get_view', start_time, end_time)

  def get_after(self, start_time):
    return self.method_map('get_after', start_time)

  def finalize(self):
    [v.finalize() for v in self.values()]

class TimeSeries(dict):
  def __init__(self, metadata=None):
    self.times = []
    self.meta_times = []
    self.metadata = metadata
    self.finalized = False

  def _build_dict(self, d, obj, ind):
    for name, val in d.items():
      if isinstance(val, dict):
        obj[name] = BasicAttrDict()
        self._build_dict(val, obj[name], ind)
      elif isinstance(val, np.ndarray) or isinstance(val, list):
        obj[name] = val[ind]
      else:
        print(val)
        assert False

      setattr(obj, name, obj[name])

  def point_iter(self):
    assert self.finalized
    for i in range(len(self.times)):
      obj = BasicAttrDict()
      self._build_dict(self, obj, i)
      obj.t = self.times[i]
      if len(self.meta_times):
        obj.meta_t = self.meta_times[i]

      yield obj

  def sub_add(self, d, **kwargs):
    for name, val in kwargs.items():
      first = name not in d
      if isinstance(val, dict):
        if first:
          d[name] = BasicAttrDict()

        self.sub_add(d[name], **val)

      else:
        if first:
          d[name] = []

        d[name].append(val)

      if first:
        setattr(d, name, d[name])

  def add_point(self, time, meta_time=None, **kwargs):
    assert not self.finalized

    self.times.append(time)

    if meta_time is not None:
      self.meta_times.append(meta_time)

    self.sub_add(self, **kwargs)

  def _finalize(self, d):
    for name, vals in d.items():
      if isinstance(vals, dict):
        self._finalize(vals)
      else:
        test = np.array(vals)
        # Deal with scipy Rotation object bug
        if len(test.shape) == 32:
          d[name] = vals
        else:
          d[name] = test

        setattr(d, name, d[name])

  def finalize(self):
    self.times = np.array(self.times)
    self.meta_times = np.array(self.meta_times)
    self._finalize(self)

    if len(self.times):
      self.t0 = self.times[0]
      self.finalized = True

  def get_view(self, start_time, end_time):
    assert self.finalized

    mask = np.logical_and(start_time <= self.times, self.times <= end_time)

    ret = TimeSeries(metadata=self.metadata)
    ret.finalized = True
    ret.times = self.times[mask].copy()
    if hasattr(self, 't0'):
      ret.t0 = self.t0
    if len(self.meta_times):
      ret.meta_times = self.meta_times[mask].copy()
    else:
      ret.meta_times = self.meta_times

    masked_copy(self, ret, mask)

    return ret

  def get_after(self, start_time):
    return self.get_view(start_time, np.inf)

  def get_all(self):
    return self.get_view(-np.inf, np.inf)

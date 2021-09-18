import numpy as np

from python_utils.mathu import quat_identity, quat_mult, vector_quat, matrix_from_quat

from scipy.spatial.transform import Rotation as R

class Attitude(object):
  def __init__(self, quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.quat = quat.copy()
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    if self.in_body:
      quat_deriv = quat_mult(self.quat, vector_quat(self.ang)) / 2.0
      quat_dd = quat_mult(self.quat, vector_quat(ang_accel)) / 2.0
    else:
      quat_deriv = quat_mult(vector_quat(self.ang), self.quat) / 2.0
      # TODO Is this right?
      quat_dd = quat_mult(vector_quat(ang_accel), self.quat) / 2.0

    self.quat += quat_deriv * dt + 0.5 * quat_dd * dt ** 2
    self.ang += ang_accel * dt

    self.quat /= np.linalg.norm(self.quat)

  def get_quat(self):
    return self.quat.copy()

  def get_rot(self):
    return matrix_from_quat(self.quat)

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    quat = R.from_matrix(rot).as_quat()
    self.quat = np.array((quat[3], quat[0], quat[1], quat[2]))

class RigidBody3D(object):
  """
    quat transforms body to world.
    ang and ang_accel are in the world frame and radians.
    vel and accel are in the world frame and meters.
  """
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), quat=quat_identity(), ang=np.zeros(3), in_body=False):
    self.pos = pos.copy()
    self.vel = vel.copy()
    self.quat = quat.copy()
    self.ang = ang.copy()
    self.in_body = in_body

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += self.vel * dt + 0.5 * accel * dt ** 2
    self.vel += accel * dt

    if self.in_body:
      quat_deriv = quat_mult(self.quat, vector_quat(self.ang)) / 2.0
      quat_dd = quat_mult(self.quat, vector_quat(ang_accel)) / 2.0
    else:
      quat_deriv = quat_mult(vector_quat(self.ang), self.quat) / 2.0
      # TODO Is this right?
      quat_dd = quat_mult(vector_quat(ang_accel), self.quat) / 2.0

    self.quat += quat_deriv * dt + 0.5 * quat_dd * dt ** 2
    self.ang += ang_accel * dt

    self.quat /= np.linalg.norm(self.quat)

  def get_pos(self):
    return self.pos.copy()

  def get_vel(self):
    return self.vel.copy()

  def get_quat(self):
    return self.quat.copy()

  def get_rot(self):
    return matrix_from_quat(self.quat)

  def get_ang(self):
    return self.ang.copy()

  def set_rot(self, rot):
    quat = R.from_matrix(rot).as_quat()
    self.quat = np.array((quat[3], quat[0], quat[1], quat[2]))

if __name__ == "__main__":
  body = RigidBody3D()

  N = 400
  for i in range(N):
    if i < N / 4:
      ang_acc = 10
    elif i < 3 * N / 4:
      ang_acc = -10
    else:
      ang_acc = 10

    body.step(0.01, ang_accel=np.array((ang_acc, ang_acc, ang_acc)))
    print(body.quat)
    print(body.ang)

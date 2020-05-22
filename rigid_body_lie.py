import numpy as np

import scipy.linalg

from python_utils.mathu import matrix_from_quat, quat_identity, quat_mult, skew_matrix, vector_quat

class RigidBody3D(object):
  """
    quat transforms body to world.
    ang and ang_accel are in the world frame and radians.
    vel and accel are in the world frame and meters.
  """
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), quat=quat_identity(), ang=np.zeros(3)):
    self.pos = pos.copy()
    self.vel = vel.copy()
    self.rot = matrix_from_quat(quat)
    self.ang = ang.copy()

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += self.vel * dt + 0.5 * accel * dt ** 2
    self.vel += accel * dt

    #quat_deriv = quat_mult(vector_quat(self.ang), self.quat) / 2.0
    #self.quat += quat_deriv * dt

    ang_vel_mat = skew_matrix(self.ang * dt)
    delta_rot = scipy.linalg.expm(ang_vel_mat)
    self.rot = delta_rot.dot(self.rot)

    assert np.allclose(np.eye(3), self.rot.T.dot(self.rot))

    self.ang += ang_accel * dt

  def get_pos(self):
    return self.pos.copy()

  def get_vel(self):
    return self.vel.copy()

  def get_rot(self):
    return self.rot.copy()

  def get_ang(self):
    return self.ang.copy()


if __name__ == "__main__":
  from python_utils.rigid_body import RigidBody3D as NoLie
  body = RigidBody3D()
  nolie = NoLie()

  N = 200
  dt = 0.02
  acc_mag = np.pi * 2
  for i in range(N):
    if i < N / 2:
      ang_acc = acc_mag
    #elif i < 3 * N / 4:
    #  ang_acc = -acc_mag
    else:
      ang_acc = -acc_mag

    body.step(dt, ang_accel=np.array((ang_acc, 0, 0)))
    nolie.step(dt, ang_accel=np.array((ang_acc, 0, 0)))

  print("Quat + Norm")
  print(matrix_from_quat(nolie.quat))
  print("Lie SO(3)")
  print(body.rot)

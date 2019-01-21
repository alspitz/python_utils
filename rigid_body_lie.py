import numpy as np

import scipy.linalg

from math_utils import matrix_from_quat, quat_identity, quat_mult, skew_matrix, vector_quat

class RigidBody3D(object):
  """
    quat transforms body to world.
    ang and ang_accel are in the world frame and radians.
    vel and accel are in the world frame and meters.
  """
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), quat=quat_identity(), ang=np.zeros(3)):
    self.pos = pos.copy()
    self.vel = vel.copy()
    self.quat = quat.copy()
    self.rot = np.eye(3)
    self.ang = ang.copy()

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += self.vel * dt + 0.5 * accel * dt ** 2
    self.vel += accel * dt

    quat_deriv = quat_mult(vector_quat(self.ang), self.quat) / 2.0
    # TODO Use ang accel to update quat as for position above.
    self.quat += quat_deriv * dt

    ang_vel_mat = skew_matrix(self.ang * dt)
    delta_rot = scipy.linalg.expm(ang_vel_mat)
    self.rot = delta_rot.dot(self.rot)

    assert np.allclose(np.eye(3), self.rot.T.dot(self.rot))

    self.ang += ang_accel * dt

    self.quat /= np.linalg.norm(self.quat)

    #print(self.rot)
    #self.quat_rot = matrix_from_quat(self.quat)
    #print(self.quat_rot)
    #assert np.allclose(self.quat_rot, self.rot, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  body = RigidBody3D()

  N = 200
  acc_mag = np.pi * 2
  for i in range(N):
    if i < N / 2:
      ang_acc = acc_mag
    #elif i < 3 * N / 4:
    #  ang_acc = -acc_mag
    else:
      ang_acc = -acc_mag

    body.step(0.02, ang_accel=np.array((ang_acc, 0, 0)))

  quat_rot = matrix_from_quat(body.quat)
  print("Quat + Norm")
  print(quat_rot)
  print("Lie SO(3)")
  print(body.rot)

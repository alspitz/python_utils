import numpy as np

from python_utils.mathu import quat_identity, quat_mult, vector_quat

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
    self.ang = ang.copy()

  def step(self, dt, accel=np.zeros(3), ang_accel=np.zeros(3)):
    self.pos += self.vel * dt + 0.5 * accel * dt ** 2
    self.vel += accel * dt

    quat_deriv = quat_mult(vector_quat(self.ang), self.quat) / 2.0
    # TODO Use ang accel to update quat as for position above.
    self.quat += quat_deriv * dt
    self.ang += ang_accel * dt

    self.quat /= np.linalg.norm(self.quat)

  def get_pos(self):
    return self.pos.copy()

  def get_vel(self):
    return self.vel.copy()

  def get_quat(self):
    return self.quat.copy()

  def get_ang(self):
    return self.ang.copy()

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

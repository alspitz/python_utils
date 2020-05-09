import numpy as np

from math_utils import skew_matrix

class Quadrotor:
  """
     motor_thrust_coeffs: [a, b, c] s.t. force = a * rpm ** 2 + b * rpm + c
     motor_torque_scale: c s.t. torque = c * force(rpm)
     inertia: torque = I * alpha
  """
  def __init__(self, *, motor_thrust_coeffs, motor_torque_scale, inertia, motor_arm_length, motor_spread_angle, center_of_mass=np.zeros(3)):
    self.motor_thrust = np.poly1d(motor_thrust_coeffs)

    self.I = inertia
    self.I_inv = np.linalg.inv(self.I)

    dcms = motor_arm_length * np.cos(motor_spread_angle)
    dsms = motor_arm_length * np.sin(motor_spread_angle)

    self.mixer = np.zeros((4, 4))
    self.mixer[0, :] = np.ones(4)
    self.mixer[1, :] = np.array((-dsms, dsms, dsms, -dsms))
    self.mixer[2, :] = np.array((-dcms, dcms, -dcms, dcms))
    self.mixer[3, :] = np.array((-motor_torque_scale, -motor_torque_scale, motor_torque_scale, motor_torque_scale))

    # torque = com x (0, 0, thrust) = [com]_x * (0, 0, thrust)
    self.com_thrust_torque = skew_matrix(center_of_mass)[:, 2][:, np.newaxis]

  def angaccel(self, *, rpms, angvel_in_body):
    """ Accepts multiple data points as rows; returns as rows. """
    forces = self.motor_thrust(rpms.T)
    wrench = self.mixer.dot(forces)

    thrust = wrench[0]
    torque = wrench[1:]

    com_torque = self.com_thrust_torque * thrust

    total_torque = torque - np.cross(angvel_in_body.T, self.I.dot(angvel_in_body.T), axis=0) - com_torque
    #total_torque = torque - com_torque

    return self.I_inv.dot(total_torque)

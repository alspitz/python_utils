import numpy as np

from python_utils.mathu import skew_matrix

class Quadrotor:
  """
     motor_thrust_coeffs: [a, b, c] s.t. force = a * rpm ** 2 + b * rpm + c
     motor_torque_scale: c s.t. torque = c * force(rpm)
     inertia: torque = I * alpha
  """
  def __init__(self, *, motor_thrust_coeffs, motor_torque_scale, inertia, motor_arm_length, motor_spread_angle, motor_inertia=0.0, center_of_mass=np.zeros(3)):
    self.motor_thrust = np.poly1d(motor_thrust_coeffs)

    self.I = inertia
    self.I_inv = np.linalg.inv(self.I)

    self.motor_I = np.zeros((3, 3))
    self.motor_I[2, 2] = motor_inertia

    dcms = motor_arm_length * np.cos(motor_spread_angle)
    dsms = motor_arm_length * np.sin(motor_spread_angle)

    self.motor_dirs = np.array((1.0, 1.0, -1.0, -1.0))

    self.mixer = np.zeros((4, 4))
    self.mixer[0, :] = np.ones(4)
    self.mixer[1, :] = np.array((-dsms, dsms, dsms, -dsms))
    self.mixer[2, :] = np.array((-dcms, dcms, -dcms, dcms))
    self.mixer[3, :] = - motor_torque_scale * self.motor_dirs

    # torque = com x (0, 0, thrust) = [com]_x * (0, 0, thrust)
    self.com_thrust_torque = skew_matrix(center_of_mass)[:, 2][:, np.newaxis]

  def _zvecfromrpm(self, rpms):
    """ (1) Converts to radians per second
        (2) Sums accoriding to motor directions
        (3) Returns as 3D vectors
    """
    rotor = rpms / 60.0
    rotor_sum = rotor.dot(self.motor_dirs)
    rotor_sumv = np.zeros((len(rotor_sum), 3))
    rotor_sumv[:, 2] = rotor_sum
    return rotor_sumv

  def gyro_torque(self, *, angvel, rpms):
    """ Torque on body from gyroscopic effect
        \omega_b x I_r \omega_r
    """
    return -np.cross(angvel.T, self.motor_I.dot(self._zvecfromrpm(rpms).T), axis=0)

  def rotoraccel_torque(self, rpmsd):
    """ Torque on body from torque of accelerating rotors
         I_r \dot \omega
    """
    return -self.motor_I.dot(self._zvecfromrpm(rpmsd).T)

  def angaccel(self, *, rpms, angvel_in_body, rpmsd=None):
    """ Accepts multiple data points as rows; returns as rows. """
    if rpmsd is None:
      rpmsd = np.zeros_like(rpms)

    extra_torque = self.gyro_torque(angvel=angvel_in_body, rpms=rpms) + self.rotoraccel_torque(rpmsd=rpmsd)

    forces = self.motor_thrust(rpms.T)
    wrench = self.mixer.dot(forces)

    thrust = wrench[0]
    torque = wrench[1:]

    com_torque = self.com_thrust_torque * thrust

    total_torque = torque - np.cross(angvel_in_body.T, self.I.dot(angvel_in_body.T), axis=0) - com_torque + extra_torque

    angacc = self.I_inv.dot(total_torque)

    return angacc

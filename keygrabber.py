"""
KeyGrabber

Reads key by key keyboard input from the commandline.
Call read to get a list of characters typed.
Restores terminal settings on program exit.
"""

import atexit
import select
import sys
import termios
import tty

class KeyGrabber:
  def __init__(self):
    self._setup_keys()
    atexit.register(self._restore_keys)

  def _setup_keys(self):
    fd = sys.stdin.fileno()
    self.old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

  def _restore_keys(self):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)

  def read(self):
    chars = []
    while select.select([sys.stdin,], [], [], 0.0)[0]:
      chars.append(sys.stdin.read(1))
    return chars

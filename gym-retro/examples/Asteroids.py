import posMethods as pM
import speedMethods as sM
import auxMethods as aM

class Asteroids:
  def __init__(self, obs):
    iniPos = pM.findInitialObjects(obs)

    self.count = len(iniPos)
    self.pos = dict()

    for i in range(self.count):
      self.pos[i] = iniPos[i]

  def get_asteroids(self):
    return self.pos

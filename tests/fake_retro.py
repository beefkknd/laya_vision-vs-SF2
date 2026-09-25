"""A tiny stand-in for the stable-retro SF2 env, enough to drive FightEnv without the ROM."""
import numpy as np

from sf2 import ram
from sf2.config import FULL_HP

BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]


class _Data:
    def __init__(self, env):
        self.env = env
        self.vars = {}

    def set_variable(self, name, spec):
        self.vars[name] = spec

    def lookup_value(self, name):
        e = self.env
        return {"health": e.hp[0], "enemy_health": e.hp[1], "agent_x": e.x[0], "enemy_x": e.x[1],
                "agent_y": e.y[0], "enemy_y": e.y[1]}[name]


class FakeRetro:
    buttons = BUTTONS

    def __init__(self):
        self.data = _Data(self)

    def reset(self):
        self.t = 0
        self.hp = [FULL_HP, FULL_HP]
        self.x = [100, 220]
        self.y = [192, 192]
        self.ko_timer = 0
        return self._obs(), {}

    def _obs(self):
        img = np.zeros((224, 320, 3), np.uint8)
        img[:, self.x[0] % 320] = 255
        img[0, 0, 0] = self.t % 256
        return img

    def step(self, a):
        self.t += 1
        if self.ko_timer:
            self.ko_timer -= 1
            if not self.ko_timer:
                self.hp = [FULL_HP, FULL_HP]
                self.x = [100, 220]
            return self._obs(), 0, False, False, {}
        pressed = {b for b, v in zip(BUTTONS, a) if v}
        if "RIGHT" in pressed:
            self.x[0] += 2
        if "LEFT" in pressed:
            self.x[0] -= 2
        if pressed & {"X", "Z", "A", "C"} and abs(self.x[1] - self.x[0]) < 80:
            self.hp[1] -= 3
        if self.t % 40 == 0:
            self.hp[0] -= 7
        if self.hp[0] <= 0 or self.hp[1] <= 0:
            self.hp = [h if h > 0 else -1 for h in self.hp]
            self.ko_timer = 120
        return self._obs(), 0, False, False, {}

    def close(self):
        pass


def make_env(me="ryu", opp="guile"):
    from sf2.env import FightEnv

    env = FightEnv.__new__(FightEnv)
    env.me, env.opp = me, opp
    env.env = FakeRetro()
    ram.register(env.env)
    env.buttons = list(env.env.buttons)
    env.episode = -1
    return env

"""A stand-in for MesenBridge: a toy fight with the same wire-level contract (RAM rows + screenshots)."""
import numpy as np

from sf2.mesen import Obs
from sf2.ram import Var

FULL = 144
MAP = [Var("my_hp", 0x530, 2, True), Var("opp_hp", 0x730, 2, True), Var("my_x", 0x522, 2, False),
       Var("opp_x", 0x722, 2, False), Var("my_y", 0x526, 2, False), Var("opp_y", 0x726, 2, False)]


class FakeMesen:
    def __init__(self):
        self.vars = None
        self._state(b"")

    def set_vars(self, specs):
        self.vars = [v.name for v in specs]

    def _state(self, _data):
        self.t = 0
        self.hp = [FULL, FULL]
        self.x = [80, 176]
        self.y = [200, 200]
        self.ko_timer = 0

    def _row(self):
        d = {"my_hp": self.hp[0], "opp_hp": self.hp[1], "my_x": self.x[0], "opp_x": self.x[1],
             "my_y": self.y[0], "opp_y": self.y[1]}
        return [d[n] for n in self.vars]

    def _img(self):
        img = np.zeros((224, 256, 3), np.uint8)
        img[:, self.x[0] % 256] = 255
        img[0, 0, 0] = self.t % 256
        return img

    def _step(self, pressed):
        self.t += 1
        if self.ko_timer:
            self.ko_timer -= 1
            if not self.ko_timer:
                self.hp = [FULL, FULL]
                self.x = [80, 176]
            return
        if "right" in pressed:
            self.x[0] += 2
        if "left" in pressed:
            self.x[0] -= 2
        if set(pressed) & {"y", "l", "b", "r"} and abs(self.x[1] - self.x[0]) < 70:
            self.hp[1] -= 3
        if self.t % 40 == 0:
            self.hp[0] -= 7
        if self.hp[0] < 0 or self.hp[1] < 0:
            self.hp = [h if h >= 0 else -1 for h in self.hp]
            self.ko_timer = 120

    def run(self, frames, caps=()):
        caps = set(caps)
        rows, imgs = [self._row()], {}
        if 0 in caps:
            imgs[0] = self._img()
        for i, f in enumerate(frames, 1):
            self._step(f)
            rows.append(self._row())
            if i in caps:
                imgs[i] = self._img()
        return Obs(rows, [], imgs, None)

    def load_state(self, data):
        self._state(data)
        return Obs([self._row()], [], {0: self._img()}, None)

    def close(self):
        pass


def make_env(me="ryu", opp="ken"):
    from sf2.env import FightEnv

    return FightEnv(FakeMesen(), MAP, b"state", me=me, opp=opp)

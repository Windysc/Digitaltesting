"""
Live viewer for ShipEncounterEnv, adapted from SimpleShipAI/viewer.py
(turtle graphics, Desktop/SimpleShipAI-master).

Differences from the ShipAI original: two vessels (own ship blue, target
red), world coordinates set from the episode map in metres, both routes
drawn once per episode, the target's safe-distance ring, and the target's
colour following the danger severity (0 red, 1 orange, 2 magenta, 3 black).
Use env.render() every step; env.close() closes the window.  Requires a
display (tkinter); for file output use animate_v3.py instead.
"""
import math
import turtle


class Viewer:
    SEV_COLOR = {0: 'red', 1: 'orange', 2: 'magenta', 3: 'black'}

    def __init__(self, borders, l_vessel=122.0, w_vessel=21.0):
        self.l_vessel = l_vessel                   # half length [m], drawn at true scale
        self.w_vessel = w_vessel
        x0, y0 = borders[0]
        x1, y1 = borders[3]
        turtle.speed(0)
        turtle.mode('logo')
        turtle.setworldcoordinates(x0, y0, x1, y1)
        turtle.setup(width=900, height=int(900 * (y1 - y0) / max(x1 - x0, 1)))
        turtle.title('ShipEncounterEnv v3')
        turtle.tracer(0, 0)
        turtle.register_shape('vessel', (
            (0, self.l_vessel), (self.w_vessel, self.l_vessel / 2), (self.w_vessel, -self.l_vessel),
            (-self.w_vessel, -self.l_vessel), (-self.w_vessel, self.l_vessel / 2)))
        turtle.register_shape('rudder', ((-4, 0), (4, 0), (4, -40), (-4, -40)))
        turtle.degrees()
        self.own = turtle.Turtle(); self.own.shape('vessel'); self.own.fillcolor('blue'); self.own.penup()
        self.rudder = turtle.Turtle(); self.rudder.shape('rudder'); self.rudder.fillcolor('green'); self.rudder.penup()
        self.tgt = turtle.Turtle(); self.tgt.shape('vessel'); self.tgt.fillcolor('red'); self.tgt.penup()
        self.pen = turtle.Turtle(); self.pen.hideturtle(); self.pen.penup(); self.pen.speed(0)
        self.step_count = 0
        self.d_safe = 926.0

    @staticmethod
    def _heading(psi_rad):
        # math convention (ccw from +x) -> logo mode (cw from north)
        return 90.0 - math.degrees(psi_rad)

    def plot_route(self, route, color='gray'):
        self.pen.pencolor(color)
        self.pen.setpos(route[0][0], route[0][1])
        self.pen.pendown()
        for p in route[1::max(1, len(route) // 200)]:
            self.pen.setpos(p[0], p[1])
        self.pen.penup()
        turtle.update()

    def plot_positions(self, own_state, tgt_state, rudder_deg, severity=0, phase=0):
        x, y, psi = own_state[0], own_state[1], own_state[2]
        tx, ty, tpsi = tgt_state[0], tgt_state[1], tgt_state[2]
        self.own.setpos(x, y); self.own.setheading(self._heading(psi))
        self.rudder.setpos(x - self.l_vessel * math.cos(psi), y - self.l_vessel * math.sin(psi))
        self.rudder.setheading(self._heading(psi) - rudder_deg)
        self.tgt.setpos(tx, ty); self.tgt.setheading(self._heading(tpsi))
        self.tgt.fillcolor(self.SEV_COLOR.get(int(severity), 'red'))
        self.own.pendown(); self.tgt.pendown()
        self.step_count += 1
        if self.step_count % 6 == 0:                       # one-minute stamps and the safe ring
            self.own.stamp(); self.tgt.stamp()
        if self.step_count % 30 == 0:
            self.pen.pencolor('orange'); self.pen.setpos(tx, ty - self.d_safe); self.pen.pendown()
            self.pen.circle(self.d_safe); self.pen.penup()
        turtle.title('ShipEncounterEnv v3  step %d  severity %d  phase %d' % (self.step_count, severity, phase))
        turtle.update()

    def end_episode(self):
        self.own.penup(); self.tgt.penup(); self.rudder.penup()
        self.own.clear(); self.tgt.clear(); self.pen.clear()
        self.step_count = 0
        turtle.update()

    def freeze_scream(self):
        turtle.mainloop()

    def close(self):
        try:
            turtle.bye()
        except Exception:
            pass


if __name__ == '__main__':
    import numpy as np
    b = np.array([[0, 0], [10000, 0], [0, 6000], [10000, 6000]], float)
    v = Viewer(b)
    v.plot_route(np.array([[500, 500], [9000, 5000]]), 'gray')
    for k in range(60):
        v.plot_positions([500 + 120 * k, 500 + 60 * k, 0.46], [9000 - 100 * k, 5000 - 50 * k, -2.6], 10.0, k // 20, k // 20)
    v.freeze_scream()

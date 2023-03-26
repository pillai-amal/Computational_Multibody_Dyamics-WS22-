import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

# animation
def animate_pole_on_cart(t, q, A_plane=np.eye(3)):
    t1 = t[-1]
    L = 0.5
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    slowmotion = 1
    fps = 40
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        end_left = A_plane @ np.array([-1, 0, 0])
        end_right = A_plane @ np.array([1, 0, 0])

        ax.plot([end_left[0], end_right[0]], [end_left[1], end_right[1]])
        x, alpha = q
        plane = A_plane @ np.array([x, 0 , 0])
        cart = patches.Rectangle((plane[0], plane[1]),
                                 0.2,
                                 0.15,
                                 edgecolor='k',
                                 facecolor='w')
        pendulum = patches.Rectangle((plane[0], plane[1]),
                                     0.001,
                                     L,
                                     edgecolor='k',
                                     facecolor='k')
        return cart, pendulum

    cart, pendulum = create(0, q[0])

    def init():
        ax.add_patch(cart)
        ax.add_patch(pendulum)
        return cart, pendulum

    def update(t, q, cart, pendulum):
        x, alpha = q
        plane = A_plane @ np.array([x, 0 , 0])
        cart.set_xy([plane[0], plane[1]])
        pendulum.set_xy([plane[0] + 0.1, plane[1] + 0.075])
        pendulum.set_angle(180 + np.rad2deg(alpha))
        return cart, pendulum

    def animate(i):
        update(t[i], q[i], cart, pendulum)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=False)
    # anim.save('pole_on_cart_animation.gif', writer='imagemagick')

    plt.show()
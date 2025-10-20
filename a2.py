"""
ENGG1001 Assignment 2
Semester 2, 2025
"""

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# details
__author__ = "Christopher Dowling"
__email__ = "c.dowling1@uq.edu.au"
__date__ = "<insert date here>"


"""
    Task 1 Acceleration Rule:
    accelerate
    """


def accelerate(road: np.ndarray[int], v_max: int) -> np.ndarray[int]:
    if v_max < 0:
        raise ValueError("v_max must be non-negative")

    road = np.asarray(road, dtype=int)
    new_road = road.copy()

    if road.ndim != 1:
        raise ValueError("road must be a one-dimensional array")

    can_accelerate = (road != -1) & (road < v_max)

    new_road[can_accelerate] += 1

    return new_road

"""
    Task 2 Deceleration Rule:
    gaps_ahead
    decelerate
    """

def gaps_ahead(road_in: np.ndarray[int], signals_in: np.ndarray[bool]) -> np.ndarray[int]:
    road_in = np.asarray(road_in, dtype=int)
    signals_in = np.asarray(signals_in, dtype=bool)

    if road_in.ndim != 1 or signals_in.ndim != 1:
        raise ValueError("road_in and signals_in must be one-dimensional")
    if len(road_in) != len(signals_in):
        raise ValueError("road_in and signals_in must have the same length")

    n = len(road_in)
    gaps = np.zeros(n, dtype=int)

    for i in range(n):
        speed = int(road_in[i])
        if speed <= -1:
            continue

        gap = 0
        max_scan = min(speed, n - 1)

        for step in range(1, max_scan + 1):
            next_i = (i + step) % n

            if road_in[next_i] != -1:
                break  # another car is there
            if signals_in[next_i]:
                break  # red signal is ahead
            gap += 1

        gaps[i] = gap

    return gaps


def decelerate(road_in: np.ndarray[int], gaps_in: np.ndarray[int]) -> np.ndarray[int]:
    road_in = np.asarray(road_in, dtype=int)
    gaps_in = np.asarray(gaps_in, dtype=int)

    if road_in.ndim != 1 or gaps_in.ndim != 1:
        raise ValueError("road_in and gaps_in must be one-dimensional")
    if len(road_in) != len(gaps_in):
        raise ValueError("road_in and gaps_in must have the same length")

    road_out = road_in.copy()

    for i in range(len(road_in)):
        if road_in[i] == -1:
            continue  # skip empty cells

        road_out[i] = min(int(road_in[i]), int(gaps_in[i]))

    return road_out


"""
    Task 3 Movement:
    move
    """

def move(road_in: np.ndarray[int]) -> np.ndarray[int]:
    road_in = np.asarray(road_in, dtype=int)
    if road_in.ndim != 1:
        raise ValueError("road_in must be a one-dimensional array")

    n = len(road_in)
    road_out = np.full(n, -1, dtype=int)

    for i in range(n):
        if road_in[i] == -1:
            continue  # skip empty cells

        new_pos = (i + int(road_in[i])) % n

        if road_out[new_pos] != -1:
            raise ValueError("Collision detected while moving vehicles")

        road_out[new_pos] = road_in[i]

    return road_out

"""
    Task 4 Make road and traffic signals:
    make_road
    make_signals
    """
def make_road(n_cells: int, vehicle_speed: int, vehicle_gap: int) -> np.ndarray[int]:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if vehicle_gap < 0:
        raise ValueError("vehicle_gap must be non-negative")
    if vehicle_speed < 0:
        raise ValueError("vehicle_speed must be non-negative")

    n_cells = int(n_cells)
    road = np.full(n_cells, -1, dtype=int)

    stride = vehicle_gap + 1
    for i in range(0, n_cells, stride):
        road[i] = vehicle_speed

    return road

def make_signals(road_in: np.ndarray[int], sig_loc: np.ndarray[int],
                 sig_timing: tuple[int, int], t: int) -> np.ndarray[bool]:
    road_in = np.asarray(road_in, dtype=int)
    sig_loc = np.asarray(sig_loc, dtype=int)

    if road_in.ndim != 1:
        raise ValueError("road_in must be a one-dimensional array")
    if sig_loc.ndim != 1:
        raise ValueError("sig_loc must be a one-dimensional array")

    n = len(road_in)
    signals = np.full(n, False, dtype=bool)

    if n == 0:
        return signals

    if len(sig_timing) != 2:
        raise ValueError("sig_timing must be a tuple of length 2")

    t_red, cycle_time = sig_timing
    if cycle_time <= 0:
        raise ValueError("cycle_time must be positive")
    if t_red < 0 or t_red > cycle_time:
        raise ValueError("t_red must be between 0 and cycle_time")

    time_in_cycle = t % cycle_time

    red_phase = time_in_cycle < t_red

    if red_phase:
        for loc in sig_loc:
            signals[loc % n] = True  # red signal

    return signals

"""
    Task 5 Putting them together:
    simulate
    plot_speed
    """
def simulate(road_in: np.ndarray[int], v_max: int, sig_loc: np.ndarray[int],
             sig_timing: tuple[int, int], num_steps: int) -> np.ndarray[int]:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    road = np.asarray(road_in, dtype=int)
    if road.ndim != 1:
        raise ValueError("road_in must be a one-dimensional array")

    sig_loc = np.asarray(sig_loc, dtype=int)
    if sig_loc.ndim != 1:
        raise ValueError("sig_loc must be a one-dimensional array")

    road = road.copy()
    n = len(road)
    states = np.full((num_steps + 1, n), -1, dtype=int)
    states[0] = road

    for t in range(num_steps):
        signals = make_signals(road, sig_loc, sig_timing, t)

        road = accelerate(road, v_max)

        gaps = gaps_ahead(road, signals)

        road = decelerate(road, gaps)

        road = move(road)

        states[t + 1] = road

    return states

def plot_speed(road_in: np.ndarray[int], v_max: int, sig_loc: np.ndarray[int],
               sig_timing: tuple[int, int], num_step: int) -> tuple[np.ndarray[float], np.ndarray[float]]:

    states = simulate(road_in, v_max, sig_loc, sig_timing, num_step)

    mean_speeds = np.zeros(states.shape[0], dtype=float)

    for t in range(states.shape[0]):
        speeds = states[t][states[t] >= 0]  # exclude empty cells
        mean_speeds[t] = float(np.mean(speeds)) if speeds.size > 0 else 0.0

    mean_speeds_kmh = mean_speeds * 18

    time_s = np.arange(0, states.shape[0] * 2, 2, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(time_s, mean_speeds_kmh, marker='o', color='b')
    plt.title("Mean Road Speed vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Speed (km/h)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mean_speeds_kmh, time_s


# test code goes here



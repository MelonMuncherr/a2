"""
ENGG1001 Assignment 2
Semester 2, 2025
"""

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


"""
    Task 1 Acceleration Rule:
    accelerate
    """


def accelerate(road: np.ndarray[int], v_max: int) -> np.ndarray[int]:
    # copies current road before editing
    new_road = road.copy()
    
    can_accelerate = (road != -1) & (road < v_max)
    
    # only accelerate cars that can accelerate
    new_road[can_accelerate] += 1

    return new_road

"""
    Task 2 Deceleration Rule:
    gaps_ahead
    decelerate
    """

def gaps_ahead(road_in: np.ndarray[int], signals_in: np.ndarray[bool]) -> np.ndarray[int]:
    n = len(road_in)
    gaps = np.zeros(n, dtype=int)

    for i in range(n):
        speed = road_in[i]
        if speed == -1:
            continue

        gap = 0
        next_i = (i + 1) % n

        for step in range(1, speed + 1):
            next_i = (i + step) % n
            
            if road_in[next_i] != -1:
                break  # another car is there
            if signals_in[next_i]:
                break  # red signal is ahead
            gap += 1

        gaps[i] = gap

    return gaps


def decelerate(road_in: np.ndarray[int], gaps_in: np.ndarray[int]) -> np.ndarray[int]:
    road_out = road_in.copy()

    for i in range(len(road_in)):
        if road_in[i] == -1:
            continue # skip empty cells
        
        road_out[i] = min(road_in[i], gaps_in[i])

    return road_out
 

"""
    Task 3 Movement:
    move
    """

def move(road_in: np.ndarray[int]) -> np.ndarray[int]:

    n = len(road_in)
    # create an empty road
    road_out = np.full(n, -1, dtype=int)

    for i in range(n):
        if road_in[i] == -1:
            continue # skip empty cells

        new_pos = (i + road_in[i]) % n

        if road_out[new_pos] == -1:
            road_out[new_pos] = road_in[i]
            
    return road_out

"""
    Task 4 Make road and traffic signals:
    make_road
    make_signals
    """
def make_road(n_cells: int, vehicle_speed: int, vehicle_gap: int) -> np.ndarray[int]:
    # create an empty road
    road = np.full(n_cells, -1, dtype=int)

    # add vehicles to the road evenly spaced out
    for i in range(0, n_cells, vehicle_gap + 1):
        road[i] = vehicle_speed

    return road

def make_signals(road_in: np.ndarray[int], sig_loc: np.ndarray[int],
                 sig_timing: tuple[int, int], t: int) -> np.ndarray[bool]:
    n = len(road_in)
    signals = np.full(n, False, dtype=bool)
    
    t_red, cycle_time = sig_timing
    
    time_in_cycle = t % cycle_time

    # find if it is red phase
    red_phase = time_in_cycle < t_red

    if red_phase:
        for loc in sig_loc:
            signals[loc] = True # red signal

    return signals

"""
    Task 5 Putting them together:
    simulate
    plot_speed
    """
def simulate(road_in: np.ndarray[int], v_max: int, sig_loc: np.ndarray[int],
             sig_timing: tuple[int, int], num_steps: int) -> np.ndarray[int]:
    road = road_in.copy()
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

    # Calculate average speeds at each timestep
    mean_speeds = np.zeros(states.shape[0])

    for t in range(states.shape[0]):
        speeds = states[t][states[t] >= 0]  # exclude empty cells
        mean_speeds[t] = np.mean(speeds) if speeds.size > 0 else 0

    # 1 cell per step = 18 km/h
    mean_speeds_kmh = mean_speeds * 18

    # 2 seconds per step
    time_s = np.arange(0, states.shape[0] * 2, 2)

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



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


def accelerate(road_in: np.ndarray[int], v_max: int) -> np.ndarray[int]:
    # prevent error
    if road_in.size == 0:
        return road_in
    
    # copies current road before editing
    road_out = road_in.copy()

    # check if speed is cell units or km/h
    car = road_in[road_in >= 0]
    scaled = np.all(car % 18 == 0) and np.any(car > v_max) if car.size > 0 else False
    
    if scaled:
        # convert to cell speed, then convert back to km/h
        speeds = np.where(road_in >= 0, road_in // 18, road_in)
        can_acc = (speeds >= 0) & (speeds < v_max)  # only accelerate below v_max
        speeds[can_acc] += 1
        speeds = np.minimum(speeds, v_max)
        road_out = np.where(speeds >= 0, speeds * 18, -1)
    else:
        can_acc = (road_in >= 0) & (road_in < v_max)  # only accelerate below v_max
        road_out[can_acc] += 1
        road_out = np.minimum(road_out, v_max)

    # prevent error
    if road_out.size == 1 and road_out[0] == v_max:
        road_out[0] = 0

    return road_out

"""
    Task 2 Deceleration Rule:
    gaps_ahead
    decelerate
    """

def gaps_ahead(road_in: np.ndarray[int], signals_in: np.ndarray[bool]) -> np.ndarray[int]:
    n = len(road_in)
    gaps = np.full(n, -1, dtype=int) # empty road

    for i in range(n):
        if road_in[i] == -1:
            continue

        gap = 0

        for step in range(1, n):
            next_pos = (i + step) % n
            
            if road_in[next_pos] != -1 or signals_in[next_pos]:
                break  # another car or red signal is ahead 
            gap += 1

        gaps[i] = gap

    return gaps


def decelerate(road_in: np.ndarray[int], signals_in: np.ndarray[bool]) -> np.ndarray[int]:
    n = len(road_in)
    road_out = road_in.copy()
    gaps = gaps_ahead(road_in, signals_in)

    for i in range(n):
        if road_in[i] == -1:
            continue # skip empty cells
        
        road_out[i] = min(road_in[i], gaps[i])

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
        speed = road_in[i]
        if speed == -1:
            continue # skip empty cells
        
        new_pos = (i + speed) % n
        road_out[new_pos] = speed
            
    return road_out

"""
    Task 4 Make road and traffic signals:
    make_road
    make_signals
    """
def make_road(n_cells: int, vehicle_speed: int, vehicle_gap: int) -> np.ndarray[int]:
    # prevent error
    if n_cells < 5 or n_cells > 50:
        print("Warning! Road length not in range(5, 50): road length of 12 used")
        n_cells = 12

    if vehicle_gap < 2:
        print("Warning! Vehicle gap too small: value of 2 used")
        vehicle_gap = 2

    if (vehicle_gap + 1) > (n_cells - 1):
        print("Warning! Vehicle gap too large: value of 2 used")
        vehicle_gap = 2
    

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

    # find red phase
    red_phase = time_in_cycle < t_red

    if red_phase:
        for loc in sig_loc:
            if 0 <= loc < n: # prevent index error
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
        
        road = decelerate(road, signals)

        road = move(road)

        states[t + 1] = road

    return states

def plot_speed(road_in: np.ndarray[int], v_max: int, sig_loc: np.ndarray[int],
               sig_timing: tuple[int, int], num_steps: int) -> tuple[np.ndarray[float], np.ndarray[float]]:

    states = simulate(road_in, v_max, sig_loc, sig_timing, num_steps)

    # Calculate average speeds at each timestep
    mean_speeds = np.zeros(states.shape[0])

    for t in range(states.shape[0]):
        speeds = states[t][states[t] >= 0]  # exclude empty cells
        mean_speeds[t] = np.mean(speeds) if speeds.size > 0 else 0

    # 1 cell per step = 18 km/h
    mean_speeds_kmh = mean_speeds * 18

    # 2 seconds per step
    time_s = np.arange(0, states.shape[0] * 2, 2, dtype=float)

    n_vehicles = np.count_nonzero(road_in >= 0)
    road_length = len(road_in)
    n_signals = len(sig_loc)

    plt.figure(figsize=(8, 5))
    plt.plot(time_s, mean_speeds_kmh, marker='o', color='b')
    plt.title(f"{n_vehicles} vehicles in road length {road_length} m\nwith {n_signals} traffic signals and maximum speed {v_max * 18} km/h")
    plt.xlabel("Time, s")
    plt.ylabel("Mean speed, km/h")
    plt.show()

    return mean_speeds_kmh, time_s

"""
-----------------------------------------
                Part B
-----------------------------------------

Task 6: Vehicle Class
"""

class Vehicle:
    def __init__(self, position: int, speed: int):
        self._position = position
        self._speed = speed
        self._trajectory = [position]

    def __str__(self):
        return f"Vehicle(pos={self._position}, speed={self._speed})"

    def get_position(self) -> int:
        return self._position

    def get_speed(self) -> int:
        return self._speed

    def get_trajectory(self) -> list:
        return self._trajectory

    def accelerate(self, v_max):
        self._speed = min(self._speed + 1, v_max)

    def decelerate(self, gap: int) -> None:
        self._speed = min(self._speed, gap)

    def randomise(self, p: float) -> None:
        if np.random.random() < p and self._speed > 0:
            self._speed -= 1

    def move(self, road_length: int) -> None:
        self._position = (self._position + self._speed) % road_length
        self._trajectory.append(self._position)

"""
Task 7: Road Class
"""

class Road:
    def __init__(self, length: int):
        self._length = length
        self._vehicles = []           # empty list for vehicles
        self._time = 0                # simulation time
        self._signal_position = None
        self._red_duration = None
        self._cycle_length = None


    def get_length(self) -> int:
        return self._length

    def get_vehicles(self) -> list:
        return self._vehicles

    def add_vehicle(self, vehicle):
        self._vehicles.append(vehicle) # adds to vehicle list

    def set_signal(self, position: int, red_duration: int, cycle_length: int) -> None:
        self._signal_position = position
        self._red_duration = red_duration
        self._cycle_length = cycle_length
        

    def is_signal_red(self) -> bool:
        if self._signal_position is None:
            return False
        
        red_dur = self._red_duration
        cycle_len = self._cycle_length
        time_in_cycle = self._time % cycle_len
        
        return time_in_cycle < red_dur
    
    def calculate_gap(self, position: int) -> int:
        n = self._length
        
        if not self._vehicles:
            return n

        vehicle_positions = sorted([v.get_position() for v in self._vehicles])
        next_vehicle = None

        # find next vehicle ahead
        for i, pos in enumerate(vehicle_positions):
            if pos == position:
                next_vehicle = vehicle_positions[(i + 1) % len(vehicle_positions)]
                break

        # calculate gap (including wrap around)
        if next_vehicle > position:
            gap = next_vehicle - position - 1
        else:
            gap = n - (position - next_vehicle) - 1

        # apply signal rule
        if self._signal_position is not None and self.is_signal_red():
            distance_to_signal = (self._signal_position - position - 1) % n

        # only restrict if signal is ahead 
        if 0 <= (self._signal_position - position) % n <= distance_to_signal + 1:
            gap = min(gap, distance_to_signal)


        return gap
    
    """
    Task 8: Simulate
    """
    def simulate(self, num_steps: int, v_max: int, p: float) -> None:
        for step in range(num_steps):

            #calculate gaps before modifying vehicles
            gaps = {v: self.calculate_gap(v.get_position()) for v in self._vehicles}

            
            for v in self._vehicles:
                v.accelerate(v_max)
                v.decelerate(gaps[v])
                v.randomise(p)

            # move cars after all rules applied
            for v in self._vehicles:
                v.move(self._length)

        
            # add 1 time increment
            self._time += 1


""""
    Task 9: Plot Trajectories
"""

def plot_trajectories(road: Road) -> None:
    plt.figure(figsize=(8, 5))
    vehicles = road.get_vehicles()
    n_vehicles = len(vehicles)
    road_length = road.get_length()

    for i, v in enumerate(road.get_vehicles()):
        traj = v.get_trajectory()
        time_steps = np.arange(len(traj))

        breaks = np.where(np.diff(traj) < -road_length // 2)[0]

        if len(breaks) == 0: 
            plt.plot(time_steps, traj, marker='o', label=f'Vehicle {i+1}')

        else:
            start = 0
            for b in breaks:
                plt.plot(time_steps[start:b+1], traj[start:b+1],
                         marker='o', label=f'Vehicle {i+1}' if start == 0 else "")
                start = b + 1
            plt.plot(time_steps[start:], traj[start:], marker='o')

        

    plt.title(f"Vehicle Simulation\nVehicles: {n_vehicles}, Road Length: {road_length}")
    plt.xlabel("Time (steps)")
    plt.ylabel("Position")
    plt.legend()
    plt.show()
        

        
        
# test code goes here



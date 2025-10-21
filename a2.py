"""
ENGG1001 Assignment 2
Semester 2, 2025
"""

# details
__author__ = "Christopher Dowling"
__email__ = "c.dowling1@uq.edu.au"
__date__ = "<insert date here>"


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
    nonneg = road_in[road_in >= 0]
    scaled = np.all(nonneg % 18 == 0) and np.any(nonneg > v_max) if nonneg.size > 0 else False
    
    if scaled:
        # convert to cell speed, then convert back to km/h
        speeds = np.where(road_in >= 0, road_in // 18, road_in)
        mask = (speeds >= 0) & (speeds < v_max)  # only accelerate below v_max
        speeds[mask] += 1
        road_out = np.where(speeds >= 0, speeds * 18, -1)
    else:
        mask = (road_in >= 0) & (road_in < v_max)  # only accelerate below v_max
        road_out[mask] += 1

    return road_out

"""
    Task 2 Deceleration Rule:
    gaps_ahead
    decelerate
    """

def gaps_ahead(road_in: np.ndarray[int], signals_in: np.ndarray[bool]) -> np.ndarray[int]:
    n = len(road_in)
    gaps = np.zeros(n, dtype=int)

    for i in range(n):
        speed = int(road_in[i])
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
    # prevent error
    if n_cells <= 0:
        return np.array([], dtype=int)

    # create an empty road
    road = np.full(n_cells, -1, dtype=int)

    # prevent error
    if vehicle_gap < 0:
        return road
    
    # add vehicles to the road evenly spaced out
    for i in range(0, n_cells, vehicle_gap + 1):
        if i < n_cells: 
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
        if self._speed < v_max:
            self._speed += 1

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
        
        time_in_cycle = self._time % self._cycle_length
        return time_in_cycle < self._red_duration
    
    def calculate_gap(self, position: int) -> int:
        n = self._length
        
        if not self._vehicles:
            return n - 1

        vehicle_positions = sorted([v.get_position() for v in self._vehicles])

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

            gap = min(gap, distance_to_signal)

        return gap
    
    """
    Task 8: Simulate
    """
    def simulate(self, num_steps: int, v_max: int, p: float) -> None:
        for step in range(num_steps):
            # accelerate all vehicles
            for v in self._vehicles:
                v.accelerate(v_max)

            # check gap ahead, decelerate if needed
            for v in self._vehicles:
                gap = self.calculate_gap(v.get_position())
                v.decelerate(gap)

            # random deceleration
            for v in self._vehicles:
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



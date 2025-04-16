# ------------------------------------------------------------
# JUGGLER RESERVOIR COMPUTER: EXPLANATION
# ------------------------------------------------------------
# This simulation demonstrates a physical reservoir computing system
# using a juggling scenario as the dynamical system.
#
# ----------------------------
# INPUTS:
# The system receives historical daily maximum temperatures from
# Washington D.C. (via the Open-Meteo API) as input signals.
# Each temperature modulates the dynamics of the juggler — specifically,
# it changes the vertical force with which the juggler throws the ball.
#
# ----------------------------
# RESERVOIR:
# The physical reservoir is the juggler-ball system itself.
# Its internal state evolves dynamically as the ball is thrown,
# caught, and repositioned in response to the input signal.
# At each time step, the juggler's reservoir state is encoded
# by the ball's position and the hand position.
#
# ----------------------------
# OUTPUTS:
# The readout layer is a simple Ridge regression model trained to
# predict future temperatures based on the juggler's state.
# After training, it outputs predictions for temperatures several
# days into the future, based on how the juggler has evolved over time.
#
# The final plot compares actual vs predicted temperatures and
# includes a 95% confidence band to visualize prediction uncertainty.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import Ridge
import matplotlib.widgets as widgets
import pandas as pd
import requests

# Constants
g = 9.81  # gravity (m/s^2)
dt = 0.01  # time step (s)
T = 168 * 3600 * dt  # simulate over 168 hours (7 days of hourly data)  # total simulation time (s)
# num_steps now depends on data size after loading

# Load real weather temperature data
def load_dc_temperature():
    try:
        url = "https://archive-api.open-meteo.com/v1/archive?latitude=38.8951&longitude=-77.0364&start_date=2025-01-10&end_date=2025-04-15&daily=temperature_2m_max&temperature_unit=fahrenheit&timezone=auto"
        response = requests.get(url)
        data = response.json()
        temps = data['daily']['temperature_2m_max']
        times = data['daily']['time']
        return np.array(temps), np.array(times)
    except Exception as e:
        print(f"Error fetching weather data: {e}. Using simulated fallback.")
        time = np.linspace(0, T, num_steps)
        temp_f = 70 + 15 * np.sin(2 * np.pi * time / T)
        return temp_f, [f"Simulated_{i}" for i in range(num_steps)]

external_signal_raw, external_times_raw = load_dc_temperature()
num_steps = min(len(external_signal_raw), int(168 * 3600 * dt / dt))  # max 7 days worth of hourly data
external_signal = external_signal_raw[:num_steps]
external_times = pd.to_datetime(external_times_raw[:num_steps])
external_times = external_times.strftime('%b %d')
num_steps = min(len(external_signal_raw), num_steps)
external_times = external_times_raw[:num_steps]
external_signal = external_signal_raw[:num_steps]

class Ball:
    def __init__(self, x, y, vx, vy):
        self.state = np.array([x, y, vx, vy])

    def update(self):
        self.state[0] += self.state[2] * dt
        self.state[1] += self.state[3] * dt
        self.state[3] -= g * dt

class Juggler:
    def __init__(self):
        self.hand_x = 0.0
        self.hand_speed = 2.0

    def move_hand(self, target_x):
        if target_x > self.hand_x:
            self.hand_x += self.hand_speed * dt
        elif target_x < self.hand_x:
            self.hand_x -= self.hand_speed * dt

    def try_catch_and_throw(self, ball, temperature):
        if ball.state[1] <= 0.5 and abs(ball.state[0] - self.hand_x) < 0.3:
            ball.state[1] = 0.5
            ball.state[3] = 3.0 + 0.02 * temperature
            ball.state[2] = np.random.uniform(-1, 1)
            return True
        return False

def normalize_temperature(temp_f):
    temp_min, temp_max = 55, 85
    return (temp_f - temp_min) / (temp_max - temp_min) * 6

def setup_simulation(pred_step):
    ball = Ball(0, 0.5, 1.0, 5.0)
    juggler = Juggler()
    reservoir_states, targets, ball_positions, hand_positions = [], [], [], []

    for t in range(num_steps):
        temperature = external_signal[t]
        ball.update()
        juggler.move_hand(ball.state[0])
        juggler.try_catch_and_throw(ball, temperature)
        state_snapshot = np.concatenate((ball.state[:2], [juggler.hand_x]))
        reservoir_states.append(state_snapshot)
        if t + pred_step < num_steps:
            targets.append(external_signal[t + pred_step])
        ball_positions.append(ball.state[:2].copy())
        hand_positions.append(juggler.hand_x)

        X = np.array(reservoir_states[:-pred_step])
    y = np.array(targets)

    # Filter out invalid rows
    # Ensure all values are numeric and valid
    X = np.array(reservoir_states[:-pred_step], dtype=np.float64)
    y = np.array(targets, dtype=np.float64)

    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)

    return ball_positions, hand_positions, model, predictions

# Start simulation with initial prediction step
pred_step_init = 10
ball_positions, hand_positions, model, predictions = setup_simulation(pred_step_init)

# Plot actual vs predicted temperature
actual = external_signal[pred_step_init:len(predictions) + pred_step_init]
predicted = predictions[:len(actual)]
times_plot = external_times[pred_step_init:len(predictions) + pred_step_init]

plt.figure(figsize=(15, 5))
plt.plot(times_plot, actual, label='Actual Temperature (°F)', color='blue')
plt.plot(times_plot, predicted, label='Predicted Temperature (°F)', color='orange', linestyle='--')

# Calculate and plot 95% confidence band
residuals = actual - predicted
std_error = np.std(residuals)
upper_band = predicted + 1.96 * std_error
lower_band = predicted - 1.96 * std_error
plt.fill_between(times_plot, lower_band, upper_band, color='orange', alpha=0.2, label='95% Confidence Band')
plt.xticks(rotation=45)
plt.title('Juggler Reservoir Computer: Actual vs Predicted Temperature (Washington D.C.)')
plt.xlabel('Time')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.tight_layout()
plt.show()

print("COMPLETE")

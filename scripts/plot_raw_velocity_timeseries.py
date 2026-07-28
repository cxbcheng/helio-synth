from astropy.visualization import time_support
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.data_ingest.ingest import run_ingest

# TODO: documentation and prompts
times, velocities = run_ingest()
elapsed = (times - times[0]).sec

time_support(format='iso', simplify=True)

plt.figure()
plt.plot(elapsed, velocities)
plt.xlabel('Time since first 2020-01-01 00:00:00 [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity vs Time at the Center of the Solar Disk')
plt.grid(True)
plt.show()
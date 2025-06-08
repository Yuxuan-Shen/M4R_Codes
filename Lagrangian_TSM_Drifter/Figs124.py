import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load the MATLAB data file
mat_data = scipy.io.loadmat('/Users/syx/Desktop/Imperial College London UG/Year4/MATH70050 M4R/Lagrangian Time Series Models for Ocean Surface Drifter Trajectories/Lagrangian_TSM_Drifter/drifterbetty2.mat')
drifterbetty = mat_data['drifterbetty']
lon = drifterbetty['lon'][0][0].flatten()
lat = drifterbetty['lat'][0][0].flatten()

# Create complex coordinates
XX = lon + 1j * lat
Z = XX[1900:4900]  # TIME SERIES (MATLAB indexing starts at 1)
X = XX[2970:3570]  # TIME SERIES BLUE PORTION
Y = XX[4200:4800]  # TIME SERIES RED PORTION

# Create the plot
plt.figure(figsize=(8, 8))
plt.plot(Z.real, Z.imag, color=[0.7, 0.7, 0.7], label='Full Trajectory')  # Gray Line
plt.plot(X.real, X.imag, 'b', label='Blue Portion')  # Blue Line
plt.plot(Y.real, Y.imag, 'g', label='Green Portion')  # Green Line

# Configure the axes
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks(ticks=[-71, -70, -69, -68, -67, -66])
plt.yticks(ticks=[32, 33, 34, 35, 36, 37])
plt.xlim([-72, -65])
plt.ylim([31, 38])
plt.axis('square')
plt.legend()

# Export the figure to EPS
plt.savefig('Fig1R.eps', format='eps', dpi=300)

import numpy as np

gps_x = np.loadtxt('GPS.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]
acc_x = np.loadtxt('Accelerometer.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]

gps_x_std = np.std(gps_x)
print(np.std(gps_x))
print(np.std(acc_x))
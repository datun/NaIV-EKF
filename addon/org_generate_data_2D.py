#!/usr/bin/python
'''
This function generates the data for a constant-velocity 2D motion example.
The motion is divided in 5 segments, passing through the points: 
  - (200,-100)  to (100,100)
  - (100,100)   to (100,300)
  - (100,300)   to (-200,300)
  - (-200,300)  to (-200,-200)
  - (-200,-200) to (0,0)


 Q are the elements of the process noise diagonal covariance matrix (only for position)
 R are the elements of the measurement noise diagonal covariance matrix
 z is the [distance; orientation] data measured from the origin
 true_data are the true values of the position and speed

example of use
python generate_data_2D.py 10 10 10 1e-3
'''

import sys
import numpy as np
import matplotlib.pyplot as plt

Q1 = float(sys.argv[1]) 
Q2 = float(sys.argv[2])
R1 = float(sys.argv[3]) 
R2 = float(sys.argv[4])

nSegments = 5
points = np.array([[200, -100],
		   [100, 100],
		   [100, 300], 
		   [-200, 300], 
		   [-200, -200],
		   [0,0]], dtype=float)  

dp = np.diff(points, axis=0)
dist = dp**2

dist = np.round(np.sqrt(dist[:,0] + dist[:,1])) # distance
ang = np.arctan2(dp[:, 1], dp[:, 0]) # orientation
ang = np.array([ang]).T

NumberOfDataPoints = np.sum(dist)

T = 0.5 # [s] Sampling time interval

v_set = 2 * np.hstack((np.cos(ang), np.sin(ang)))

idx = 0
v = np.kron(np.ones((dist[idx], 1)), v_set[idx, :])
for idx in range(1, nSegments):
      v = np.vstack((v, np.kron(np.ones((dist[idx], 1)), v_set[idx, :])))
      

# ==motion generation====================================================

A = np.array([[1, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 1, 0], 
              [0, 0, 0, 0]], dtype=float)  

B = np.array([[T, 0], 
              [1, 0], 
              [0, T], 
              [0, 1]], dtype=float)  

G = np.array([[T**2/2, 0], 
              [T, 0], 
              [0, T**2/2], 
              [0, T]], dtype=float)  

w_x = np.random.normal(0.0, np.sqrt(Q1), NumberOfDataPoints) # noise in x-direction
w_y = np.random.normal(0.0, np.sqrt(Q2), NumberOfDataPoints) # noise in y-direction

w = np.hstack((np.array([w_x]).T, np.array([w_x]).T))
x = np.zeros((NumberOfDataPoints, 4))
x[0, :] = [200, 0, -100, 0]
for idx in range(1, int(NumberOfDataPoints)):
  x[idx, :] = np.dot(A, np.array(x[idx-1, :])) + np.dot(B, v[idx,:]) + np.dot(G, w[idx, :] )

true_data = x # 2D data: [px; vx; py; vy]

# ==measurement generation===============================================
position = x[:,(0,2)] # 2D position data

# distance and orientation with respect to the origin
z = np.zeros((NumberOfDataPoints, 2))
for idx in range(0, int(NumberOfDataPoints)):
  z[idx, 0] = np.sqrt(np.dot(position[idx, :], position[idx, :]))
  z[idx, 1] = np.arctan2(position[idx, 1], position[idx, 0]) 

# unwrap radian phases by changing absolute jumps greater than pi to their 2*pi complement
z[:, 1] = np.unwrap(z[:, 1])

v_meas = np.vstack((np.random.normal(0.0, np.sqrt(R1), NumberOfDataPoints), np.random.normal(0.0, np.sqrt(R2), NumberOfDataPoints))).T
z_exact = z
z = z + v_meas # add measurement noise

# == plots ============================
f1 = plt.figure()
plt.plot(x[:, 0], x[:, 2], label='linear')
plt.xlabel('x-axis [m]')
plt.ylabel('y-axis [m]')
plt.savefig('xy.pdf') 
f1.show()

xlab = [[' '], ['Time step [s]']]
ylab = [['r [m]'], ['$\theta$ [rad]']]
f2 = plt.figure()
for idx in range(0, 2):
  plt.subplot(2, 1, idx+1)
  line_z, = plt.plot(z[:, idx], label='linear')
  line_ze, = plt.plot(z_exact[:, idx], label='linear')
  plt.xlabel(xlab[idx])
  plt.ylabel(ylab[idx])
  plt.legend([line_z, line_ze], ['Measured', 'Exact'], fancybox=True, framealpha=0.0, loc='lower center', ncol=2)
  #leg.get_frame().set_linewidth(0.0)
plt.savefig('r_th.pdf') 
f2.show()


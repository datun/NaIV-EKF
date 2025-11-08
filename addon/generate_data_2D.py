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

import numpy as np
import matplotlib.pyplot as plt

class gen_2D_data:
    def __init__(self, Q1_in, Q2_in, R1_in, R2_in):
        self.Q1 = Q1_in
        self.Q2 = Q2_in
        self.R1 = R1_in
        self.R2 = R2_in
        self.T = 0.5

        self.nSegments = 5
        self.points = np.array([[200, -100],
                               [100, 100],
                               [100, 300],
                               [-200, 300],
                               [-200, -200],
                               [0,0]], dtype=float)
        self.idx = 0
        self.A = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]], dtype=float)

        self.dp = None
        self.dist = None
        self.ang = None
        self.NumberOfDataPoints = None
        self.v_set = None
        self.v = None
        self.B = None
        self.G = None
        self.w_x = None
        self.w_y = None
        self.w = None
        self.x = None
        self.true_data = None
        self.position = None
        self.v_meas = None
        self.z_exact = None
        self.z = None

        self.init_variables()

    def init_variables(self):
        self.dp = np.diff(self.points, axis=0)
        self.dist = self.dp**2

        self.dist = np.round(np.sqrt(self.dist[:,0] + self.dist[:,1])) # distance
        self.ang = np.arctan2(self.dp[:, 1], self.dp[:, 0]) # orientation
        self.ang = np.array([self.ang]).T
        self.NumberOfDataPoints = int(np.sum(self.dist))

        self.v_set = 2 * np.hstack((np.cos(self.ang), np.sin(self.ang)))
        self.v = np.kron(np.ones((int(self.dist[self.idx]), 1)), self.v_set[self.idx, :])
        for idx in range(1, self.nSegments):
            self.v = np.vstack((self.v, np.kron(np.ones((int(self.dist[idx]), 1)), self.v_set[idx, :])))

        # ==motion generation====================================================
        self.A = np.array([[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]], dtype=float)

        self.B = np.array([[self.T, 0],
                          [1, 0],
                          [0, self.T],
                          [0, 1]], dtype=float)

        self.G = np.array([[self.T**2/2, 0],
                           [self.T, 0],
                           [0, self.T**2/2],
                           [0, self.T]], dtype=float)

        self.w_x = np.random.normal(0.0, np.sqrt(self.Q1), self.NumberOfDataPoints) # noise in x-direction
        self.w_y = np.random.normal(0.0, np.sqrt(self.Q2), self.NumberOfDataPoints) # noise in y-direction

        self.w = np.hstack((np.array([self.w_x]).T, np.array([self.w_x]).T))
        self.x = np.zeros((self.NumberOfDataPoints, 4))
        self.x[0, :] = [500, 0, 200, 0]
        for idx in range(1, int(self.NumberOfDataPoints)):
            self.x[idx, :] = np.dot(self.A, np.array(self.x[idx-1, :])) + np.dot(self.B, self.v[idx,:]) \
                           + np.dot(self.G, self.w[idx, :] )

        self.true_data = self.x # 2D data: [px; vx; py; vy]

        # ==measurement generation===============================================
        self.position = self.x[:,(0,2)] # 2D position data

        # distance and orientation with respect to the origin
        self.z = np.zeros((self.NumberOfDataPoints, 2))
        for idx in range(0, int(self.NumberOfDataPoints)):
            self.z[idx, 0] = np.sqrt(np.dot(self.position[idx, :], self.position[idx, :]))
            self.z[idx, 1] = np.arctan2(self.position[idx, 1], self.position[idx, 0])

        # unwrap radian phases by changing absolute jumps greater than pi to their 2*pi complement
        self.z[:, 1] = np.unwrap(self.z[:, 1])

        self.v_meas = np.vstack((np.random.normal(0.0, np.sqrt(self.R1), self.NumberOfDataPoints),
                                 np.random.normal(0.0, np.sqrt(self.R2), self.NumberOfDataPoints))).T
        self.z_exact = self.z
        self.z = self.z + self.v_meas # add measurement noise

        return self.z

    def plot_graphs(self):
        # == plots ============================
        f1 = plt.figure()
        plt.plot(self.x[:, 0], self.x[:, 2], label='linear')
        plt.xlabel('x-axis [m]')
        plt.ylabel('y-axis [m]')
        plt.savefig('xy.pdf')
        f1.show()

        xlab = [[' '], ['Time step [s]']]
        ylab = [['r [m]'], ['$\theta$ [rad]']]
        f2 = plt.figure()
        for idx in range(0, 2):
            plt.subplot(2, 1, idx+1)
            line_z, = plt.plot(self.z[:, idx], label='linear')
            line_ze, = plt.plot(self.z_exact[:, idx], label='linear')
            plt.xlabel(xlab[idx])
            plt.ylabel(ylab[idx])
            plt.legend([line_z, line_ze], ['Measured', 'Exact'], fancybox=True, framealpha=0.0, loc='lower center', ncol=2)
            # leg.get_frame().set_linewidth(0.0)
        plt.savefig('r_th.pdf')
        f2.show()


# TESTING CODE

# test = gen_2D_data(10,10,10,1e-3)
#
# data = test.init_variables()
# print(data)
# print()
# # print(data[0].max)
# xmax, ymax = data.max(axis=0)
# xmin, ymin = data.min(axis=0)
#
# print(xmax, ymax)
# print(xmin, ymin)


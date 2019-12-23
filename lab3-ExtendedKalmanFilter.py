import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff

import addon.generate_data_2D as gen2D

# Sympy also has jacobi stuff, but it may be for polynomials
# I may remove it if we don't ever use diff stuff

class ExtendedKF:
    def __init__(self, T_in):


        self.matR = None
        self.Ti = T_in

        self.P_pred_list = []
        self.P_corr_list = []
        self.K_gain_list = []
        self.X_pred_list = []
        self.X_corr_list = []

        # matQ by Eq(10)
        self.matQ = np.ndarray([[self.Ti**3/3, self.Ti**2/2, 0, 0],
                                [self.Ti**2/2, self.Ti, 0, 0],
                                [0, 0, self.Ti**3/3, self.Ti**2/2],
                                [0, 0, self.Ti**2/2, self.Ti**3/3]])
        # matA or part of f by Eq(10)
        self.f_KF = np.ndarray([[1, self.Ti, 0, 0],
                                [0, 1 ,0 ,0],
                                [0, 0, 1, self.Ti],
                                [0, 0, 0, 1]])

    def h_KF(self,x_k):
        # around Eq (9)
        # x(k) = [ x_pos x_vel y_pos y_vel ]
        # Assuming h(x(k),v(k)) is converted from cartesian to polar like this
        # Converting x position values to polar as r_x and theta_x
        # Converting x velocity values to polar as r_v and theta_v
        r_x = np.sqrt(x_k[0]**2 + x_k[2]**2)
        theta_x = np.arctan(x_k[0]/x_k[2]) * 180 / np.pi
        r_v = np.sqrt(x_k[1]**2 + x_k[3]**2)
        theta_v = np.arctan(x_k[1]/x_k[3])
        return [(r_x,theta_x), (r_v,theta_v)]


    def process_noise(self, size, Q):
        # Ex: process_noise(self, (1,3), Q)
        self.matQ = np.random.normal(0, Q, size)

    def meas_noise(self, size, R):
        # Ex: meas_noise(self, (1,3), Q)
        self.matR = np.random.normal(0, R, size)

    def jacobA(self,T):
    # Calculates the Jacobian A at every time step.
    # Needs the time step T as an input
        A = np.array([
                        [1, T, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, T],
                        [0, 0, 0, 1]
                    ])
        return A

    def jacobG(self,k):

    def jacobC(self,x_k):

    def jacobH(self,x_k):

    def t_up_x_pred(self,x_corr):
        # Eq (3)
        self.X_pred_list.append(self.f_KF @ x_corr + self.matQ)

    def t_up_p_pred(self, P_corr,Q_k,k):
        # Eq(4)
        matA = self.jacobA(k)
        matG = self.jacobG(k)
        self.P_pred_list.append(matA @ P_corr @ matA.T + matG @ Q_k @ matG.T)

    def m_up_k_gain(self, P_pred_k, step):
        # Eq(5)
        temp1 = np.linalg.inv(self.jacobC(step) @ P_pred_k @ self.jacobC(step).T +
                              self.jacobH(step) @ self.matR @ self.jacobH(step).T)
        self.K_gain_list.append(P_pred_k @ self.jacobC(step).T @ temp1)

    def m_up_x_corr(self,X_pred_k, K_k, Z_k):
        # Eq(6)
        self.X_corr_list.append(X_pred_k + K_k @ (Z_k - self.h_KF(X_pred_k)))

    def m_up_p_corr(self,K_k, P_pred_k, step):
        # Eq(7)
        # identity 5 to be changed to a matrix size stuff
        temp1 = np.linalg.inv(P_pred_k)
        self.P_corr_list.append((np.identity(5) - K_k @ self.jacobC(step)) @ temp1)


def nees(x_true,x_pred,p_list):
    x_inter = x_true - x_pred
    return x_inter @ np.linalg.inv(p_list) @ x_inter.T


def nis(x_pred,z_in,p_list,C_in,H_in,R_in):
    S_k = C_in @ p_list @ C_in.T + H_in @ R_in @ H_in.T
    z1_k = z_in - C_in @ x_pred.T
    return z1_k.T @ np.linalg.inv(S_k) @ z1_k



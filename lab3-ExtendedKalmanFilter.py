import numpy as np
import matplotlib.pyplot as plt

from addon.generate_data_2D import gen_2D_data as gen2D

from sympy import symbols, diff
# Sympy also has jacobi stuff, but it may be for polynomials
# I may remove it if we don't ever use diff stuff


class ExtendedKF:
    def __init__(self, T_in, sigma):
        # Initialise piecewise constant velocity model.

        self.matR = None
        self.Ti = T_in
        self.P_pred_list = []
        self.P_corr_list = []
        self.K_gain_list = []
        self.x_hat_minus = []
        self.x_hat = []
        self.z = np.zeros((1, 2))

        # # matQ by Eq(10)
        # self.matQ = np.array([[self.Ti**3/3, self.Ti**2/2, 0, 0],
        #                         [self.Ti**2/2, self.Ti, 0, 0],
        #                         [0, 0, self.Ti**3/3, self.Ti**2/2],
        #                         [0, 0, self.Ti**2/2, self.Ti**3/3]])

        # in piecewise constant model w is a scalar process noise and Q = sigma
        self.matQ = sigma

        # matA or part of f by Eq(10)
        self.matA = np.array([[1, self.Ti, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, self.Ti],
                                [0, 0, 0, 1]])
        # matrix G relates the process noise w to the state x
        self.matG = np.identity(4)

        # matrix G relates the process noise w to the state x
        self.matH = np.zeros((2,2))

    def get_x_hat_0(self, z_in):
        z_0 = z_in[0]
        z_1 = z_in[1]

        # Convert Polar z to cartesian
        z0_x = z_0[0] * np.cos(z_0[1])
        z0_y = z_0[0] * np.sin(z_0[1])
        z1_x = z_1[0] * np.cos(z_1[1])
        z1_y = z_1[0] * np.sin(z_1[1])

        x = z0_x
        y = z0_y
        v_x = z1_x - z0_x / self.Ti
        v_y = z1_y - z1_y / self.Ti
        return np.array([x, v_x, y, v_y]).reshape((1, 4))

    def get_P_0(self):
        P_0 = np.identity(4)
        return P_0

    def h_KF(self, x_k):

        """
        INPUT: X Vector in Cartesian Co-ordinates
            x(k) = [ x_position x_velocity y_position y_velocity ]
        OUTPUT: Position and Velocity in Polar Co-ordinates

        Ref: Eq (9)
            x(k) = [ x_pos x_vel y_pos y_vel ]

        Assuming h(x(k),v(k)) is converted from cartesian to polar as follows
        Converting position values to polar as r_x and theta_x
        Converting velocity values to polar as r_v and theta_v
        """

        # Sid check these two out:
        # [1] https://math.stackexchange.com/questions/2444965/relationship-between-cartesian-velocity-and-polar-velocity
        # [2] https://robotics.stackexchange.com/questions/1992/jacobian-of-the-observation-model
        # [1] is for changing cartesian coordinate into polar coordinates
        # [2] is for the Jacobian of C.
        # I updated the polar coordinate conversion according to the [1],
        # we also do need to check jacobian of C.

        # r_x = np.sqrt(x_k[0]**2 + x_k[2]**2)
        # theta_x = np.arctan(x_k[0]/x_k[2]) * 180 / np.pi
        # r_v = np.sqrt(x_k[1]**2 + x_k[3]**2)
        # if x_k[3] == 0:
        #     theta_v = 0
        # else:
        #     theta_v = np.arctan(x_k[1]/x_k[3])
        # return np.array([r_x,theta_x, r_v,theta_v]).reshape(1,4)

        r_xy = np.sqrt(x_k[0]**2 + x_k[2]**2)
        theta_xy = np.arctan(x_k[0]/x_k[2]) * 180 / np.pi

        # r_v = (x_k[0] * x_k[1] + x_k[2] * x_k[3]) / np.sqrt(x_k[0]**2 + x_k[2]**2)
        # theta_v = (x_k[0] * x_k[3] - x_k[1] * x_k[2]) / (x_k[0]**2 + x_k[2]**2)

        return np.array([r_xy, theta_xy]).reshape(1, 2)

    def gen_z(self, x_in):
        for i in range(len(x_in)):
            self.z = np.vstack((self.z, self.h_KF(x_in[i])))
        self.z = np.delete(self.z, 0, 0)

    def meas_noise(self, size, R):
        # Ex: meas_noise(self, (1,3), Q)
        self.matR = np.random.normal(0, R, size)

    def jacobA(self):
        print("Read commented section!")
        print("If we will have additional time after documenting, we will support comment explanation with code!")
        # This was intended for Jacobian matrix of part. der. of f w.r.t. state x.
        # ( Eq.1 groups w(k) with x_k and u(k), fortunately explicit version is found in Eq.10;
        #  clarifying what is meant by f(x_k,u(k),w(k)) since it can mean anything!)
        # Considering the equation x(k) = matA @ x(k-1) + █ * u(k) + █ * 0
        # derivative wrt state x yields matrix A.
        # Thus A[i,j] Jacobian is always A
        # /// After manual computation, here is the matlab link as a help source which describes what is stated above
        # /// https://mathworks.com/help/driving/ug/extended-kalman-filters.html
        return

    def jacobG(self):
        print("Read commented section!")
        print("If we will have have additional time after documenting, we will support comment explanation with code!")
        # This was intended for Jacobian matrix of part. der. of f w.r.t. process noise w(k).
        # ( Eq.1 groups w(k) with x_k and u(k), fortunately explicit version is found in Eq.10;
        #  clarifying what is meant by f(x_k,u(k),w(k)) since it can mean anything!)
        # Considering the equation x(k) = matA @ x(k-1) + █ * u(k) + █ * 0 (where 0 = w(k))
        # As given in the pdf, w(k) is equal to zero, which is a constant. Derivative of constant also yields 0.
        # Thus G[i,j] Jacobian is always 0.
        # /// After manual computation, here is the matlab link as a help source which describes what is stated above
        # /// https://mathworks.com/help/driving/ug/extended-kalman-filters.html
        return

    def hardJacobC(self, x_in):
        # The partial derivative of function h is calculated on paper and provided in this function to only compute
        # given input data (aka hardcoded the Jacobian matrix). Depending on the time we have left, we may also add
        # dynamic variant where it calculates wrt to the input vector.
        # x_in => x_minus_k
        jac_c = np.array([[-x_in[0] / (x_in[0]**2 + x_in[2]**2),       1/x_in[0]*(1+(x_in[2]/x_in[0])**2),       0, 0],
                          [x_in[0] / np.sqrt(x_in[0]**2 + x_in[2]**2), x_in[2]/np.sqrt(x_in[0]**2 + x_in[2]**2), 0, 0]
                          ])
        return jac_c.reshape(2, 4)

    def jacobH(self):
        print("Read commented section!")
        print("If we will have have additional time after documenting, we will support comment explanation with code!")
        # This was intended for Jacobian matrix of part. der. of h w.r.t. measurement noise v(k).
        # Considering the equation z(k) = h(x_k) + v(k)
        # ( Eq.2 groups v(k) with x_k unnecessarily; causing additional confusion and there is no other reference to
        # function h unlike function f that is shown in Eq.1 and LATER EXPLAINED in Eq.10)
        # Obvious derivation wrt v(k) will yield 1 and 1 is identity matrix in terms of matrices.
        # /// After manual computation, here is the matlab link as a help source which describes what is stated above
        # /// https://mathworks.com/help/driving/ug/extended-kalman-filters.html
        return

    def gen_x_hat_minus(self, x_corr_k):
        # Eq (3)
        self.x_hat_minus.append(self.matA @ x_corr_k)

    def gen_p_minus(self, P_corr):
        # Eq(4)
        self.P_pred_list.append(self.matA @ P_corr @ self.matA.T)  # + matG @ Q_k @ matG.T) commented due to derivative

    def gen_k_gain(self, P_pred_k, jacobC):
        # Eq(5)
        temp1 = np.linalg.inv(jacobC @ P_pred_k @ jacobC.T + self.matR)
        self.K_gain_list.append(P_pred_k @ jacobC.T @ temp1)

    def corr_x_hat(self, X_pred_k, K_k, Z_k):
        # Eq(6)
        self.x_hat.append(X_pred_k + K_k @ (Z_k - self.h_KF(X_pred_k)))

    def corr_p(self, K_k, P_pred_k, jacobC):
        # Eq(7)
        # identity 5 to be changed to a matrix size stuff
        temp1 = np.linalg.inv(P_pred_k)
        self.P_corr_list.append((np.identity(5) - K_k @ jacobC) @ temp1)

    def KalmanFiltering(self):

        # In piecewise constant model w is a scalar process noise variable w(k) ~ N(0, Q) with zero mean.
        # Q in this case is the process noise variance.
        w = np.random.normal(0, self.matQ, 4)
        for i, meas in enumerate(self.z):
            self.gen_x_hat_minus(self.x_hat[i])
            self.gen_p_minus(self.P_corr_list[i])
            matC_k = self.hardJacobC(self.x_hat_minus[i])
            self.gen_k_gain(self.P_pred_list[i], matC_k)
            self.corr_x_hat(self.x_hat_minus[i], self.K_gain_list[i], meas)
            self.corr_p(self.K_gain_list[i], self.P_pred_list[i], matC_k)


def nees(x_true, x_pred, p_list):
    x_inter = x_true - x_pred
    return x_inter @ np.linalg.inv(p_list) @ x_inter.T


def nis(x_pred, z_in, p_list, C_in, H_in, R_in):
    S_k = C_in @ p_list @ C_in.T + H_in @ R_in @ H_in.T
    z1_k = z_in - C_in @ x_pred.T
    return z1_k.T @ np.linalg.inv(S_k) @ z1_k


def main():
    gen_data = gen2D(10, 10, 10, 1e-3)  # init real values, measured values etc.
    extKF_T = ExtendedKF(5)  # T value for initialising
    extKF_T.gen_z(gen_data.x)
    print(7)



main()
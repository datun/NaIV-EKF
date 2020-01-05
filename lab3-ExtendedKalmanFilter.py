import numpy as np
import matplotlib.pyplot as plt
from addon.generate_data_2D import gen_2D_data as gen2D


class ExtendedKF:
    def __init__(self, T_in, z_in, Q_x, Q_y, R_x, R_y):
        # Initialise piecewise constant velocity model.

        self.Ti = T_in
        self.P_pred_list = [np.zeros((4,4))]  # 4x4
        self.P_corr_list = [np.identity(4)]  # 4x4
        self.K_gain_list = [np.zeros((4,2))]  # 4x2
        self.x_hat_minus = [np.zeros((4,1))]  # 4x1
        self.x_hat = []  # 4x1

        self.z = z_in  # 1x2
        self.num_observ = len(self.z)

        self.matA = np.array([[1, self.Ti, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, self.Ti],
                              [0, 0, 0, 1]]).reshape(4, 4)

        # matrix G relates the process noise w to the state x
        self.matG = np.array([[self.Ti**2/2, 0],
                              [self.Ti, 0],
                              [0, self.Ti**2/2],
                              [0, self.Ti]]).reshape(4, 2)

        self.matQ = np.array([[Q_x, 0],
                              [0, Q_y]]).reshape(2, 2)

        self.matR = np.array([[R_x, 0],
                              [0, R_y]]).reshape(2, 2)

        self.matC = [np.zeros((2,4))]

        # in piecewise constant model w is a scalar process noise and Q = sigma
        self.matGQGT = self.matG @ self.matQ @ self.matG.T

        # matrix G relates the process noise w to the state x
        self.matH = np.identity(2)

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
        self.x_hat.append(np.array([x, v_x, y, v_y]).reshape((4, 1)))

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

        r_xy = np.sqrt(x_k[0]**2 + x_k[2]**2)
        theta_xy = np.arctan(x_k[2]/x_k[0]) * 180 / np.pi
        return np.array([r_xy, theta_xy]).reshape(2, 1)

    def meas_noise(self, size, R):
        # Ex: meas_noise(self, (1,3), Q)
        self.matR = np.random.normal(0, R, size)

    def hardJacobC(self, x_in):
        # The partial derivative of function h is calculated on paper and provided in this function to only compute
        # given input data (aka hardcoded the Jacobian matrix). Depending on the time we have left, we may also add
        # dynamic variant where it calculates wrt to the input vector.
        # x_in => x_minus_k

        # Below is the pre-ugly fix of array problem:
        # jac_c = np.array([[x_in[0]/np.sqrt(x_in[0]**2 + x_in[2]**2), 0, x_in[2]/np.sqrt(x_in[0]**2 + x_in[2]**2), 0],
        #                           [-x_in[2]/(x_in[0]**2 + x_in[2]**2), 0, x_in[0]/(x_in[0]**2 + x_in[2]**2), 0]])
        jac_c = np.array([[float(x_in[0]/np.sqrt(x_in[0]**2 + x_in[2]**2)), 0., float(x_in[2]/np.sqrt(x_in[0]**2 + x_in[2]**2)), 0.],
                          [float(-x_in[2]/(x_in[0]**2 + x_in[2]**2)), 0., float(x_in[0]/(x_in[0]**2 + x_in[2]**2)), 0.]])
        return jac_c

    def gen_x_hat_minus(self, x_corr_k):
        # Eq (3)
        self.x_hat_minus.append(self.matA @ x_corr_k)

    def gen_p_minus(self, P_corr_k):
        # Eq(4)
        self.P_pred_list.append(self.matA @ P_corr_k @ self.matA.T + self.matGQGT)

    def gen_k_gain(self, P_pred_k, jacobC):
        # Eq(5)
        temp1 = np.linalg.pinv(jacobC @ P_pred_k @ jacobC.T + self.matR)
        self.K_gain_list.append(P_pred_k @ jacobC.T @ temp1)

    def corr_x_hat(self, X_pred_k, K_k, Z_k):
        # Eq(6)
        self.x_hat.append(X_pred_k + K_k @ (Z_k.reshape(2,1) - self.h_KF(X_pred_k)))

    def corr_p(self, K_k, P_pred_k, jacobC):
        # Eq(7)
        self.P_corr_list.append((np.identity(4) - K_k @ jacobC) @ P_pred_k)

    def KalmanFiltering(self):
        # In piecewise constant model w is a scalar process noise variable w(k) ~ N(0, Q) with zero mean.
        # Q in this case is the process noise variance.
        for i, meas in enumerate(self.z, start=1):
            self.gen_x_hat_minus(self.x_hat[i-1])
            # print(self.x_hat_minus)
            self.gen_p_minus(self.P_corr_list[i-1])
            # print(self.P_pred_list)
            self.matC.append(self.hardJacobC(self.x_hat_minus[i]))
            self.gen_k_gain(self.P_pred_list[i], self.matC[i])
            # print(self.K_gain_list)
            self.corr_x_hat(self.x_hat_minus[i], self.K_gain_list[i], meas)
            # print(self.x_hat)
            self.corr_p(self.K_gain_list[i], self.P_pred_list[i], self.matC[i])
            # print(self.P_corr_list)


def nees(x_true, x_pred, p_list):
    x_inter = x_true - x_pred
    return float(x_inter @ np.linalg.pinv(p_list) @ x_inter.T)


def nis(x_pred, z_in, p_list, C_in, H_in, R_in):
    S_k = C_in @ p_list @ C_in.T + H_in @ R_in @ H_in.T
    z1_k = z_in.reshape(2,1) - (C_in @ x_pred)
    return float(z1_k.T @ np.linalg.pinv(S_k) @ z1_k)


def main():
    gen_data = gen2D(1, 1, 1e-3, 1e-3)  # init real values, measured values etc.
    extKF_T = ExtendedKF(0.5, gen_data.z, gen_data.Q1, gen_data.Q2, gen_data.R1, gen_data.R2)  # T value for initialising
    extKF_T.get_x_hat_0(gen_data.z)
    extKF_T.KalmanFiltering()

    test = np.asarray(extKF_T.x_hat[:-1])

    nees_out = []
    nis_out = []
    for i in range(len(gen_data.true_data)):
        nees_out.append(nees(gen_data.true_data[i],test[i].reshape(-1,4),extKF_T.P_pred_list[i+1]))
    nees_out = np.asarray(nees_out)

    for i in range(len(gen_data.true_data)):
         nis_out.append(nis(test[i],extKF_T.z[i],extKF_T.P_pred_list[i+1],extKF_T.matC[i+1],extKF_T.matH,extKF_T.matR))
    nis_out = np.asarray(nis_out)


    plt.plot()
    plt.plot(test[:, 0], test[:, 2], label="Filter est.")
    plt.plot(gen_data.x[:, 0], gen_data.x[:, 2], label="True pos.")
    plt.title("Position Map")
    plt.legend(loc='upper right')
    plt.xlabel('x-axis [m]')
    plt.ylabel('y-axis [m]')
    plt.show()

    plt.subplot(211)
    plt.plot(nees_out, linestyle='-', marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('NEES')
    plt.title("NEES Values")

    plt.subplot(212)
    plt.plot(nis_out, linestyle='-', marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('NIS')
    plt.title("NIS Values")
    plt.show()


main()
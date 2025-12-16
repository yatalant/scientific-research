import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTabWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class P_Controller:
    def __init__(self, k, min_val=None, max_val=None):
        self.k = k
        self.min_val = min_val
        self.max_val = max_val

    def update(self, error):
        output = error * self.k
        if self.min_val is not None and self.max_val is not None:
            output = np.clip(output, self.min_val, self.max_val)
        return output


class UAVModel:
    def __init__(self):
        self.g = 9.81
        self.T_nxa = 0.2
        self.T_nya = 0.2
        self.xi_nya = 0.7
        self.T_nza = 0.2
        self.T_gamma = 0.2
        self.V_max_limit = 100.0

    def get_derivatives(self, state, u_control):
        x, y_h, z, V, theta, psi = state[0:6]
        nxa, nya, d_nya, nza = state[6:10]
        gamma, d_gamma = state[10:12]

        u_nxa, u_nya, u_nza, u_gamma = u_control

        if V < 0.1: V = 0.1

        dx = V * np.cos(psi) * np.cos(theta)
        dy = V * np.sin(theta)
        dz = -V * np.sin(psi) * np.cos(theta)

        dV = self.g * (nxa - np.sin(theta))

        cos_theta = np.cos(theta)
        if abs(cos_theta) < 1e-3: cos_theta = 1e-3 * np.sign(cos_theta)

        dTheta = (self.g / V) * (nya * np.cos(gamma) - nza * np.sin(gamma) - cos_theta)
        dPsi = -(self.g / (V * cos_theta)) * (nya * np.sin(gamma) + nza * np.cos(gamma))

        dnxa = (u_nxa - nxa) / self.T_nxa
        dd_nya = (u_nya - 2 * self.xi_nya * self.T_nya * d_nya - nya) / (self.T_nya ** 2)
        dnza = (u_nza - nza) / self.T_nza
        dd_gamma = (u_gamma - d_gamma) / self.T_gamma

        return np.array([dx, dy, dz, dV, dTheta, dPsi, dnxa, d_nya, dd_nya, dnza, d_gamma, dd_gamma])

    def rk4_step(self, state, u_control, dt):
        k1 = self.get_derivatives(state, u_control)
        k2 = self.get_derivatives(state + 0.5 * dt * k1, u_control)
        k3 = self.get_derivatives(state + 0.5 * dt * k2, u_control)
        k4 = self.get_derivatives(state + dt * k3, u_control)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state


class Autopilot:
    def __init__(self):
        self.reg_V = P_Controller(k=2.0, min_val=-10.0, max_val=10.0)
        self.reg_H_outer = P_Controller(k=0.5, min_val=-2.0, max_val=2.0)
        self.reg_H_inner = P_Controller(k=5.0, min_val=None, max_val=None)
        self.limit_gamma = np.deg2rad(20.0)
        self.reg_Psi_outer = P_Controller(k=-0.8, min_val=-self.limit_gamma, max_val=self.limit_gamma)
        self.reg_Gamma_inner = P_Controller(k=3.0)
        self.reg_nz = P_Controller(k=2.0)

    def calculate_controls(self, state, targets, uav_params):
        y_h = state[1]
        V = state[3]
        theta = state[4]
        psi = state[5]
        nya = state[7]
        nza = state[9]
        gamma = state[10]

        V_err = targets['V'] - V
        u_nxa = self.reg_V.update(V_err)

        H_err = targets['H'] - y_h
        H_dot_zad = self.reg_H_outer.update(H_err)

        H_dot_curr = V * np.sin(theta)
        dH_dot = H_dot_zad - H_dot_curr

        u_nya = self.reg_H_inner.update(dH_dot) + 1.0

        Psi_err = targets['Psi'] - psi
        while Psi_err > np.pi: Psi_err -= 2 * np.pi
        while Psi_err < -np.pi: Psi_err += 2 * np.pi

        gamma_zad = self.reg_Psi_outer.update(Psi_err)

        gamma_err = gamma_zad - gamma
        u_gamma = self.reg_Gamma_inner.update(gamma_err)

        nz_err = 0.0 - nza
        u_nza = self.reg_nz.update(nz_err)

        return np.array([u_nxa, u_nya, u_nza, u_gamma])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование БПЛА")
        self.resize(1100, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        settings_panel = QWidget()
        settings_panel.setFixedWidth(250)
        settings_layout = QVBoxLayout(settings_panel)
        main_layout.addWidget(settings_panel)

        grp_targets = QGroupBox("Задание (Target)")
        layout_t = QVBoxLayout()
        self.spin_V_zad = self.create_spinbox("Скорость V (м/с):", 25.0, 10.0, 100.0, layout_t)
        self.spin_H_zad = self.create_spinbox("Высота H (м):", 100.0, 0.0, 5000.0, layout_t)
        self.spin_Psi_zad = self.create_spinbox("Курс Psi (град):", 45.0, -180.0, 360.0, layout_t)
        grp_targets.setLayout(layout_t)
        settings_layout.addWidget(grp_targets)

        grp_init = QGroupBox("Начальные условия (Init)")
        layout_i = QVBoxLayout()
        self.spin_V0 = self.create_spinbox("V0 (м/с):", 20.0, 1.0, 100.0, layout_i)
        self.spin_H0 = self.create_spinbox("H0 (м):", 50.0, 0.0, 5000.0, layout_i)
        self.spin_Psi0 = self.create_spinbox("Psi0 (град):", 0.0, -180.0, 360.0, layout_i)
        grp_init.setLayout(layout_i)
        settings_layout.addWidget(grp_init)

        grp_sim = QGroupBox("Симуляция")
        layout_s = QVBoxLayout()
        self.spin_Time = self.create_spinbox("Время T (c):", 60.0, 5.0, 500.0, layout_s)
        grp_sim.setLayout(layout_s)
        settings_layout.addWidget(grp_sim)

        self.btn_start = QPushButton("Запустить моделирование")
        self.btn_start.setStyleSheet("background-color: #FF69B4; color: white; font-weight: bold; padding: 12px;")
        self.btn_start.clicked.connect(self.run_simulation)
        settings_layout.addWidget(self.btn_start)
        settings_layout.addStretch()

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.fig1 = plt.figure(figsize=(10, 8))
        self.canvas1 = FigureCanvas(self.fig1)
        self.tabs.addTab(self.canvas1, "Динамика полета")

        self.fig2 = plt.figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        self.tabs.addTab(self.canvas2, "Перегрузки и углы")

        self.fig3 = plt.figure(figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.fig3)
        self.tabs.addTab(self.canvas3, "Сигналы управления")

    def create_spinbox(self, text, val, min_v, max_v, parent_layout, decimals=1):
        label = QLabel(text)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(val)
        spin.setDecimals(decimals)
        parent_layout.addWidget(label)
        parent_layout.addWidget(spin)
        return spin

    def run_simulation(self):
        V_zad = self.spin_V_zad.value()
        H_zad = self.spin_H_zad.value()
        Psi_zad = np.deg2rad(self.spin_Psi_zad.value())

        V0 = self.spin_V0.value()
        H0 = self.spin_H0.value()
        Psi0 = np.deg2rad(self.spin_Psi0.value())

        T_max = self.spin_Time.value()
        dt = 0.02
        steps = int(T_max / dt)

        model = UAVModel()
        autopilot = Autopilot()

        state = np.zeros(12)
        state[0] = 0.0
        state[1] = H0
        state[3] = V0
        state[5] = Psi0
        state[7] = 1.0

        time_hist = np.linspace(0, T_max, steps)

        H_hist = np.zeros(steps)
        V_hist = np.zeros(steps)
        Psi_hist = np.zeros(steps)

        Nxa_hist = np.zeros(steps)
        Nya_hist = np.zeros(steps)
        Nza_hist = np.zeros(steps)
        Gamma_hist = np.zeros(steps)

        U_nxa_hist = np.zeros(steps)
        U_nya_hist = np.zeros(steps)
        U_nza_hist = np.zeros(steps)
        U_gamma_hist = np.zeros(steps)

        target_dict = {'V': V_zad, 'H': H_zad, 'Psi': Psi_zad}

        for i in range(steps):
            H_hist[i] = state[1]
            V_hist[i] = state[3]
            Psi_hist[i] = np.rad2deg(state[5])

            Nxa_hist[i] = state[6]
            Nya_hist[i] = state[7]
            Nza_hist[i] = state[9]
            Gamma_hist[i] = np.rad2deg(state[10])

            controls = autopilot.calculate_controls(state, target_dict, model)

            U_nxa_hist[i] = controls[0]
            U_nya_hist[i] = controls[1]
            U_nza_hist[i] = controls[2]
            U_gamma_hist[i] = np.rad2deg(controls[3])

            state = model.rk4_step(state, controls, dt)
            state[7] = np.clip(state[7], -8.0, 8.0)

        self.fig1.clear()
        ax1 = self.fig1.add_subplot(2, 2, 1)
        ax1.plot(time_hist, H_hist, 'b', linewidth=2, label=r'$H_{tek}$')
        ax1.plot(time_hist, [H_zad] * steps, 'r--', label=r'$H_{zad}$')
        ax1.set_title("Высота")
        ax1.set_xlabel("Время, с")
        ax1.set_ylabel("H, м")
        ax1.grid(True)
        ax1.legend()

        ax2 = self.fig1.add_subplot(2, 2, 2)
        ax2.plot(time_hist, V_hist, 'g', linewidth=2, label=r'$V_{tek}$')
        ax2.plot(time_hist, [V_zad] * steps, 'r--', label=r'$V_{zad}$')
        ax2.set_title("Скорость")
        ax2.set_xlabel("Время, с")
        ax2.set_ylabel("V, м/с")
        ax2.grid(True)
        ax2.legend()

        ax3 = self.fig1.add_subplot(2, 2, 3)
        ax3.plot(time_hist, Psi_hist, 'purple', linewidth=2, label=r'$\Psi_{tek}$')
        ax3.plot(time_hist, [self.spin_Psi_zad.value()] * steps, 'r--', label=r'$\Psi_{zad}$')
        ax3.set_title("Курс")
        ax3.set_xlabel("Время, с")
        ax3.set_ylabel(r'$\Psi$, град')
        ax3.grid(True)
        ax3.legend()
        self.fig1.tight_layout()
        self.canvas1.draw()

        self.fig2.clear()
        bx1 = self.fig2.add_subplot(2, 2, 1)
        bx1.plot(time_hist, Nxa_hist, 'k')
        bx1.set_title(r"Продольная перегрузка ($n_{xa}$)")
        bx1.set_xlabel("Время, с")
        bx1.grid(True)
        bx2 = self.fig2.add_subplot(2, 2, 2)
        bx2.plot(time_hist, Nya_hist, 'k')
        bx2.plot(time_hist, [8] * steps, 'r:', alpha=0.5)
        bx2.plot(time_hist, [-8] * steps, 'r:', alpha=0.5)
        bx2.set_title(r"Нормальная перегрузка ($n_{ya}$)")
        bx2.set_xlabel("Время, с")
        bx2.grid(True)
        bx3 = self.fig2.add_subplot(2, 2, 3)
        bx3.plot(time_hist, Nza_hist, 'k')
        bx3.set_title(r"Боковая перегрузка ($n_{za}$)")
        bx3.set_xlabel("Время, с")
        bx3.grid(True)
        bx4 = self.fig2.add_subplot(2, 2, 4)
        bx4.plot(time_hist, Gamma_hist, 'orange')
        bx4.plot(time_hist, [20] * steps, 'r:', alpha=0.5)
        bx4.plot(time_hist, [-20] * steps, 'r:', alpha=0.5)
        bx4.set_title(r"Крен ($\gamma$)")
        bx4.set_xlabel("Время, с")
        bx4.set_ylabel("град")
        bx4.grid(True)
        self.fig2.tight_layout()
        self.canvas2.draw()

        self.fig3.clear()
        cx1 = self.fig3.add_subplot(2, 2, 1)
        cx1.plot(time_hist, U_nxa_hist, 'm')
        cx1.plot(time_hist, [10] * steps, 'r:', alpha=0.3)
        cx1.plot(time_hist, [-10] * steps, 'r:', alpha=0.3)
        cx1.set_title(r"Упр. скоростью ($u_{n_{xa}}$)")
        cx1.set_xlabel("Время, с")
        cx1.grid(True)
        cx2 = self.fig3.add_subplot(2, 2, 2)
        cx2.plot(time_hist, U_nya_hist, 'm')
        cx2.set_title(r"Упр. высотой ($u_{n_{ya}}$)")
        cx2.set_xlabel("Время, с")
        cx2.grid(True)
        cx3 = self.fig3.add_subplot(2, 2, 3)
        cx3.plot(time_hist, U_nza_hist, 'm')
        cx3.set_title(r"Упр. боковое ($u_{n_{za}}$)")
        cx3.set_xlabel("Время, с")
        cx3.grid(True)
        cx4 = self.fig3.add_subplot(2, 2, 4)
        cx4.plot(time_hist, U_gamma_hist, 'm')
        cx4.set_title(r"Упр. креном ($u_{\gamma}$)")
        cx4.set_xlabel("Время, с")
        cx4.grid(True)
        self.fig3.tight_layout()
        self.canvas3.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

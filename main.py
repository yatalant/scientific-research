import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTabWidget, QDialog, QFormLayout)
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
        self.T_nxa = 0.1
        self.T_nya = 0.1
        self.xi_nya = 0.7
        self.T_nza = 0.1
        self.T_gamma = 0.2
        self.V_max_limit = 100.0

    def get_derivatives(self, state, u_control):
        # 0:x, 1:y(H), 2:z, 3:V, 4:theta, 5:psi
        x, y_h, z, V, theta, psi = state[0:6]
        # 6:nxa, 7:nya, 8:d_nya, 9:nza
        nxa, nya, d_nya, nza = state[6:10]
        # 10:gamma, 11:d_gamma
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
        self.reg_V = P_Controller(k=0.1, min_val=-10.0, max_val=10.0)
        self.reg_H_outer = P_Controller(k=0.1, min_val=-2.0, max_val=2.0)
        self.reg_H_inner = P_Controller(k=0.3, min_val=-10.0, max_val=10.0)
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


class ControllerSettingsDialog(QDialog):
    def __init__(self, autopilot):
        super().__init__()
        self.setWindowTitle("Настройки коэффициентов (K)")
        self.autopilot = autopilot
        self.resize(300, 300)
        layout = QFormLayout()

        self.spin_V = self.create_spin(self.autopilot.reg_V)
        layout.addRow("Скорость (V) K:", self.spin_V)

        self.spin_H_out = self.create_spin(self.autopilot.reg_H_outer)
        layout.addRow("Высота (H) K:", self.spin_H_out)

        self.spin_H_in = self.create_spin(self.autopilot.reg_H_inner)
        layout.addRow("Вертик. скор. (H_dot) K:", self.spin_H_in)

        self.spin_Psi = self.create_spin(self.autopilot.reg_Psi_outer)
        layout.addRow("Курс (Psi) K:", self.spin_Psi)

        self.spin_Gamma = self.create_spin(self.autopilot.reg_Gamma_inner)
        layout.addRow("Крен (Gamma) K:", self.spin_Gamma)

        self.spin_nz = self.create_spin(self.autopilot.reg_nz)
        layout.addRow("Бок. перегрузка (nz) K:", self.spin_nz)

        self.setLayout(layout)

    def create_spin(self, controller):
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setSingleStep(0.1)
        spin.setValue(controller.k)
        spin.valueChanged.connect(lambda val, c=controller: self.update_k(c, val))
        return spin

    def update_k(self, controller, value):
        controller.k = value


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование БПЛА (Полный анализ)")
        self.resize(1200, 800)

        self.autopilot = Autopilot()

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

        self.btn_settings = QPushButton("Настройки ПИ-регулятора")
        self.btn_settings.clicked.connect(self.open_settings)
        settings_layout.addWidget(self.btn_settings)

        self.btn_start = QPushButton("Запустить моделирование")
        self.btn_start.setStyleSheet("background-color: #FF69B4; color: white; font-weight: bold; padding: 12px;")
        self.btn_start.clicked.connect(self.run_simulation)
        settings_layout.addWidget(self.btn_start)
        settings_layout.addStretch()

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.fig1 = plt.figure(figsize=(10, 8))
        self.canvas1 = FigureCanvas(self.fig1)
        self.tabs.addTab(self.canvas1, "Кинематика (Координаты и Углы)")

        self.fig2 = plt.figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        self.tabs.addTab(self.canvas2, "Динамика (Вектор состояния)")

        self.fig3 = plt.figure(figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.fig3)
        self.tabs.addTab(self.canvas3, "Управление")

    def create_spinbox(self, text, val, min_v, max_v, parent_layout, decimals=1):
        label = QLabel(text)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(val)
        spin.setDecimals(decimals)
        parent_layout.addWidget(label)
        parent_layout.addWidget(spin)
        return spin

    def open_settings(self):
        dialog = ControllerSettingsDialog(self.autopilot)
        dialog.exec()

    def run_simulation(self):
        V_zad = self.spin_V_zad.value()
        H_zad = self.spin_H_zad.value()
        Psi_zad_deg = self.spin_Psi_zad.value()
        Psi_zad = np.deg2rad(Psi_zad_deg)

        V0 = self.spin_V0.value()
        H0 = self.spin_H0.value()
        Psi0 = np.deg2rad(self.spin_Psi0.value())

        T_max = self.spin_Time.value()
        dt = 0.02
        steps = int(T_max / dt)

        model = UAVModel()

        # Вектор состояния (12 переменных)
        state = np.zeros(12)
        state[0] = 0.0  # X
        state[1] = H0  # H (Y)
        state[2] = 0.0  # Z
        state[3] = V0  # V
        state[5] = Psi0  # Psi
        state[7] = 1.0  # nya

        time_hist = np.linspace(0, T_max, steps)
        state_hist = np.zeros((steps, 12))
        control_hist = np.zeros((steps, 4))
        target_dict = {'V': V_zad, 'H': H_zad, 'Psi': Psi_zad}

        for i in range(steps):
            state_hist[i] = state
            controls = self.autopilot.calculate_controls(state, target_dict, model)
            control_hist[i] = controls
            state = model.rk4_step(state, controls, dt)
            state[7] = np.clip(state[7], -8.0, 8.0)

    
        self.fig1.clear()

        # 1. X
        ax1 = self.fig1.add_subplot(2, 3, 1)
        ax1.plot(time_hist, state_hist[:, 0], 'b')
        ax1.set_title("Координата X")
        ax1.set_xlabel("Время, с")
        ax1.set_ylabel("X, м")
        ax1.grid(True)

        # 2. Z
        ax2 = self.fig1.add_subplot(2, 3, 2)
        ax2.plot(time_hist, state_hist[:, 2], 'b')
        ax2.set_title("Координата Z")
        ax2.set_xlabel("Время, с")
        ax2.set_ylabel("Z, м")
        ax2.grid(True)

        # 3. Высота (H)
        ax3 = self.fig1.add_subplot(2, 3, 3)
        ax3.plot(time_hist, state_hist[:, 1], 'b', linewidth=2, label=r'$H_{тек}$')
        ax3.plot(time_hist, [H_zad] * steps, 'r--', label=r'$H_{зад}$')
        ax3.set_title("Высота")
        ax3.set_xlabel("Время, с")
        ax3.set_ylabel("H, м")
        ax3.legend()
        ax3.grid(True)

        # 4. Скорость (V)
        ax4 = self.fig1.add_subplot(2, 3, 4)
        ax4.plot(time_hist, state_hist[:, 3], 'g', linewidth=2, label=r'$V_{тек}$')
        ax4.plot(time_hist, [V_zad] * steps, 'r--', label=r'$V_{зад}$')
        ax4.set_title("Скорость")
        ax4.set_xlabel("Время, с")
        ax4.set_ylabel("V, м/с")
        ax4.legend()
        ax4.grid(True)

        # 5. Курс (Psi)
        ax5 = self.fig1.add_subplot(2, 3, 5)
        ax5.plot(time_hist, np.rad2deg(state_hist[:, 5]), 'purple', linewidth=2, label=r'$\Psi_{тек}$')
        ax5.plot(time_hist, [Psi_zad_deg] * steps, 'r--', label=r'$\Psi_{зад}$')
        ax5.set_title("Курс")
        ax5.set_xlabel("Время, с")
        ax5.set_ylabel(r'$\Psi$, град')
        ax5.legend()
        ax5.grid(True)

        # 6. Тангаж (Theta)
        ax6 = self.fig1.add_subplot(2, 3, 6)
        ax6.plot(time_hist, np.rad2deg(state_hist[:, 4]), 'k')
        ax6.set_title(r"Тангаж ($\theta$)")
        ax6.set_xlabel("Время, с")
        ax6.set_ylabel("град")
        ax6.grid(True)

        self.fig1.tight_layout()
        self.canvas1.draw()

        # --- Вкладка 2: Динамика и Внутренние переменные ---
        self.fig2.clear()

        # 1. Nxa
        bx1 = self.fig2.add_subplot(2, 3, 1)
        bx1.plot(time_hist, state_hist[:, 6], 'k')
        bx1.set_title(r"Продольная ($n_{xa}$)")
        bx1.set_xlabel("Время, с")
        bx1.grid(True)

        # 2. Nya
        bx2 = self.fig2.add_subplot(2, 3, 2)
        bx2.plot(time_hist, state_hist[:, 7], 'k')
        bx2.set_title(r"Нормальная ($n_{ya}$)")
        bx2.set_xlabel("Время, с")
        bx2.grid(True)

        # 3. Производная dNya
        bx3 = self.fig2.add_subplot(2, 3, 3)
        bx3.plot(time_hist, state_hist[:, 8], 'gray')
        bx3.set_title(r"Производная ($dn_{ya}/dt$)")
        bx3.set_xlabel("Время, с")
        bx3.grid(True)

        # 4. Nza
        bx4 = self.fig2.add_subplot(2, 3, 4)
        bx4.plot(time_hist, state_hist[:, 9], 'k')
        bx4.set_title(r"Боковая ($n_{za}$)")
        bx4.set_xlabel("Время, с")
        bx4.grid(True)

        # 5. Крен (Gamma)
        bx5 = self.fig2.add_subplot(2, 3, 5)
        bx5.plot(time_hist, np.rad2deg(state_hist[:, 10]), 'orange')
        bx5.plot(time_hist, [20] * steps, 'r:', alpha=0.5)
        bx5.plot(time_hist, [-20] * steps, 'r:', alpha=0.5)
        bx5.set_title(r"Крен ($\gamma$)")
        bx5.set_xlabel("Время, с")
        bx5.set_ylabel("град")
        bx5.grid(True)

        # 6. Угловая скорость крена (dGamma)
        bx6 = self.fig2.add_subplot(2, 3, 6)
        bx6.plot(time_hist, state_hist[:, 11], 'orange')
        bx6.set_title(r"Скор. крена ($d\gamma/dt$)")
        bx6.set_xlabel("Время, с")
        bx6.grid(True)

        self.fig2.tight_layout()
        self.canvas2.draw()

        # --- Вкладка 3: Управление ---
        self.fig3.clear()

        cx1 = self.fig3.add_subplot(2, 2, 1)
        cx1.plot(time_hist, control_hist[:, 0], 'm')
        cx1.set_title(r"Упр. скоростью ($u_{n_{xa}}$)")
        cx1.set_xlabel("Время, с")
        cx1.grid(True)

        cx2 = self.fig3.add_subplot(2, 2, 2)
        cx2.plot(time_hist, control_hist[:, 1], 'm')
        cx2.set_title(r"Упр. высотой ($u_{n_{ya}}$)")
        cx2.set_xlabel("Время, с")
        cx2.grid(True)

        cx3 = self.fig3.add_subplot(2, 2, 3)
        cx3.plot(time_hist, control_hist[:, 2], 'm')
        cx3.set_title(r"Упр. боковое ($u_{n_{za}}$)")
        cx3.set_xlabel("Время, с")
        cx3.grid(True)

        cx4 = self.fig3.add_subplot(2, 2, 4)
        cx4.plot(time_hist, np.rad2deg(control_hist[:, 3]), 'm')
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

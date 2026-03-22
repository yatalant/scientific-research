import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTabWidget, QDialog, QFormLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure  # Импортируем Figure явно


class PI_Controller:
    def __init__(self, kp, ki, min_val=None, max_val=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.min_val = min_val
        self.max_val = max_val
        self.integral_limit = abs(integral_limit) if integral_limit is not None else None

        self.integral_error = 0.0

    def update(self, error, dt):
        p_output = error * self.kp

        self.integral_error += error * dt
        if self.integral_limit is not None:
            self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)

        i_output = self.integral_error * self.ki

        output = p_output + i_output

        if self.min_val is not None and self.max_val is not None:
            output = np.clip(output, self.min_val, self.max_val)

        return output

    def reset(self):
        self.integral_error = 0.0


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
        dPsi = -(self.g / (V * cos_theta)) * (
                    nya * np.sin(gamma) + nza * np.cos(gamma))  

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
        # kp, ki, min_val, max_val, integral_limit
        self.reg_V = PI_Controller(kp=0.10, ki=0.01, min_val=-10.0, max_val=10.0, integral_limit=5.0)
        self.reg_H_outer = PI_Controller(kp=0.04, ki=0.005, min_val=-2.0, max_val=2.0, integral_limit=1.0)
        self.reg_H_inner = PI_Controller(kp=0.80, ki=0.1, min_val=-10.0, max_val=10.0, integral_limit=5.0)
        self.limit_gamma = np.deg2rad(20.0)
        self.reg_Psi_outer = PI_Controller(kp=-0.80, ki=-0.05, min_val=-self.limit_gamma, max_val=self.limit_gamma,
                                           integral_limit=np.deg2rad(10.0))
        self.reg_Gamma_inner = PI_Controller(kp=3.00, ki=0.5, integral_limit=10.0)
        self.reg_nz = PI_Controller(kp=2.00, ki=0.2, integral_limit=5.0)

    def reset_controllers(self):
        self.reg_V.reset()
        self.reg_H_outer.reset()
        self.reg_H_inner.reset()
        self.reg_Psi_outer.reset()
        self.reg_Gamma_inner.reset()
        self.reg_nz.reset()

    def calculate_controls(self, state, targets, uav_params, dt):
        y_h = state[1]
        V = state[3]
        theta = state[4]
        psi = state[5]
        nya = state[7]
        nza = state[9]
        gamma = state[10]

        V_err = targets['V'] - V
        u_nxa = self.reg_V.update(V_err, dt)

        H_err = targets['H'] - y_h
        H_dot_zad = self.reg_H_outer.update(H_err, dt)

        H_dot_curr = V * np.sin(theta)
        dH_dot = H_dot_zad - H_dot_curr

        u_nya = self.reg_H_inner.update(dH_dot, dt) + 1.0

        Psi_err = targets['Psi'] - psi
        while Psi_err > np.pi: Psi_err -= 2 * np.pi
        while Psi_err < -np.pi: Psi_err += 2 * np.pi

        gamma_zad = self.reg_Psi_outer.update(Psi_err, dt)

        gamma_err = gamma_zad - gamma
        u_gamma = self.reg_Gamma_inner.update(gamma_err, dt)

        nz_err = 0.0 - nza
        u_nza = self.reg_nz.update(nz_err, dt)

        return np.array([u_nxa, u_nya, u_nza, u_gamma])


class ControllerSettingsDialog(QDialog):
    def __init__(self, autopilot):
        super().__init__()
        self.setWindowTitle("Настройки коэффициентов регулятора (ПИ)")
        self.autopilot = autopilot
        self.resize(400, 350)

        self.layout = QFormLayout()  # Создаем layout
        self.setLayout(self.layout)  # Присваиваем его диалогу

        # Теперь вызываем create_spin, который сам добавляет строки в self.layout
        self.spin_V_kp = self.create_spin(self.autopilot.reg_V, 'kp', "Скорость (V) Kp:")
        self.spin_V_ki = self.create_spin(self.autopilot.reg_V, 'ki', "Скорость (V) Ki:")

        self.spin_H_out_kp = self.create_spin(self.autopilot.reg_H_outer, 'kp', "Высота (H) Kp:")
        self.spin_H_out_ki = self.create_spin(self.autopilot.reg_H_outer, 'ki', "Высота (H) Ki:")

        self.spin_H_in_kp = self.create_spin(self.autopilot.reg_H_inner, 'kp', "Вертик. скор. (H_dot) Kp:")
        self.spin_H_in_ki = self.create_spin(self.autopilot.reg_H_inner, 'ki', "Вертик. скор. (H_dot) Ki:")

        self.spin_Psi_kp = self.create_spin(self.autopilot.reg_Psi_outer, 'kp', "Курс (Psi) Kp:")
        self.spin_Psi_ki = self.create_spin(self.autopilot.reg_Psi_outer, 'ki', "Курс (Psi) Ki:")

        self.spin_Gamma_kp = self.create_spin(self.autopilot.reg_Gamma_inner, 'kp', "Крен (Gamma) Kp:")
        self.spin_Gamma_ki = self.create_spin(self.autopilot.reg_Gamma_inner, 'ki', "Крен (Gamma) Ki:")

        self.spin_nz_kp = self.create_spin(self.autopilot.reg_nz, 'kp', "Бок. перегрузка (nz) Kp:")
        self.spin_nz_ki = self.create_spin(self.autopilot.reg_nz, 'ki', "Бок. перегрузка (nz) Ki:")

    def create_spin(self, controller, param_name, label_text):
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(getattr(controller, param_name))
        spin.valueChanged.connect(lambda val, c=controller, p=param_name: setattr(c, p, val))

        self.layout.addRow(label_text, spin)  # Добавляем строку в layout, который принадлежит диалогу
        return spin


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование БПЛА")
        self.resize(1100, 800)

        self.autopilot = Autopilot()
        self.uav_model = UAVModel()

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
        self.spin_Time = self.create_spinbox("Время T (c):", 150.0, 5.0, 500.0, layout_s)
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

        # Вкладка Динамика полета
        self.fig1 = Figure(figsize=(10, 8))
        self.canvas1 = FigureCanvas(self.fig1)
        self.tabs.addTab(self.canvas1, "Динамика полета")

        # Вкладка Перегрузки
        self.fig2 = Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        self.tabs.addTab(self.canvas2, "Перегрузки")

        # Вкладка Сигналы управления
        self.fig3 = Figure(figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.fig3)
        self.tabs.addTab(self.canvas3, "Сигналы управления")

        self.ax1_h = None
        self.ax1_v = None
        self.ax1_psi = None
        self.ax1_gamma = None

        self.ax2_nxa = None
        self.ax2_nya = None
        self.ax2_nza = None

        self.ax3_unxa = None
        self.ax3_unya = None
        self.ax3_unza = None
        self.ax3_ugamma = None

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
        self.autopilot.reset_controllers()

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

            controls = self.autopilot.calculate_controls(state, target_dict, self.uav_model, dt)

            U_nxa_hist[i] = controls[0]
            U_nya_hist[i] = controls[1]
            U_nza_hist[i] = controls[2]
            U_gamma_hist[i] = np.rad2deg(controls[3])

            state = self.uav_model.rk4_step(state, controls, dt)
            state[7] = np.clip(state[7], -8.0, 8.0)

        self.update_plot1(time_hist, H_hist, V_hist, Psi_hist, Gamma_hist, target_dict)
        self.update_plot2(time_hist, Nxa_hist, Nya_hist, Nza_hist)
        self.update_plot3(time_hist, U_nxa_hist, U_nya_hist, U_nza_hist, U_gamma_hist)

    def update_plot1(self, time_hist, H_hist, V_hist, Psi_hist, Gamma_hist, target_dict):
        self.fig1.clear()  
        self.ax1_h = self.fig1.add_subplot(2, 2, 1)
        self.ax1_v = self.fig1.add_subplot(2, 2, 2)
        self.ax1_psi = self.fig1.add_subplot(2, 2, 3)
        self.ax1_gamma = self.fig1.add_subplot(2, 2, 4)

        self.ax1_h.plot(time_hist, H_hist, 'b', linewidth=2, label=r'$H_{тек}$')
        self.ax1_h.plot(time_hist, [target_dict['H']] * len(time_hist), 'r--', label=r'$H_{зад}$')
        self.ax1_h.set_title("Высота")
        self.ax1_h.set_xlabel("Время, с")
        self.ax1_h.set_ylabel("H, м")
        self.ax1_h.grid(True)
        self.ax1_h.legend()

        self.ax1_v.plot(time_hist, V_hist, 'g', linewidth=2, label=r'$V_{тек}$')
        self.ax1_v.plot(time_hist, [target_dict['V']] * len(time_hist), 'r--', label=r'$V_{зад}$')
        self.ax1_v.set_title("Скорость")
        self.ax1_v.set_xlabel("Время, с")
        self.ax1_v.set_ylabel("V, м/с")
        self.ax1_v.grid(True)
        self.ax1_v.legend()

        self.ax1_psi.plot(time_hist, Psi_hist, 'purple', linewidth=2, label=r'$\Psi_{тек}$')
        self.ax1_psi.plot(time_hist, [np.rad2deg(target_dict['Psi'])] * len(time_hist), 'r--', label=r'$\Psi_{зад}$')
        self.ax1_psi.set_title("Курс")
        self.ax1_psi.set_xlabel("Время, с")
        self.ax1_psi.set_ylabel(r'$\Psi$, град')
        self.ax1_psi.grid(True)
        self.ax1_psi.legend()

        self.ax1_gamma.plot(time_hist, Gamma_hist, 'orange')
        self.ax1_gamma.plot(time_hist, [20] * len(time_hist), 'r:', alpha=0.5)
        self.ax1_gamma.plot(time_hist, [-20] * len(time_hist), 'r:', alpha=0.5)
        self.ax1_gamma.set_title(r"Крен ($\gamma$)")
        self.ax1_gamma.set_xlabel("Время, с")
        self.ax1_gamma.set_ylabel("град")
        self.ax1_gamma.grid(True)

        self.fig1.tight_layout()
        self.canvas1.draw()

    def update_plot2(self, time_hist, Nxa_hist, Nya_hist, Nza_hist):
        self.fig2.clear()  
        self.ax2_nxa = self.fig2.add_subplot(2, 2, 1)
        self.ax2_nya = self.fig2.add_subplot(2, 2, 2)
        self.ax2_nza = self.fig2.add_subplot(2, 2, 3)

        self.ax2_nxa.plot(time_hist, Nxa_hist, 'k')
        self.ax2_nxa.set_title(r"Продольная перегрузка ($n_{xa}$)")
        self.ax2_nxa.set_xlabel("Время, с")
        self.ax2_nxa.grid(True)

        self.ax2_nya.plot(time_hist, Nya_hist, 'k')
        self.ax2_nya.set_title(r"Нормальная перегрузка ($n_{ya}$)")
        self.ax2_nya.set_xlabel("Время, с")
        self.ax2_nya.grid(True)

        self.ax2_nza.plot(time_hist, Nza_hist, 'k')
        self.ax2_nza.set_title(r"Боковая перегрузка ($n_{za}$)")
        self.ax2_nza.set_xlabel("Время, с")
        self.ax2_nza.grid(True)

        self.fig2.tight_layout()
        self.canvas2.draw()

    def update_plot3(self, time_hist, U_nxa_hist, U_nya_hist, U_nza_hist, U_gamma_hist):
        self.fig3.clear()  
        self.ax3_unxa = self.fig3.add_subplot(2, 2, 1)
        self.ax3_unya = self.fig3.add_subplot(2, 2, 2)
        self.ax3_unza = self.fig3.add_subplot(2, 2, 3)
        self.ax3_ugamma = self.fig3.add_subplot(2, 2, 4)

        self.ax3_unxa.plot(time_hist, U_nxa_hist, 'm')
        self.ax3_unxa.set_title(r"Упр. скоростью ($u_{n_{xa}}$)")
        self.ax3_unxa.set_xlabel("Время, с")
        self.ax3_unxa.grid(True)

        self.ax3_unya.plot(time_hist, U_nya_hist, 'm')
        self.ax3_unya.set_title(r"Упр. высотой ($u_{n_{ya}}$)")
        self.ax3_unya.set_xlabel("Время, с")
        self.ax3_unya.grid(True)

        self.ax3_unza.plot(time_hist, U_nza_hist, 'm')
        self.ax3_unza.set_title(r"Упр. боковое ($u_{n_{za}}$)")
        self.ax3_unza.set_xlabel("Время, с")
        self.ax3_unza.grid(True)

        self.ax3_ugamma.plot(time_hist, U_gamma_hist, 'm')
        self.ax3_ugamma.set_title(r"Упр. креном ($u_{\gamma}$)")
        self.ax3_ugamma.set_xlabel("Время, с")
        self.ax3_ugamma.grid(True)

        self.fig3.tight_layout()
        self.canvas3.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

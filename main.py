import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# Класс П-регулятора

class P_Controller:
    """
    Закон управления: u = error * k
    """

    def __init__(self, k, min_val=None, max_val=None):
        self.k = k  # Коэффициент усиления
        self.min_val = min_val  # Нижнее ограничение
        self.max_val = max_val  # Верхнее ограничение

    def update(self, error, dt=None):
        output = error * self.k

        # Ограничение выхода
        if self.min_val is not None and self.max_val is not None:
            output = np.clip(output, self.min_val, self.max_val)

        return output


# Математическая модель БПЛА

class UAVModel:
    def __init__(self):
        self.g = 9.81

        # Параметры из схемы (постоянные времени)
        self.T_nxa = 0.5
        self.T_nya = 0.3
        self.xi_nya = 0.7
        self.T_nza = 0.5
        self.T_gamma = 0.2

        # Ограничение скорости Vmax
        self.V_max_limit = 100.0

    def get_derivatives(self, state, u_control):
        """
        Правые части ДУ
        state: [x, y, z, V, theta, psi, nxa, nya, d_nya, nza, gamma, d_gamma]
        """
        x, y_h, z, V, theta, psi = state[0:6]
        nxa, nya, d_nya, nza = state[6:10]
        gamma, d_gamma = state[10:12]

        u_nxa, u_nya, u_nza, u_gamma = u_control

        if V < 0.1: V = 0.1

        # Кинематика
        dx = V * np.cos(psi) * np.cos(theta)
        dy = V * np.sin(theta)
        dz = -V * np.sin(psi) * np.cos(theta)

        dV = self.g * (nxa - np.sin(theta))

        cos_theta = np.cos(theta)
        if abs(cos_theta) < 1e-3: cos_theta = 1e-3 * np.sign(cos_theta)

        dTheta = (self.g / V) * (nya * np.cos(gamma) - nza * np.sin(gamma) - cos_theta)
        dPsi = -(self.g / (V * cos_theta)) * (nya * np.sin(gamma) + nza * np.cos(gamma))

        # Динамика приводов
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


# Автопилот

class Autopilot:
    def __init__(self):

        # Контур скорости
        # K=0.5 (подобран)
        self.reg_V = P_Controller(k=0.5)

        # Контур высоты
        # Внешний контур (по высоте) огран выхода +-2
        self.reg_H_outer = P_Controller(k=0.5, min_val=-2.0, max_val=2.0)
        # Внутренний контур (по верт скорости) огран выхода +-10
        self.reg_H_inner = P_Controller(k=2.0, min_val=-10.0, max_val=10.0)

        # Контур курса
        # Огран крена +-20 градусов (в радианах)
        self.limit_gamma = np.deg2rad(20.0)
        # Внешний контур (по курсу)
        self.reg_Psi_outer = P_Controller(k=2.5, min_val=-self.limit_gamma, max_val=self.limit_gamma)
        # Внутренний контур (по крену)
        self.reg_Gamma_inner = P_Controller(k=4.0)

        # Контур боковой перегрузки
        self.reg_nz = P_Controller(k=1.0)

    def calculate_controls(self, state, targets, dt, uav_params):
        y_h = state[1]
        V = state[3]
        theta = state[4]
        psi = state[5]
        nya = state[7]
        nza = state[9]
        gamma = state[10]

        # Скорость
        V_err = targets['V'] - V
        # Ограничение ошибки ( +- Vmax/2)
        limit_dV = uav_params.V_max_limit / 2.0
        V_err_clamped = np.clip(V_err, -limit_dV, limit_dV)

        u_nxa = self.reg_V.update(V_err_clamped)

        #Высота
        H_err = targets['H'] - y_h

        # Внешний контур ошибка высоты требуемая H_dot
        H_dot_zad = self.reg_H_outer.update(H_err)

        # Внутренний контур ошибка H_dot -> перегрузка
        H_dot_curr = V * np.sin(theta)
        dH_dot = H_dot_zad - H_dot_curr

        # +1 для компенсации веса
        u_nya = self.reg_H_inner.update(dH_dot) + 1.0

        # Курс
        Psi_err = targets['Psi'] - psi
        # Корректировка угла (чтобы не крутиться через 360)
        while Psi_err > np.pi: Psi_err -= 2 * np.pi
        while Psi_err < -np.pi: Psi_err += 2 * np.pi

        # Внешний контур ошибка курса -> требуемый крен
        gamma_zad = self.reg_Psi_outer.update(Psi_err)

        # Внутренний контур ошибка крена
        gamma_err = gamma_zad - gamma
        u_gamma = self.reg_Gamma_inner.update(gamma_err)

        # Боковая перегрузка
        nz_err = 0.0 - nza
        u_nza = self.reg_nz.update(nz_err)

        return np.array([u_nxa, u_nya, u_nza, u_gamma])


# ГУИшка

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование БПЛА (П-регулятор)")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель настроек
        settings_panel = QWidget()
        settings_panel.setFixedWidth(250)
        settings_layout = QVBoxLayout(settings_panel)
        main_layout.addWidget(settings_panel)

        # Задание
        grp_targets = QGroupBox("Задание")
        layout_t = QVBoxLayout()
        self.spin_V_zad = self.create_spinbox("Скорость V (м/с):", 25.0, 10.0, 100.0, layout_t)
        self.spin_H_zad = self.create_spinbox("Высота H (м):", 100.0, 0.0, 5000.0, layout_t)
        self.spin_Psi_zad = self.create_spinbox("Курс Psi (град):", 45.0, -180.0, 360.0, layout_t)
        grp_targets.setLayout(layout_t)
        settings_layout.addWidget(grp_targets)

        # Начальные условия
        grp_init = QGroupBox("Начальные условия")
        layout_i = QVBoxLayout()
        self.spin_V0 = self.create_spinbox("V0 (м/с):", 20.0, 1.0, 100.0, layout_i)
        self.spin_H0 = self.create_spinbox("H0 (м):", 50.0, 0.0, 5000.0, layout_i)
        self.spin_Psi0 = self.create_spinbox("Psi0 (град):", 0.0, -180.0, 360.0, layout_i)
        grp_init.setLayout(layout_i)
        settings_layout.addWidget(grp_init)

        # Моделирование
        grp_sim = QGroupBox("Моделирование")
        layout_s = QVBoxLayout()
        self.spin_Time = self.create_spinbox("Время T (c):", 30.0, 5.0, 200.0, layout_s)
        self.spin_dt = self.create_spinbox("Шаг dt (c):", 0.05, 0.001, 0.5, layout_s, decimals=3)
        grp_sim.setLayout(layout_s)
        settings_layout.addWidget(grp_sim)

        # Кнопка
        self.btn_start = QPushButton("Запустить")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.run_simulation)
        settings_layout.addWidget(self.btn_start)
        settings_layout.addStretch()

        # Графики
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

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
        # Считывание параметров
        V_zad = self.spin_V_zad.value()
        H_zad = self.spin_H_zad.value()
        Psi_zad_deg = self.spin_Psi_zad.value()
        Psi_zad = np.deg2rad(Psi_zad_deg)

        V0 = self.spin_V0.value()
        H0 = self.spin_H0.value()
        Psi0 = np.deg2rad(self.spin_Psi0.value())

        T_max = self.spin_Time.value()
        dt = self.spin_dt.value()
        steps = int(T_max / dt)

        # Инициализация
        model = UAVModel()
        autopilot = Autopilot()

        # Начальный вектор состояния
        state = np.zeros(12)
        state[0] = 0.0  # x
        state[1] = H0  # H
        state[3] = V0  # V
        state[5] = Psi0  # Psi
        state[7] = 1.0  # nya

        # Массивы для графиков
        time_hist = np.linspace(0, T_max, steps)
        H_hist = np.zeros(steps)
        V_hist = np.zeros(steps)
        Psi_hist = np.zeros(steps)
        Gamma_hist = np.zeros(steps)
        Nya_hist = np.zeros(steps)

        target_dict = {'V': V_zad, 'H': H_zad, 'Psi': Psi_zad}

        # Цикл симуляции
        for i in range(steps):
            H_hist[i] = state[1]
            V_hist[i] = state[3]
            Psi_hist[i] = np.rad2deg(state[5])
            Gamma_hist[i] = np.rad2deg(state[10])
            Nya_hist[i] = state[7]

            # Расчет управления
            controls = autopilot.calculate_controls(state, target_dict, dt, model)

            state = model.rk4_step(state, controls, dt)

        # Отрисовка
        self.figure.clear()

        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.plot(time_hist, H_hist, 'b', label='H тек')
        ax1.plot(time_hist, [H_zad] * steps, 'r--', label='H зад')
        ax1.set_title("Высота (м)")
        ax1.grid(True)
        ax1.legend()

        ax2 = self.figure.add_subplot(2, 2, 2)
        ax2.plot(time_hist, V_hist, 'g', label='V тек')
        ax2.plot(time_hist, [V_zad] * steps, 'r--', label='V зад')
        ax2.set_title("Скорость (м/с)")
        ax2.grid(True)
        ax2.legend()

        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.plot(time_hist, Psi_hist, 'purple', label='Psi тек')
        ax3.plot(time_hist, [Psi_zad_deg] * steps, 'r--', label='Psi зад')
        ax3.set_title("Курс (град)")
        ax3.grid(True)
        ax3.legend()

        ax4 = self.figure.add_subplot(2, 2, 4)
        ax4.plot(time_hist, Gamma_hist, 'orange', label='Крен (град)')
        ax4.plot(time_hist, [20] * steps, 'r:', alpha=0.5)
        ax4.plot(time_hist, [-20] * steps, 'r:', alpha=0.5)
        ax4.set_title("Крен (град)")
        ax4.grid(True)
        ax4.legend()

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

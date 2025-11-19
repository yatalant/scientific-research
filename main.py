import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



# Классы для моделирования

class PID:
    # ПИД регулятор с ограничением выхода

    def __init__(self, kp, ki, kd, min_val=None, max_val=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_val = min_val
        self.max_val = max_val
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt

        # Не даем интегралу улетать в бесконечность
        if self.max_val is not None:
            # Огран интегральной части
            limit_i = max(abs(self.min_val), abs(self.max_val))
            self.integral = np.clip(self.integral, -limit_i, limit_i)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.min_val is not None and self.max_val is not None:
            output = np.clip(output, self.min_val, self.max_val)

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class UAVModel:
    def __init__(self):
        self.g = 9.81

        # Значения предположительно типичные для бпла
        self.T_nxa = 0.5
        self.T_nya = 0.3
        self.xi_nya = 0.7  # демпфирование
        self.T_nza = 0.5
        self.T_gamma = 0.2

        # Ограничение скорости для первого контура
        self.V_max_limit = 100.0

    def get_derivatives(self, state, u_control):
        """
        Вычисляет правые части ду: dy/dt = f(y, u)
        state: [x, y, z, V, theta, psi, nxa, nya, d_nya, nza, gamma, d_gamma]
        Индексы: 0  1  2  3  4      5    6    7    8      9    10     11
        u_control: [u_nxa, u_nya, u_nza, u_gamma]
        """
        # Распаковка состояния
        x, y_h, z, V, theta, psi = state[0:6]
        nxa, nya, d_nya, nza = state[6:10]
        gamma, d_gamma = state[10:12]

        u_nxa, u_nya, u_nza, u_gamma = u_control

        # От деления на ноль
        if V < 0.1: V = 0.1

        # Уравнения движения
        dx = V * np.cos(psi) * np.cos(theta)
        dy = V * np.sin(theta)  # y - высота H
        dz = -V * np.sin(psi) * np.cos(theta)

        dV = self.g * (nxa - np.sin(theta))

        # Косинус тэта в знаменателе может дать 0 при 90 град добавляем eps для защиты
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 1e-3: cos_theta = 1e-3 * np.sign(cos_theta)

        dTheta = (self.g / V) * (nya * np.cos(gamma) - nza * np.sin(gamma) - cos_theta)

        dPsi = -(self.g / (V * cos_theta)) * (nya * np.sin(gamma) + nza * np.cos(gamma))

        # Контур скорости nxa апериодическое звено 1 порядка
        dnxa = (u_nxa - nxa) / self.T_nxa

        # Контур высоты перегрузка ny: колебательное звено 2 порядка
        # d(nya)/dt = d_nya (переменная состояния 8)
        dd_nya = (u_nya - 2 * self.xi_nya * self.T_nya * d_nya - nya) / (self.T_nya ** 2)

        # Контур боковой перегрузки nz: апериодическое звено
        dnza = (u_nza - nza) / self.T_nza

        # Контур крена гамма: интегрирующее звено с запаздыванием
        dd_gamma = (u_gamma - d_gamma) / self.T_gamma

        return np.array([dx, dy, dz, dV, dTheta, dPsi, dnxa, d_nya, dd_nya, dnza, d_gamma, dd_gamma])

    def rk4_step(self, state, u_control, dt):
        # Один шаг интегрирования
        k1 = self.get_derivatives(state, u_control)
        k2 = self.get_derivatives(state + 0.5 * dt * k1, u_control)
        k3 = self.get_derivatives(state + 0.5 * dt * k2, u_control)
        k4 = self.get_derivatives(state + dt * k3, u_control)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state


class Autopilot:
    # Контуры управления

    def __init__(self):
        # Создаем пид регуляторы и ошибку ограничим
        self.pid_V = PID(kp=0.5, ki=0.05, kd=0.1)

        # Контур высоты
        # Внешний контур
        self.pid_H_outer = PID(kp=0.5, ki=0.0, kd=0.0, min_val=-2.0, max_val=2.0)  # +-2 огран
        # Внутренний контур
        self.pid_H_inner = PID(kp=2.0, ki=0.1, kd=0.1, min_val=-10.0, max_val=10.0)  # Выход - перегрузка

        # Контур курса пси
        # Внешний контур
        self.limit_gamma = np.deg2rad(20.0)  # +-20
        self.pid_Psi_outer = PID(kp=2.0, ki=0.0, kd=0.0, min_val=-self.limit_gamma, max_val=self.limit_gamma)
        # Внутренний контур
        self.pid_Gamma_inner = PID(kp=5.0, ki=0.0, kd=1.0)

        # Контур боковой перегрузки (nz)
        self.pid_nz = PID(kp=1.0, ki=0.1, kd=0.0)

    def calculate_controls(self, state, targets, dt, uav_params):

        # state текущий вектор состояния
        # targets словарь с целевыми значениями
        # Извлекаем переменные (в радианах и СИ)
        y_h = state[1]
        V = state[3]
        theta = state[4]
        psi = state[5]
        nya = state[7]
        nza = state[9]
        gamma = state[10]

        # Канал скорости V
        V_err = targets['V'] - V

        # Ограничение ошибки dV (+- Vmax/2)
        # Пусть Vmax = 100
        limit_dV = uav_params.V_max_limit / 2.0
        V_err_clamped = np.clip(V_err, -limit_dV, limit_dV)

        u_nxa = self.pid_V.update(V_err_clamped, dt)

        # Канал высоты H
        H_err = targets['H'] - y_h

        # Внешний контур получаем желаемую вертикальную скорость
        H_dot_zad = self.pid_H_outer.update(H_err, dt)  #  +-2

        # Текущая верт скорость
        H_dot_curr = V * np.sin(theta)

        dH_dot = H_dot_zad - H_dot_curr

        # Внутренний контур + компенсация тяжести
        # Выход внутреннего ПИД + 1 = u_nya
        u_nya = self.pid_H_inner.update(dH_dot, dt) + 1.0

        # Канал курса пси
        # Обработка перехода через 360 градусов для корректной ошибки
        Psi_err = targets['Psi'] - psi
        while Psi_err > np.pi: Psi_err -= 2 * np.pi
        while Psi_err < -np.pi: Psi_err += 2 * np.pi

        # Внешний контур получаем желаемый крен
        gamma_zad = self.pid_Psi_outer.update(Psi_err, dt)  #  +-20 град

        # Внутренний контур
        gamma_err = gamma_zad - gamma
        u_gamma = self.pid_Gamma_inner.update(gamma_err, dt)

        # Канал боковой перегрузки (Nz)
        # Задача держать 0
        nz_err = 0.0 - nza
        u_nza = self.pid_nz.update(nz_err, dt)

        return np.array([u_nxa, u_nya, u_nza, u_gamma])

    def reset(self):
        self.pid_V.reset()
        self.pid_H_outer.reset()
        self.pid_H_inner.reset()
        self.pid_Psi_outer.reset()
        self.pid_Gamma_inner.reset()
        self.pid_nz.reset()


# ГУИшка


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование динамики БПЛА")
        self.resize(1000, 700)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Панелька настроек
        settings_panel = QWidget()
        settings_panel.setFixedWidth(250)
        settings_layout = QVBoxLayout(settings_panel)
        main_layout.addWidget(settings_panel)

        # Целевые параметры
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

        # Параметры моделирования
        grp_sim = QGroupBox("Моделирование")
        layout_s = QVBoxLayout()
        self.spin_Time = self.create_spinbox("Время T (c):", 30.0, 5.0, 200.0, layout_s)
        self.spin_dt = self.create_spinbox("Шаг dt (c):", 0.05, 0.001, 0.5, layout_s, decimals=3)
        grp_sim.setLayout(layout_s)
        settings_layout.addWidget(grp_sim)

        # Кнопка запуска
        self.btn_start = QPushButton("Запустить моделирование")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.run_simulation)
        settings_layout.addWidget(self.btn_start)

        settings_layout.addStretch()  # Пустое место снизу

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
        # Считываем параметры из интерфейса
        V_zad = self.spin_V_zad.value()
        H_zad = self.spin_H_zad.value()
        Psi_zad_deg = self.spin_Psi_zad.value()
        Psi_zad = np.deg2rad(Psi_zad_deg)  # Перевод в радианы

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
        state[1] = H0  # y (H)
        state[2] = 0.0  # z
        state[3] = V0  # V
        state[4] = 0.0  # Theta (пусть горизонтально летит)
        state[5] = Psi0  # Psi
        state[6] = 0.0  # nxa
        state[7] = 1.0  # nya (в гориз полете перегрузка = 1)
        state[8] = 0.0  # d_nya
        state[9] = 0.0  # nza
        state[10] = 0.0  # gamma
        state[11] = 0.0  # d_gamma

        # Массивы для хранения истории (для графиков)
        time_hist = np.linspace(0, T_max, steps)
        H_hist = np.zeros(steps)
        V_hist = np.zeros(steps)
        Psi_hist = np.zeros(steps)
        Gamma_hist = np.zeros(steps)
        Nya_hist = np.zeros(steps)

        target_dict = {'V': V_zad, 'H': H_zad, 'Psi': Psi_zad}

        # Цикл моделирования
        for i in range(steps):
            # Сохраняем текущие значения
            H_hist[i] = state[1]
            V_hist[i] = state[3]
            Psi_hist[i] = np.rad2deg(state[5])  # Сохраняем в градусах
            Gamma_hist[i] = np.rad2deg(state[10])
            Nya_hist[i] = state[7]

            # Шаг управления (расчет U)
            controls = autopilot.calculate_controls(state, target_dict, dt, model)

            # Шаг Рунге-Кутта
            state = model.rk4_step(state, controls, dt)

        # Графики
        self.figure.clear()

        # График 1 Высота
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.plot(time_hist, H_hist, label='H тек', color='blue')
        ax1.plot(time_hist, [H_zad] * steps, 'r--', label='H зад')
        ax1.set_title("Высота (м)")
        ax1.grid(True)
        ax1.legend()

        # График 2 Скорость
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax2.plot(time_hist, V_hist, label='V тек', color='green')
        ax2.plot(time_hist, [V_zad] * steps, 'r--', label='V зад')
        ax2.set_title("Скорость (м/с)")
        ax2.grid(True)
        ax2.legend()

        # График 3 Курс
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.plot(time_hist, Psi_hist, label='Psi тек', color='purple')
        ax3.plot(time_hist, [Psi_zad_deg] * steps, 'r--', label='Psi зад')
        ax3.set_title("Курс (град)")
        ax3.grid(True)
        ax3.legend()

        # График 4 Крен и Перегрузка
        ax4 = self.figure.add_subplot(2, 2, 4)
        ax4.plot(time_hist, Gamma_hist, label='Крен (град)', color='orange')
        ax4.plot(time_hist, Nya_hist, label='Ny (g)', color='black', alpha=0.5)
        # Линия ограничения крена
        ax4.plot(time_hist, [20] * steps, 'r:', alpha=0.5)
        ax4.plot(time_hist, [-20] * steps, 'r:', alpha=0.5)
        ax4.set_title("Параметры управления")
        ax4.grid(True)
        ax4.legend()

        self.figure.tight_layout()
        self.canvas.draw()


# Создаем приложение, окно, бесконечный цикл


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
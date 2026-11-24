import sys
import json
import socket
import time
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTabWidget, QDialog, QFormLayout,
                             QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView)
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont, QBrush
from PyQt6.QtCore import Qt, QTimer, QPointF
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QComboBox, QLineEdit, QMessageBox


class PrimaryFlightDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 500)
        self.pitch = 0.0
        self.roll = 0.0
        self.hdg = 0.0
        self.alt = 0.0
        self.spd = 0.0
        self.vsi = 0.0
        self.nx = 0.0
        self.ny = 1.0
        self.nz = 0.0

    def update_data(self, pitch, roll, hdg, alt, spd, vsi, nx, ny, nz):
        self.pitch = pitch
        self.roll = roll
        self.hdg = hdg
        self.alt = alt
        self.spd = spd
        self.vsi = vsi
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(20, 20, 20))

        cx, cy = w / 2, h / 2 - 20
        radius = 180
        font = painter.font()

        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(cx - radius, cy - radius, radius * 2, radius * 2, 15, 15)
        painter.setClipPath(clip_path)

        painter.translate(cx, cy)
        painter.rotate(self.roll)
        pitch_scale = 4.0
        painter.translate(0, self.pitch * pitch_scale)

        painter.fillRect(-400, -600, 800, 600, QColor(187, 224, 245))
        painter.fillRect(-400, 0, 800, 600, QColor(240, 200, 100))
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawLine(-400, 0, 400, 0)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        for i in range(-90, 100, 10):
            if i == 0: continue
            y = -i * pitch_scale
            w_line = 30 if i % 20 == 0 else 15
            painter.drawLine(-w_line, int(y), w_line, int(y))
            if abs(i) % 10 == 0:
                painter.drawText(-w_line - 25, int(y) + 5, str(abs(i)))
                painter.drawText(w_line + 5, int(y) + 5, str(abs(i)))
        painter.restore()

        painter.save()
        painter.translate(cx, cy)
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawArc(-radius, -radius, radius * 2, radius * 2, 30 * 16, 120 * 16)
        for angle in [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]:
            painter.save()
            painter.rotate(angle)
            length = 15 if angle % 30 == 0 else 8
            painter.drawLine(0, -radius, 0, -radius + length)
            painter.restore()

        painter.rotate(self.roll)
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawPolygon(QPointF(0, -radius), QPointF(-10, -radius + 15), QPointF(10, -radius + 15))
        painter.restore()

        painter.setPen(QPen(Qt.GlobalColor.magenta, 4))
        painter.drawLine(int(cx - 70), int(cy), int(cx - 20), int(cy))
        painter.drawLine(int(cx + 20), int(cy), int(cx + 70), int(cy))
        painter.drawLine(int(cx), int(cy), int(cx), int(cy + 15))
        painter.setBrush(QBrush(Qt.GlobalColor.magenta))
        painter.drawEllipse(int(cx - 4), int(cy - 4), 8, 8)

        tape_x = cx - radius - 80
        tape_w = 60
        painter.fillRect(int(tape_x), int(cy - radius), tape_w, radius * 2, QColor(50, 50, 50, 200))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))

        speed_scale = 8.0
        for v in range(max(0, int(self.spd) - 30), int(self.spd) + 30):
            if v % 5 == 0:
                y = cy - (v - self.spd) * speed_scale
                if cy - radius < y < cy + radius:
                    w_line = 10 if v % 10 == 0 else 5
                    painter.drawLine(int(tape_x + tape_w - w_line), int(y), int(tape_x + tape_w), int(y))
                    if v % 10 == 0:
                        painter.drawText(int(tape_x + 5), int(y + 5), str(v))

        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        painter.drawRect(int(tape_x - 5), int(cy - 15), tape_w + 10, 30)
        font.setPixelSize(16)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.green)
        painter.drawText(int(tape_x), int(cy + 5), f"{self.spd:.1f}")

        font.setPixelSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.cyan)
        painter.drawText(int(tape_x), int(cy + radius + 25), "Скорость, м/с")

        g_x = tape_x - 130
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)

        def draw_g_bar(x, y, name, val, min_v, max_v, normal_v, limit_min, limit_max):
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(x - 5, y - 5, f"{val:.1f}")
            painter.drawRect(x, y, 12, 100)
            color = Qt.GlobalColor.green if (limit_min <= val <= limit_max) else Qt.GlobalColor.red
            val_c = np.clip(val, min_v, max_v)
            fill_h = ((val_c - min_v) / (max_v - min_v)) * 100
            painter.fillRect(x + 1, int(y + 100 - fill_h), 11, int(fill_h), color)
            norm_y = y + 100 - ((normal_v - min_v) / (max_v - min_v)) * 100
            painter.setPen(Qt.GlobalColor.white)
            painter.drawLine(x - 3, int(norm_y), x + 15, int(norm_y))
            painter.drawText(x, y + 115, name)

        draw_g_bar(g_x, int(cy - 50), "Nx", self.nx, -1, 2, 0, -0.5, 1.5)
        draw_g_bar(g_x + 40, int(cy - 50), "Ny", self.ny, -2, 6, 1, 0.0, 5.0)
        draw_g_bar(g_x + 80, int(cy - 50), "Nz", self.nz, -2, 2, 0, -1.0, 1.0)

        font.setPixelSize(14)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.cyan)
        painter.drawText(int(g_x + 10), int(cy + 90), "Перегрузки")

        tape_x_alt = cx + radius + 20
        painter.fillRect(int(tape_x_alt), int(cy - radius), tape_w, radius * 2, QColor(50, 50, 50, 200))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)

        alt_scale = 1.0
        for h_tick in range(max(0, int(self.alt) - 200), int(self.alt) + 200):
            if h_tick % 20 == 0:
                y = cy - (h_tick - self.alt) * alt_scale
                if cy - radius < y < cy + radius:
                    w_line = 10 if h_tick % 100 == 0 else 5
                    painter.drawLine(int(tape_x_alt), int(y), int(tape_x_alt + w_line), int(y))
                    if h_tick % 100 == 0:
                        painter.drawText(int(tape_x_alt + 15), int(y + 5), str(h_tick))

        painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        painter.drawRect(int(tape_x_alt), int(cy - 15), tape_w + 15, 30)
        font.setPixelSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.yellow)
        painter.drawText(int(tape_x_alt + 5), int(cy + 5), f"{self.alt:.1f}")

        font.setPixelSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.cyan)
        painter.drawText(int(tape_x_alt), int(cy + radius + 25), "Высота, м")

        hdg_y = cy + radius + 20
        hdg_w = 300
        painter.fillRect(int(cx - hdg_w / 2), int(hdg_y), hdg_w, 40, QColor(50, 50, 50, 200))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        font.setPixelSize(12)
        painter.setFont(font)

        hdg_scale = 3.0
        for h_tick in range(int(self.hdg) - 60, int(self.hdg) + 60):
            if h_tick % 10 == 0:
                x = cx + (h_tick - self.hdg) * hdg_scale
                if cx - hdg_w / 2 < x < cx + hdg_w / 2:
                    painter.drawLine(int(x), int(hdg_y), int(x), int(hdg_y + 10))
                    norm_h = h_tick % 360
                    if norm_h < 0: norm_h += 360
                    if h_tick % 30 == 0:
                        painter.drawText(int(x - 10), int(hdg_y + 25), str(norm_h))

        painter.setPen(QPen(Qt.GlobalColor.magenta, 2))
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        painter.drawRect(int(cx - 22), int(hdg_y - 5), 44, 25)

        painter.drawPolygon(QPointF(cx - 6, hdg_y - 5), QPointF(cx + 6, hdg_y - 5), QPointF(cx, hdg_y - 12))

        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.magenta)
        val_str = f"{self.hdg % 360:.0f}"
        offset = 12 if len(val_str) > 2 else (8 if len(val_str) > 1 else 4)
        painter.drawText(int(cx - offset), int(hdg_y + 13), val_str)

        font.setPixelSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.cyan)
        painter.drawText(int(cx - 40), int(hdg_y + 60), "Курс, град")

        vsi_x = cx + radius + 130
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(int(vsi_x), int(cy - 150), int(vsi_x), int(cy + 150))
        painter.drawLine(int(vsi_x - 10), int(cy), int(vsi_x + 10), int(cy))

        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(int(vsi_x + 15), int(cy - 145), "+15")
        painter.drawText(int(vsi_x + 15), int(cy + 155), "-15")

        font.setPixelSize(12)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.cyan)
        painter.drawText(int(vsi_x - 15), int(cy + 175), "dH/dt, м/с")

        vsi_y = cy - np.clip(self.vsi, -15, 15) * 10.0
        painter.setPen(QPen(Qt.GlobalColor.cyan, 4))
        painter.drawLine(int(vsi_x), int(cy), int(vsi_x), int(vsi_y))
        painter.drawLine(int(vsi_x), int(vsi_y), int(vsi_x + 5), int(vsi_y + 5))
        painter.drawLine(int(vsi_x), int(vsi_y), int(vsi_x - 5), int(vsi_y + 5))


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
        self.reg_V = PI_Controller(kp=0.10, ki=0.01, min_val=-10.0, max_val=10.0, integral_limit=1.0)
        self.reg_H_outer = PI_Controller(kp=0.04, ki=0.0, min_val=-2.0, max_val=2.0, integral_limit=1.0)
        self.reg_H_inner = PI_Controller(kp=0.80, ki=0.1, min_val=-10.0, max_val=10.0, integral_limit=5.0)
        self.limit_gamma = np.deg2rad(45.0)
        self.reg_Psi_outer = PI_Controller(kp=-0.80, ki=0.0, min_val=-self.limit_gamma, max_val=self.limit_gamma,
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
        y_h, V, theta, psi = state[1], state[3], state[4], state[5]
        nya, nza, gamma = state[7], state[9], state[10]

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
        self.setWindowTitle("Настройка коэффициентов регулятора")
        self.autopilot = autopilot
        self.resize(350, 300)
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.create_spin(self.autopilot.reg_V, 'kp', "V (Kp):")
        self.create_spin(self.autopilot.reg_V, 'ki', "V (Ki):")
        self.create_spin(self.autopilot.reg_H_outer, 'kp', "H (Kp):")
        self.create_spin(self.autopilot.reg_H_inner, 'kp', "H_dot (Kp):")
        self.create_spin(self.autopilot.reg_H_inner, 'ki', "H_dot (Ki):")
        self.create_spin(self.autopilot.reg_Psi_outer, 'kp', "Psi (Kp):")
        self.create_spin(self.autopilot.reg_Gamma_inner, 'kp', "Gamma (Kp):")
        self.create_spin(self.autopilot.reg_Gamma_inner, 'ki', "Gamma (Ki):")
        self.create_spin(self.autopilot.reg_nz, 'kp', "nz (Kp):")
        self.create_spin(self.autopilot.reg_nz, 'ki', "nz (Ki):")

    def create_spin(self, controller, param_name, label_text):
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(getattr(controller, param_name))
        spin.valueChanged.connect(lambda val, c=controller, p=param_name: setattr(c, p, val))
        self.layout.addRow(label_text, spin)
        return spin


class UdpJson:
    def __init__(self, listen_port=None, destination_host=None, destination_port=None):
        self.rx = None
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.destination = None
        if destination_host is not None and destination_port is not None:
            self.destination = (str(destination_host), int(destination_port))
        if listen_port is not None:
            self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rx.bind(("0.0.0.0", int(listen_port)))
            self.rx.setblocking(False)

    def send(self, payload):
        if self.destination is None or self.tx is None: return False
        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.tx.sendto(data, self.destination)
            return True
        except OSError:
            return False

    def receive_latest(self):
        if self.rx is None: return None
        latest = None
        while True:
            try:
                data, _addr = self.rx.recvfrom(65535)
                packet = json.loads(data.decode("utf-8"))
                if isinstance(packet, dict): latest = packet
            except BlockingIOError:
                break
            except Exception:
                continue
        return latest

    def close(self):
        for sock in (self.rx, self.tx):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.rx = None
        self.tx = None


class TargetSubsystem:
    def __init__(self, x0, north0, h0, v0, psi0_deg, guidance_host, guidance_port):
        self.model = UAVModel()
        self.autopilot = Autopilot()
        self.state = np.zeros(12, dtype=float)
        self.state[0], self.state[1], self.state[2] = float(x0), float(h0), -float(north0)
        self.state[3] = float(v0)
        self.state[5] = np.deg2rad(float(psi0_deg))
        self.state[7] = 1.0
        self.reference = {"V": float(v0), "H": float(h0), "Psi": self.state[5]}
        self.link = UdpJson(destination_host=guidance_host, destination_port=guidance_port)

    def step(self, dt):
        controls = self.autopilot.calculate_controls(self.state, self.reference, self.model, dt)
        self.state = self.model.rk4_step(self.state, controls, dt)
        self.state[7] = np.clip(self.state[7], -8.0, 8.0)

    def publish(self, sim_time):
        self.link.send({
            "type": "target_state", "time": float(sim_time),
            "x": float(self.state[0]), "north": float(-self.state[2]),
            "h": float(self.state[1]), "v": float(max(0.1, self.state[3])),
            "psi": float(self.state[5]), "theta": float(self.state[4])
        })

    def close(self):
        self.link.close()


class GuidanceSubsystem:
    def __init__(self, target_port, interceptor_telemetry_port, interceptor_host, interceptor_command_port):
        self.target_link = UdpJson(listen_port=target_port)
        self.interceptor_link = UdpJson(listen_port=interceptor_telemetry_port)
        self.command_link = UdpJson(destination_host=interceptor_host, destination_port=interceptor_command_port)

        self.latest_target = None
        self.latest_interceptor = None
        self.c_nav = 4.0
        self.last_error = "Ожидание телеметрии..."

    def make_command(self):
        tgt_pkt = self.target_link.receive_latest()
        int_pkt = self.interceptor_link.receive_latest()
        if tgt_pkt: self.latest_target = tgt_pkt
        if int_pkt: self.latest_interceptor = int_pkt

        if not self.latest_target or not self.latest_interceptor:
            return None

        tx, ty, th = self.latest_target["x"], self.latest_target["north"], self.latest_target["h"]
        ix, iy, ih = self.latest_interceptor["x"], self.latest_interceptor["north"], self.latest_interceptor["h"]

        tv, tpsi, ttheta = self.latest_target["v"], self.latest_target["psi"], self.latest_target.get("theta", 0.0)
        iv, ipsi, itheta = self.latest_interceptor["v"], self.latest_interceptor["psi"], self.latest_interceptor.get(
            "theta", 0.0)

        dx = tx - ix
        dy = ty - iy
        dh = th - ih

        vtx = tv * np.cos(tpsi) * np.cos(ttheta)
        vty = tv * np.sin(tpsi) * np.cos(ttheta)
        vtz = tv * np.sin(ttheta)

        vix = iv * np.cos(ipsi) * np.cos(itheta)
        viy = iv * np.sin(ipsi) * np.cos(itheta)
        viz = iv * np.sin(itheta)

        dvx = vtx - vix
        dvy = vty - viy
        dvz = vtz - viz

        r2d_sq = dx ** 2 + dy ** 2
        r2d = np.sqrt(r2d_sq)
        r3d_sq = r2d_sq + dh ** 2
        r3d = np.sqrt(r3d_sq)

        if r2d_sq > 0.1:
            az_rate = (dx * dvy - dy * dvx) / r2d_sq
            r2d_dot = (dx * dvx + dy * dvy) / r2d
        else:
            az_rate, r2d_dot = 0.0, 0.0

        if r3d_sq > 0.1:
            el_rate = (r2d * dvz - dh * r2d_dot) / r3d_sq
        else:
            el_rate = 0.0

        W_lat = self.c_nav * iv * az_rate
        W_vert = self.c_nav * iv * el_rate

        g = 9.81
        gamma_cmd = np.arctan(-W_lat / g)
        gamma_cmd = np.clip(gamma_cmd, -np.deg2rad(60.0), np.deg2rad(60.0))

        ny_cmd = (W_vert / g) + (1.0 / np.cos(gamma_cmd))
        ny_cmd = np.clip(ny_cmd, -2.0, 8.0)

        cmd = {
            "type": "pn_command",
            "gamma_cmd": float(gamma_cmd),
            "ny_cmd": float(ny_cmd)
        }
        self.command_link.send(cmd)
        self.last_error = f"Наведение активно. Дистанция: {r3d:.0f} м"
        return cmd

    def close(self):
        self.target_link.close()
        self.interceptor_link.close()
        self.command_link.close()


class InterceptorSubsystem:
    def __init__(self, x0, north0, h0, v0, psi0_deg, command_port, guidance_host, telemetry_port):
        self.model = UAVModel()
        self.autopilot = Autopilot()
        self.state = np.zeros(12, dtype=float)
        self.state[0], self.state[1], self.state[2] = float(x0), float(h0), -float(north0)
        self.state[3] = float(v0)
        self.state[5] = np.deg2rad(float(psi0_deg))
        self.state[7] = 1.0
        self.command_link = UdpJson(listen_port=command_port)
        self.telemetry_link = UdpJson(destination_host=guidance_host, destination_port=telemetry_port)
        self.latest_command = None

    def publish_telemetry(self, sim_time):
        self.telemetry_link.send({
            "type": "interceptor_state", "time": float(sim_time),
            "x": float(self.state[0]), "north": float(-self.state[2]),
            "h": float(self.state[1]), "v": float(max(0.1, self.state[3])),
            "psi": float(self.state[5]), "theta": float(self.state[4])
        })

    def receive_command(self):
        cmd = self.command_link.receive_latest()
        if cmd and cmd.get("type") == "pn_command":
            self.latest_command = cmd
        return self.latest_command

    def step(self, command, dt):
        if not command: return

        nya, nza, gamma = self.state[7], self.state[9], self.state[10]

        u_nxa = self.autopilot.reg_V.update(100.0 - self.state[3], dt)
        u_nya = float(command["ny_cmd"])
        u_gamma = self.autopilot.reg_Gamma_inner.update(float(command["gamma_cmd"]) - gamma, dt)
        u_nza = self.autopilot.reg_nz.update(0.0 - nza, dt)

        controls = np.array([u_nxa, u_nya, u_nza, u_gamma])
        self.state = self.model.rk4_step(self.state, controls, dt)
        self.state[7] = np.clip(self.state[7], -10.0, 10.0)

    def close(self):
        self.command_link.close()
        self.telemetry_link.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование БПЛА (UDP)")
        self.resize(1200, 800)

        self.autopilot = Autopilot()
        self.uav_model = UAVModel()

        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.on_live_tick)
        self.dt = 0.02
        self.live_state = None
        self.live_current_wp = 0
        self.live_waypoints = []

        self.h_time, self.h_X, self.h_Y, self.h_Z = [], [], [], []
        self.h_V, self.h_Psi, self.h_Gamma = [], [], []
        self.h_Nxa, self.h_Nya, self.h_Nza = [], [], []
        self.h_U_nxa, self.h_U_nya, self.h_U_nza, self.h_U_gamma = [], [], [], []
        self.h_Vzad, self.h_Hzad, self.h_Psizad = [], [], []
        self.tick_counter = 0
        self.ui_tick_counter = 0

        self.intercept_timer = QTimer()
        self.intercept_timer.timeout.connect(self.on_intercept_tick)
        self.intercept_target = None
        self.intercept_vehicle = None
        self.intercept_guidance = None
        self.intercept_step = 0
        self.intercept_dt = 0.02
        self.intercept_time = 0.0
        self.intercept_history = {"target_x": [], "target_y": [], "target_h": [],
                                  "vehicle_x": [], "vehicle_y": [], "vehicle_h": []}

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.left_tabs = QTabWidget()
        self.left_tabs.setFixedWidth(320)
        main_layout.addWidget(self.left_tabs)

        tab_std = QWidget()
        layout_std = QVBoxLayout(tab_std)
        grp_targets_std = QGroupBox("Задание")
        form_t_std = QFormLayout()
        self.spin_V_std = self.create_spinbox("V (м/с):", 25.0, 10.0, 100.0)
        self.spin_H_std = self.create_spinbox("H (м):", 100.0, 0.0, 5000.0)
        self.spin_Psi_std = self.create_spinbox("Psi (град):", 45.0, -180.0, 360.0)
        form_t_std.addRow(self.spin_V_std[0], self.spin_V_std[1])
        form_t_std.addRow(self.spin_H_std[0], self.spin_H_std[1])
        form_t_std.addRow(self.spin_Psi_std[0], self.spin_Psi_std[1])
        grp_targets_std.setLayout(form_t_std)
        layout_std.addWidget(grp_targets_std)
        grp_init_std = QGroupBox("Начальные условия")
        form_i_std = QFormLayout()
        self.spin_V0_std = self.create_spinbox("V0 (м/с):", 20.0, 1.0, 100.0)
        self.spin_H0_std = self.create_spinbox("H0 (м):", 50.0, 0.0, 5000.0)
        self.spin_Psi0_std = self.create_spinbox("Psi0 (град):", 0.0, -180.0, 360.0)
        form_i_std.addRow(self.spin_V0_std[0], self.spin_V0_std[1])
        form_i_std.addRow(self.spin_H0_std[0], self.spin_H0_std[1])
        form_i_std.addRow(self.spin_Psi0_std[0], self.spin_Psi0_std[1])
        grp_init_std.setLayout(form_i_std)
        layout_std.addWidget(grp_init_std)
        grp_sim_std = QGroupBox("Параметры симуляции")
        row_time_std = QHBoxLayout()
        self.spin_Time_std = self.create_spinbox("Время T(c):", 150.0, 5.0, 500.0)
        row_time_std.addWidget(self.spin_Time_std[0])
        row_time_std.addWidget(self.spin_Time_std[1])
        grp_sim_std.setLayout(row_time_std)
        layout_std.addWidget(grp_sim_std)
        self.btn_settings_std = QPushButton("Настройки ПИ-регулятора")
        self.btn_settings_std.clicked.connect(self.open_settings)
        layout_std.addWidget(self.btn_settings_std)
        self.btn_start_std = QPushButton("Запустить моделирование")
        self.btn_start_std.setStyleSheet("background-color: #FF69B4; color: white; font-weight: bold; padding: 10px;")
        self.btn_start_std.clicked.connect(self.run_simulation)
        layout_std.addWidget(self.btn_start_std)
        layout_std.addStretch()
        self.left_tabs.addTab(tab_std, "Точка")

        tab_rte = QWidget()
        layout_rte = QVBoxLayout(tab_rte)
        grp_targets_rte = QGroupBox("Задание")
        form_t_rte = QFormLayout()
        self.spin_V_rte = self.create_spinbox("V (м/с):", 25.0, 10.0, 100.0)
        form_t_rte.addRow(self.spin_V_rte[0], self.spin_V_rte[1])
        grp_targets_rte.setLayout(form_t_rte)
        layout_rte.addWidget(grp_targets_rte)
        grp_init_rte = QGroupBox("Начальные условия")
        form_i_rte = QFormLayout()
        self.spin_V0_rte = self.create_spinbox("V0 (м/с):", 20.0, 1.0, 100.0)
        self.spin_H0_rte = self.create_spinbox("H0 (м):", 0.0, 0.0, 5000.0)
        self.spin_Psi0_rte = self.create_spinbox("Psi0 (град):", 0.0, -180.0, 360.0)
        form_i_rte.addRow(self.spin_V0_rte[0], self.spin_V0_rte[1])
        form_i_rte.addRow(self.spin_H0_rte[0], self.spin_H0_rte[1])
        form_i_rte.addRow(self.spin_Psi0_rte[0], self.spin_Psi0_rte[1])
        grp_init_rte.setLayout(form_i_rte)
        layout_rte.addWidget(grp_init_rte)
        grp_table = QGroupBox("Точки маршрута (X, Y, Z)")
        layout_tbl = QVBoxLayout()
        self.wp_table = QTableWidget(0, 3)
        self.wp_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.wp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.wp_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.wp_table.setFixedHeight(180)
        default_waypoints = [(3000, 800, 50), (3000, 1800, 50), (0, 1800, 50), (0, 800, 50), (2850, 800, 50)]
        for pt in default_waypoints:
            self.add_wp_row(pt[0], pt[1], pt[2])
        layout_tbl.addWidget(self.wp_table)
        row_btns = QHBoxLayout()
        btn_add = QPushButton("Добавить точку")
        btn_add.clicked.connect(lambda: self.add_wp_row(0, 0, 150))
        btn_del = QPushButton("Удалить")
        btn_del.clicked.connect(self.del_wp_row)
        row_btns.addWidget(btn_add)
        row_btns.addWidget(btn_del)
        layout_tbl.addLayout(row_btns)
        grp_table.setLayout(layout_tbl)
        layout_rte.addWidget(grp_table)
        self.btn_settings_rte = QPushButton("Настройки ПИ-регулятора")
        self.btn_settings_rte.clicked.connect(self.open_settings)
        layout_rte.addWidget(self.btn_settings_rte)
        self.btn_start_rte = QPushButton("Запустить полёт по маршруту")
        self.btn_start_rte.setStyleSheet("background-color: #DA70D6; color: white; font-weight: bold; padding: 10px;")
        self.btn_start_rte.clicked.connect(lambda: self.start_live_simulation(switch_tab=False))
        layout_rte.addWidget(self.btn_start_rte)
        layout_rte.addStretch()
        self.left_tabs.addTab(tab_rte, "Маршрут")

        tab_intercept = QWidget()
        layout_intercept = QVBoxLayout(tab_intercept)

        grp_target = QGroupBox("БПЛА-цель: начальные условия")
        form_target = QFormLayout(grp_target)
        self.intercept_tx0 = self.create_spinbox("X0, м:", 2000.0, -100000.0, 100000.0)[1]
        self.intercept_ty0 = self.create_spinbox("Y0, м:", 2000.0, -100000.0, 100000.0)[1]
        self.intercept_th0 = self.create_spinbox("H0, м:", 500.0, 0.0, 5000.0)[1]
        self.intercept_tv0 = self.create_spinbox("V0, м/с:", 20.0, 10.0, 100.0)[1]
        self.intercept_tpsi0 = self.create_spinbox("Курс, град (270 = Юг):", 270.0, -180.0, 360.0)[1]
        form_target.addRow("X0, м:", self.intercept_tx0)
        form_target.addRow("Y0, м:", self.intercept_ty0)
        form_target.addRow("H0, м:", self.intercept_th0)
        form_target.addRow("V0, м/с:", self.intercept_tv0)
        form_target.addRow("Курс, град:", self.intercept_tpsi0)
        layout_intercept.addWidget(grp_target)

        grp_interceptor = QGroupBox("БПЛА-перехватчик: начальные условия")
        form_interceptor = QFormLayout(grp_interceptor)
        self.intercept_ix0 = self.create_spinbox("X0, м:", 0.0, -100000.0, 100000.0)[1]
        self.intercept_iy0 = self.create_spinbox("Y0, м:", 0.0, -100000.0, 100000.0)[1]
        self.intercept_ih0 = self.create_spinbox("H0, м:", 500.0, 0.0, 5000.0)[1]
        self.intercept_iv0 = self.create_spinbox("V0, м/с:", 40.0, 10.0, 100.0)[1]
        self.intercept_ipsi0 = self.create_spinbox("Курс, град (0 = Восток):", 0.0, -180.0, 360.0)[1]
        form_interceptor.addRow("X0, м:", self.intercept_ix0)
        form_interceptor.addRow("Y0, м:", self.intercept_iy0)
        form_interceptor.addRow("H0, м:", self.intercept_ih0)
        form_interceptor.addRow("V0, м/с:", self.intercept_iv0)
        form_interceptor.addRow("Курс, град:", self.intercept_ipsi0)
        layout_intercept.addWidget(grp_interceptor)

        grp_params = QGroupBox("Параметры перехвата")
        form_params = QFormLayout(grp_params)
        self.intercept_radius = self.create_spinbox("Критерий поражения, м:", 15.0, 0.0, 1000.0)[1]
        form_params.addRow("Критерий поражения, м:", self.intercept_radius)
        layout_intercept.addWidget(grp_params)

        grp_network = QGroupBox("Локальная сеть — UDP/JSON")
        form_network = QFormLayout(grp_network)
        self.intercept_guidance_ip = QLineEdit("127.0.0.1")
        self.intercept_vehicle_ip = QLineEdit("127.0.0.1")
        self.intercept_port_target = self.create_spinbox("Телеметрия цели:", 5001, 1024, 65535)[1]
        self.intercept_port_command = self.create_spinbox("Команды перехватчику:", 5002, 1024, 65535)[1]
        self.intercept_port_feedback = self.create_spinbox("Телеметрия перехватчика:", 5003, 1024, 65535)[1]
        for port_spin in (self.intercept_port_target, self.intercept_port_command, self.intercept_port_feedback):
            port_spin.setDecimals(0)
            port_spin.setSingleStep(1.0)
        form_network.addRow("IP блока наведения:", self.intercept_guidance_ip)
        form_network.addRow("IP перехватчика:", self.intercept_vehicle_ip)
        form_network.addRow("Телеметрия цели:", self.intercept_port_target)
        form_network.addRow("Команды перехватчику:", self.intercept_port_command)
        form_network.addRow("Телеметрия перехватчика:", self.intercept_port_feedback)
        layout_intercept.addWidget(grp_network)

        self.btn_intercept_target = QPushButton("Запустить БПЛА-цель")
        self.btn_intercept_target.setStyleSheet(
            "background-color: #4169E1; color: white; font-weight: bold; padding: 9px;")
        self.btn_intercept_target.clicked.connect(self.start_intercept_target)
        layout_intercept.addWidget(self.btn_intercept_target)

        self.btn_intercept_guidance = QPushButton("Включить блок наведения")
        self.btn_intercept_guidance.setEnabled(False)
        self.btn_intercept_guidance.clicked.connect(self.toggle_intercept_guidance)
        layout_intercept.addWidget(self.btn_intercept_guidance)

        intercept_speed_row = QHBoxLayout()
        intercept_speed_label = QLabel("Скорость симуляции:")
        self.intercept_speed_combo = QComboBox()
        self.intercept_speed_combo.addItems(["1x (Реальное время)", "2x", "5x", "10x"])
        self.intercept_speed_combo.setCurrentIndex(2)
        intercept_speed_row.addWidget(intercept_speed_label)
        intercept_speed_row.addWidget(self.intercept_speed_combo, stretch=1)
        layout_intercept.addLayout(intercept_speed_row)

        self.lbl_intercept_status = QLabel("Симуляция остановлена")
        self.lbl_intercept_status.setWordWrap(True)
        self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: #333;")
        layout_intercept.addWidget(self.lbl_intercept_status)
        layout_intercept.addStretch()
        self.left_tabs.addTab(tab_intercept, "Перехват")

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.fig1 = Figure(figsize=(10, 8))
        self.canvas1 = FigureCanvas(self.fig1)
        self.tabs.addTab(self.canvas1, "Динамика полета")
        self.fig2 = Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        self.tabs.addTab(self.canvas2, "Перегрузки")
        self.fig3 = Figure(figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.fig3)
        self.tabs.addTab(self.canvas3, "Сигналы управления")
        self.fig4 = Figure(figsize=(10, 8))
        self.canvas4 = FigureCanvas(self.fig4)
        self.tabs.addTab(self.canvas4, "Маршрут")
        tab_pfd = QWidget()
        layout_pfd = QVBoxLayout(tab_pfd)
        control_layout = QHBoxLayout()
        self.btn_start_live = QPushButton("Старт")
        self.btn_start_live.setStyleSheet(
            "background-color: #32CD32; color: white; font-weight: bold; font-size: 14px; padding: 15px;")
        self.btn_start_live.clicked.connect(lambda: self.start_live_simulation(switch_tab=True))
        control_layout.addWidget(self.btn_start_live, stretch=3)
        lbl_speed = QLabel("Скорость симуляции:")
        lbl_speed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        control_layout.addWidget(lbl_speed, stretch=1)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x (Реал. время)", "2x", "5x", "10x"])
        control_layout.addWidget(self.speed_combo, stretch=1)
        layout_pfd.addLayout(control_layout)
        self.pfd_widget = PrimaryFlightDisplay()
        layout_pfd.addWidget(self.pfd_widget)
        self.tabs.addTab(tab_pfd, "Авиагоризонт")

        tab_intercept_plot = QWidget()
        intercept_plot_layout = QVBoxLayout(tab_intercept_plot)
        self.fig_intercept = Figure(figsize=(9, 7))
        self.canvas_intercept = FigureCanvas(self.fig_intercept)
        intercept_plot_layout.addWidget(self.canvas_intercept)
        self.lbl_intercept_range = QLabel(
            f"Дистанция до цели: — м | Критерий поражения: {self.intercept_radius.value():.1f} м")
        self.lbl_intercept_range.setWordWrap(True)
        intercept_plot_layout.addWidget(self.lbl_intercept_range)
        self.tabs.addTab(tab_intercept_plot, "Перехват")

    def _close_intercept_subsystems(self):
        if self.intercept_target: self.intercept_target.close()
        if self.intercept_guidance: self.intercept_guidance.close()
        if self.intercept_vehicle: self.intercept_vehicle.close()
        self.intercept_target = None
        self.intercept_vehicle = None
        self.intercept_guidance = None

    def start_intercept_target(self):
        if self.intercept_target is not None:
            if self.intercept_timer.isActive():
                self.intercept_timer.stop()
            self._close_intercept_subsystems()
            self.btn_intercept_target.setText("Запустить БПЛА-цель")
            self.btn_intercept_guidance.setText("Включить блок наведения")
            self.btn_intercept_guidance.setEnabled(False)
            self.lbl_intercept_status.setText("Симуляция остановлена")
            self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: #333;")
            self.lbl_intercept_range.setText(
                f"Дистанция до цели: — м | Критерий поражения: {self.intercept_radius.value():.1f} м")
            return

        g_ip = self.intercept_guidance_ip.text().strip()
        t_port = int(self.intercept_port_target.value())
        self.intercept_target = TargetSubsystem(
            self.intercept_tx0.value(), self.intercept_ty0.value(), self.intercept_th0.value(),
            self.intercept_tv0.value(), self.intercept_tpsi0.value(), g_ip, t_port
        )

        self.intercept_step = 0
        self.intercept_time = 0.0
        self.intercept_history = {key: [] for key in self.intercept_history}

        self.fig_intercept.clear()
        self.ax_intercept = self.fig_intercept.add_subplot(111, projection="3d")
        self.ax_intercept.view_init(elev=30, azim=-45)
        self.ax_intercept.set_xlabel("X, м")
        self.ax_intercept.set_ylabel("Y, м")
        self.ax_intercept.set_zlabel("H, м")
        self.ax_intercept.set_title("Траектория БПЛА-цели")
        self.ax_intercept.grid(True)
        self.canvas_intercept.draw()

        self.btn_intercept_target.setText("Остановить моделирование")
        self.btn_intercept_guidance.setEnabled(True)
        self.btn_intercept_guidance.setText("Включить блок наведения")
        self.lbl_intercept_status.setText("Цель запущена")
        self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: blue;")

        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        self.intercept_timer.start(20)

    def toggle_intercept_guidance(self):
        if self.intercept_target is None:
            return

        if self.intercept_guidance is not None:
            self.intercept_guidance.close()
            self.intercept_vehicle.close()
            self.intercept_guidance = None
            self.intercept_vehicle = None
            self.btn_intercept_guidance.setText("Включить блок наведения")
            self.lbl_intercept_status.setText("Блок наведения выключен")
            self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: blue;")
            return

        g_ip = self.intercept_guidance_ip.text().strip()
        v_ip = self.intercept_vehicle_ip.text().strip()
        t_port = int(self.intercept_port_target.value())
        c_port = int(self.intercept_port_command.value())
        f_port = int(self.intercept_port_feedback.value())

        self.intercept_vehicle = InterceptorSubsystem(
            self.intercept_ix0.value(), self.intercept_iy0.value(), self.intercept_ih0.value(),
            self.intercept_iv0.value(), self.intercept_ipsi0.value(), c_port, g_ip, f_port
        )
        self.intercept_guidance = GuidanceSubsystem(t_port, f_port, v_ip, c_port)

        self.btn_intercept_guidance.setText("Отключить блок наведения")
        self.lbl_intercept_status.setText("Метод Пропорционального Наведения (UDP) АКТИВЕН")
        self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: green;")

    def on_intercept_tick(self):
        if self.intercept_target is None:
            return

        speed_text = self.intercept_speed_combo.currentText()
        if speed_text.startswith("2x"):
            physics_steps = 2
        elif speed_text.startswith("5x"):
            physics_steps = 5
        elif speed_text.startswith("10x"):
            physics_steps = 10
        else:
            physics_steps = 1

        dt = self.intercept_dt
        target_state = None
        vehicle_state = None
        terminal_reached = False

        hit_radius = self.intercept_radius.value()

        for _ in range(physics_steps):
            if self.intercept_target is None: break

            self.intercept_time += dt
            self.intercept_target.step(dt)

            if self.intercept_step % 5 == 0:
                self.intercept_target.publish(self.intercept_time)
                if self.intercept_vehicle is not None:
                    self.intercept_vehicle.publish_telemetry(self.intercept_time)

            if self.intercept_vehicle is not None:
                if self.intercept_guidance is not None:
                    self.intercept_guidance.make_command()
                cmd = self.intercept_vehicle.receive_command()
                self.intercept_vehicle.step(cmd, dt)

            target_state = self.intercept_target.state.copy()
            if self.intercept_vehicle is not None:
                vehicle_state = self.intercept_vehicle.state.copy()

            if self.intercept_step % 5 == 0:
                history = self.intercept_history
                history["target_x"].append(float(target_state[0]))
                history["target_y"].append(float(-target_state[2]))
                history["target_h"].append(float(target_state[1]))

                if vehicle_state is not None:
                    history["vehicle_x"].append(float(vehicle_state[0]))
                    history["vehicle_y"].append(float(-vehicle_state[2]))
                    history["vehicle_h"].append(float(vehicle_state[1]))

                for values in history.values():
                    if len(values) > 10000: del values[:-10000]

            if vehicle_state is not None:
                dx = float(target_state[0] - vehicle_state[0])
                dy = float(-target_state[2] + vehicle_state[2])
                dh = float(target_state[1] - vehicle_state[1])
                distance = float(np.sqrt(dx * dx + dy * dy + dh * dh))

                if distance <= hit_radius:
                    self.intercept_timer.stop()
                    terminal_reached = True

                    self.btn_intercept_target.setText("Очистить и перезапустить")
                    self.btn_intercept_guidance.setText("Включить блок наведения")
                    self.btn_intercept_guidance.setEnabled(False)

                    self.lbl_intercept_status.setText("ЦЕЛЬ ДОСТИГНУТА")
                    self.lbl_intercept_status.setStyleSheet("font-weight: bold; color: red; font-size: 16px;")

                    self.lbl_intercept_range.setText(
                        f"Финальная дистанция: {distance:.1f} м | Критерий поражения: {hit_radius:.1f} м")

                    self.intercept_target.close()
                    if self.intercept_guidance: self.intercept_guidance.close()
                    if self.intercept_vehicle: self.intercept_vehicle.close()
                    break

            self.intercept_step += 1

        history = self.intercept_history
        if target_state is not None or terminal_reached:
            self.ax_intercept.clear()

            all_x = history["target_x"] + (
                history["vehicle_x"] if history["vehicle_x"] else [self.intercept_ix0.value()])
            all_y = history["target_y"] + (
                history["vehicle_y"] if history["vehicle_y"] else [self.intercept_iy0.value()])

            if all_x and all_y:
                mid_x = (max(all_x) + min(all_x)) * 0.5
                mid_y = (max(all_y) + min(all_y)) * 0.5
                max_range = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.5
                max_range = max_range if max_range > 500 else 1000
                self.ax_intercept.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax_intercept.set_ylim(mid_y - max_range, mid_y + max_range)

            self.ax_intercept.set_box_aspect((1, 1, 0.5))

            if history["target_x"]:
                self.ax_intercept.plot(history["target_x"], history["target_y"], history["target_h"],
                                       color="tab:red", label="БПЛА-цель", linewidth=2)
                self.ax_intercept.scatter(history["target_x"][-1], history["target_y"][-1],
                                          history["target_h"][-1], color="tab:red", marker="o", s=50)

            if history["vehicle_x"]:
                self.ax_intercept.plot(history["vehicle_x"], history["vehicle_y"], history["vehicle_h"],
                                       color="tab:blue", label="БПЛА-перехватчик", linewidth=2)

                marker = "*" if terminal_reached else "^"
                size = 200 if terminal_reached else 50
                color = "orange" if terminal_reached else "tab:blue"

                self.ax_intercept.scatter(history["vehicle_x"][-1], history["vehicle_y"][-1],
                                          history["vehicle_h"][-1], color=color, marker=marker, s=size)

            self.ax_intercept.set_xlabel("X, м")
            self.ax_intercept.set_ylabel("Y, м")
            self.ax_intercept.set_zlabel("H, м")
            self.ax_intercept.set_title("Метод пропорционального наведения")

            max_h = max(history["target_h"]) if history["target_h"] else 500.0
            if history["vehicle_h"]: max_h = max(max_h, max(history["vehicle_h"]))
            self.ax_intercept.set_zlim(0.0, max(200.0, max_h + 100.0))

            self.ax_intercept.legend()
            self.ax_intercept.grid(True)
            self.canvas_intercept.draw_idle()

        if vehicle_state is not None and not terminal_reached:
            dx = float(target_state[0] - vehicle_state[0])
            dy = float(-target_state[2] + vehicle_state[2])
            dh = float(target_state[1] - vehicle_state[1])
            distance = float(np.sqrt(dx * dx + dy * dy + dh * dh))
            self.lbl_intercept_range.setText(
                f"Дистанция до цели: {distance:.1f} м | Критерий поражения: {hit_radius:.1f} м")

    def closeEvent(self, event):
        if hasattr(self, "intercept_timer") and self.intercept_timer.isActive():
            self.intercept_timer.stop()
        self._close_intercept_subsystems()
        if hasattr(self, "live_timer") and self.live_timer.isActive():
            self.live_timer.stop()
        event.accept()

    def create_spinbox(self, text, val, min_v, max_v):
        label = QLabel(text)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(val)
        spin.setDecimals(1)
        return label, spin

    def add_wp_row(self, x, y, z):
        row = self.wp_table.rowCount()
        self.wp_table.insertRow(row)
        self.wp_table.setItem(row, 0, QTableWidgetItem(str(x)))
        self.wp_table.setItem(row, 1, QTableWidgetItem(str(y)))
        self.wp_table.setItem(row, 2, QTableWidgetItem(str(z)))

    def del_wp_row(self):
        current_row = self.wp_table.currentRow()
        if current_row >= 0: self.wp_table.removeRow(current_row)

    def open_settings(self):
        dialog = ControllerSettingsDialog(self.autopilot)
        dialog.exec()

    def start_live_simulation(self, switch_tab):
        if self.live_timer.isActive():
            self.live_timer.stop()
            self.btn_start_rte.setText("Запустить полёт по маршруту")
            self.btn_start_rte.setStyleSheet(
                "background-color: #DA70D6; color: white; font-weight: bold; padding: 10px;")
            self.btn_start_live.setText("СТАРТ")
            self.btn_start_live.setStyleSheet(
                "background-color: #32CD32; color: white; font-weight: bold; font-size: 14px; padding: 15px;")
            return

        self.autopilot.reset_controllers()
        self.live_V_zad = self.spin_V_rte[1].value()

        self.live_waypoints = []
        for row in range(self.wp_table.rowCount()):
            try:
                x = float(self.wp_table.item(row, 0).text())
                y = float(self.wp_table.item(row, 1).text())
                z = float(self.wp_table.item(row, 2).text())
                self.live_waypoints.append((x, y, z))
            except:
                continue

        if not self.live_waypoints: return

        self.live_current_wp = 0
        self.live_state = np.zeros(12)
        self.live_state[1] = self.spin_H0_rte[1].value()
        self.live_state[3] = self.spin_V0_rte[1].value()
        self.live_state[5] = np.deg2rad(self.spin_Psi0_rte[1].value())
        self.live_state[7] = 1.0

        self.h_time, self.h_X, self.h_Y, self.h_Z = [], [], [], []
        self.h_V, self.h_Psi, self.h_Gamma = [], [], []
        self.h_Nxa, self.h_Nya, self.h_Nza = [], [], []
        self.h_U_nxa, self.h_U_nya, self.h_U_nza, self.h_U_gamma = [], [], [], []
        self.h_Vzad, self.h_Hzad, self.h_Psizad = [], [], []
        self.tick_counter = 0
        self.ui_tick_counter = 0

        self.fig4.clear()
        self.ax4 = self.fig4.add_subplot(111, projection='3d')
        wp_x = [w[0] for w in self.live_waypoints]
        wp_y = [w[1] for w in self.live_waypoints]
        wp_z = [w[2] for w in self.live_waypoints]
        route_x = [0] + wp_x
        route_y = [0] + wp_y
        route_z = [0] + wp_z

        self.ax4.plot(route_x, route_y, route_z, 'r--', alpha=0.5, label='Маршрут')
        self.ax4.scatter(wp_x, wp_y, wp_z, color='#FFB6C1', s=60, zorder=5)

        max_z = max(wp_z) if wp_z else 100
        z_offset = max_z * 0.08 + 5
        for i, (x, y, z) in enumerate(zip(wp_x, wp_y, wp_z)):
            self.ax4.text(x, y, z + z_offset, f'{i + 1}', color='darkred', weight='bold', fontsize=12)

        self.live_line_3d, = self.ax4.plot([], [], [], '#BA55D3', linewidth=2, label='Траектория')
        self.ax4.set_xlabel('X, м')
        self.ax4.set_ylabel('Y, м')
        self.ax4.set_zlabel('H, м')
        self.ax4.set_xlim(min(0, min(wp_x)) - 100, max(0, max(wp_x)) + 100)
        self.ax4.set_ylim(min(0, min(wp_y)) - 100, max(0, max(wp_y)) + 100)
        self.ax4.set_zlim(0, max_z + 10)
        self.ax4.set_title('Полёт по 3D маршруту')
        self.ax4.view_init(elev=25, azim=-45)
        self.ax4.legend()
        self.canvas4.draw()

        self.btn_start_rte.setText("СТОП")
        self.btn_start_rte.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 10px;")
        self.btn_start_live.setText("СТОП")
        self.btn_start_live.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; font-size: 14px; padding: 15px;")

        if switch_tab:
            self.tabs.setCurrentIndex(4)

        self.live_timer.start(20)

    def on_live_tick(self):
        speed_mode = self.speed_combo.currentText()
        if "2x" in speed_mode:
            loops = 2
        elif "5x" in speed_mode:
            loops = 5
        elif "10x" in speed_mode:
            loops = 10
        else:
            loops = 1

        for _ in range(loops):
            state = self.live_state
            map_X = state[0]
            map_Y = -state[2]

            w_x, w_y, w_z = self.live_waypoints[self.live_current_wp]
            dist = np.hypot(w_x - map_X, w_y - map_Y)

            if dist < 50.0:
                self.live_current_wp += 1
                if self.live_current_wp >= len(self.live_waypoints):
                    self.start_live_simulation(switch_tab=False)
                    return
                w_x, w_y, w_z = self.live_waypoints[self.live_current_wp]

            Psi_zad = np.arctan2(w_y - map_Y, w_x - map_X)
            target_dict = {'V': self.live_V_zad, 'H': w_z, 'Psi': Psi_zad}

            controls = self.autopilot.calculate_controls(state, target_dict, self.uav_model, self.dt)
            self.live_state = self.uav_model.rk4_step(state, controls, self.dt)
            self.live_state[7] = np.clip(self.live_state[7], -8.0, 8.0)

            t = self.tick_counter * self.dt
            self.h_time.append(t)
            self.h_X.append(self.live_state[0])
            self.h_Y.append(self.live_state[1])
            self.h_Z.append(-self.live_state[2])
            self.h_V.append(self.live_state[3])
            self.h_Gamma.append(np.rad2deg(self.live_state[10]))
            self.h_Nxa.append(self.live_state[6])
            self.h_Nya.append(self.live_state[7])
            self.h_Nza.append(self.live_state[9])
            self.h_U_nxa.append(controls[0])
            self.h_U_nya.append(controls[1])
            self.h_U_nza.append(controls[2])
            self.h_U_gamma.append(np.rad2deg(controls[3]))

            self.h_Psi.append(np.rad2deg(self.live_state[5]))
            Psi_err = Psi_zad - self.live_state[5]
            while Psi_err > np.pi: Psi_err -= 2 * np.pi
            while Psi_err < -np.pi: Psi_err += 2 * np.pi
            self.h_Psizad.append(np.rad2deg(self.live_state[5] + Psi_err))
            self.h_Vzad.append(self.live_V_zad)
            self.h_Hzad.append(w_z)
            self.tick_counter += 1

        pitch = np.rad2deg(self.live_state[4])
        roll = np.rad2deg(self.live_state[10])
        hdg = np.rad2deg(self.live_state[5])
        alt = self.live_state[1]
        spd = self.live_state[3]
        vsi = self.live_state[3] * np.sin(self.live_state[4])
        self.pfd_widget.update_data(pitch, roll, hdg, alt, spd, vsi, self.live_state[6], self.live_state[7],
                                    self.live_state[9])

        self.ui_tick_counter += 1
        if self.ui_tick_counter % 10 == 0:
            self.live_line_3d.set_data(np.array(self.h_X), np.array(self.h_Z))
            self.live_line_3d.set_3d_properties(np.array(self.h_Y))
            self.canvas4.draw()

            self.update_plot1(self.h_time, self.h_Y, self.h_V, self.h_Psi, self.h_Gamma, self.h_Hzad, self.h_Vzad,
                              self.h_Psizad)
            self.update_plot2(self.h_time, self.h_Nxa, self.h_Nya, self.h_Nza)
            self.update_plot3(self.h_time, self.h_U_nxa, self.h_U_nya, self.h_U_nza, self.h_U_gamma)

    def run_simulation(self):
        self.autopilot.reset_controllers()
        V_zad, H_zad = self.spin_V_std[1].value(), self.spin_H_std[1].value()
        Psi_zad = np.deg2rad(self.spin_Psi_std[1].value())
        V0, H0 = self.spin_V0_std[1].value(), self.spin_H0_std[1].value()
        Psi0 = np.deg2rad(self.spin_Psi0_std[1].value())
        T_max = self.spin_Time_std[1].value()

        dt = 0.02
        steps = int(T_max / dt)
        state = np.zeros(12)
        state[1], state[3], state[5], state[7] = H0, V0, Psi0, 1.0

        time_hist = np.linspace(0, T_max, steps)
        H_hist, V_hist, Psi_hist, Gamma_hist = (np.zeros(steps) for _ in range(4))
        Nxa_hist, Nya_hist, Nza_hist = (np.zeros(steps) for _ in range(3))
        U_nxa_hist, U_nya_hist, U_nza_hist, U_gamma_hist = (np.zeros(steps) for _ in range(4))

        H_zad_hist = np.full(steps, H_zad)
        V_zad_hist = np.full(steps, V_zad)
        Psi_zad_hist = np.zeros(steps)

        target_dict = {'V': V_zad, 'H': H_zad, 'Psi': Psi_zad}

        for i in range(steps):
            H_hist[i], V_hist[i] = state[1], state[3]
            Gamma_hist[i] = np.rad2deg(state[10])
            Nxa_hist[i], Nya_hist[i], Nza_hist[i] = state[6], state[7], state[9]

            Psi_hist[i] = np.rad2deg(state[5])
            Psi_err = Psi_zad - state[5]
            while Psi_err > np.pi: Psi_err -= 2 * np.pi
            while Psi_err < -np.pi: Psi_err += 2 * np.pi
            Psi_zad_hist[i] = np.rad2deg(state[5] + Psi_err)

            controls = self.autopilot.calculate_controls(state, target_dict, self.uav_model, dt)
            U_nxa_hist[i], U_nya_hist[i], U_nza_hist[i], U_gamma_hist[i] = controls[0], controls[1], controls[
                2], np.rad2deg(controls[3])
            state = self.uav_model.rk4_step(state, controls, dt)
            state[7] = np.clip(state[7], -8.0, 8.0)

        self.update_plot1(time_hist, H_hist, V_hist, Psi_hist, Gamma_hist, H_zad_hist, V_zad_hist, Psi_zad_hist)
        self.update_plot2(time_hist, Nxa_hist, Nya_hist, Nza_hist)
        self.update_plot3(time_hist, U_nxa_hist, U_nya_hist, U_nza_hist, U_gamma_hist)
        self.tabs.setCurrentIndex(0)

    def update_plot1(self, time_hist, H_hist, V_hist, Psi_hist, Gamma_hist, H_zad_hist, V_zad_hist, Psi_zad_hist):
        self.fig1.clear()
        ax1 = self.fig1.add_subplot(2, 2, 1)
        ax2 = self.fig1.add_subplot(2, 2, 2)
        ax3 = self.fig1.add_subplot(2, 2, 3)
        ax4 = self.fig1.add_subplot(2, 2, 4)

        ax1.plot(time_hist, H_hist, 'b', label='H тек')
        ax1.plot(time_hist, H_zad_hist, 'r--', label='H зад')
        ax1.set(title="Высота", xlabel="Время, с", ylabel="H, м")
        ax1.grid()
        ax1.legend()

        ax2.plot(time_hist, V_hist, 'g', label='V тек')
        ax2.plot(time_hist, V_zad_hist, 'r--', label='V зад')
        ax2.set(title="Скорость", xlabel="Время, с", ylabel="V, м/с")
        ax2.grid()
        ax2.legend()

        ax3.plot(time_hist, Psi_hist, 'purple', label='Psi тек')
        ax3.plot(time_hist, Psi_zad_hist, 'r--', label='Psi зад')
        ax3.set(title="Курс", xlabel="Время, с", ylabel="град")
        ax3.grid()
        ax3.legend()

        ax4.plot(time_hist, Gamma_hist, 'orange')
        ax4.plot(time_hist, [20] * len(time_hist), 'r:', alpha=0.5)
        ax4.plot(time_hist, [-20] * len(time_hist), 'r:', alpha=0.5)
        ax4.set(title="Крен", xlabel="Время, с", ylabel="град")
        ax4.grid()
        self.fig1.tight_layout()
        self.canvas1.draw()

    def update_plot2(self, time_hist, Nxa_hist, Nya_hist, Nza_hist):
        self.fig2.clear()
        ax1 = self.fig2.add_subplot(2, 2, 1)
        ax2 = self.fig2.add_subplot(2, 2, 2)
        ax3 = self.fig2.add_subplot(2, 2, 3)

        ax1.plot(time_hist, Nxa_hist, 'k')
        ax1.set(title="Продольная перегрузка ($n_{xa}$)", xlabel="Время, с")
        ax1.grid()
        ax2.plot(time_hist, Nya_hist, 'k')
        ax2.set(title="Нормальная перегрузка ($n_{ya}$)", xlabel="Время, с")
        ax2.grid()
        ax3.plot(time_hist, Nza_hist, 'k')
        ax3.set(title="Боковая перегрузка ($n_{za}$)", xlabel="Время, с")
        ax3.grid()
        self.fig2.tight_layout()
        self.canvas2.draw()

    def update_plot3(self, time_hist, U_nxa_hist, U_nya_hist, U_nza_hist, U_gamma_hist):
        self.fig3.clear()
        ax1 = self.fig3.add_subplot(2, 2, 1)
        ax2 = self.fig3.add_subplot(2, 2, 2)
        ax3 = self.fig3.add_subplot(2, 2, 3)
        ax4 = self.fig3.add_subplot(2, 2, 4)

        ax1.plot(time_hist, U_nxa_hist, 'm')
        ax1.set(title="Упр. скоростью ($u_{n_{xa}}$)", xlabel="Время, с")
        ax1.grid()
        ax2.plot(time_hist, U_nya_hist, 'm')
        ax2.set(title="Упр. высотой ($u_{n_{ya}}$)", xlabel="Время, с")
        ax2.grid()
        ax3.plot(time_hist, U_nza_hist, 'm')
        ax3.set(title="Упр. боковое ($u_{n_{za}}$)", xlabel="Время, с")
        ax3.grid()
        ax4.plot(time_hist, U_gamma_hist, 'm')
        ax4.set(title=r"Упр. креном ($u_{\gamma}$)", xlabel="Время, с")
        ax4.grid()
        self.fig3.tight_layout()
        self.canvas3.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

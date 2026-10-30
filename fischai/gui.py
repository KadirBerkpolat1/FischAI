"""
Settings window: start and stop the macro, watch the fishing cycle, and tune
every setting of config.toml. Changes reach the running macro at once and
are saved without a save button. Colours follow the desktop's matugen
palette when it exists; the hotkey keeps working while the window is open.
"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__, config
from .app import Controller, Log, assets_installed, single_instance
from .config import DATA_DIR, STATE_DIR
from .recovery import ROUTE_FILE, RouteRecorder, load_route, save_route

PALETTE_FILE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ryoku" / "colors.json"
FALLBACK_PALETTE = {
    "surface": "#131313", "surfaceContainerLowest": "#0e0e0e", "surfaceContainerLow": "#1b1b1b",
    "surfaceContainer": "#1f1f1f", "surfaceContainerHigh": "#2a2a2a", "surfaceContainerHighest": "#353535",
    "onSurface": "#e3e3e3", "onSurfaceVariant": "#c6c6c6", "outline": "#909090", "outlineVariant": "#474747",
    "primary": "#c6c6c6", "onPrimary": "#303030",
}
UI_FONTS = ("Space Grotesk", "Inter", "Noto Sans")
LOG_FONTS = ("SpaceMono Nerd Font", "Space Mono", "JetBrains Mono", "Noto Sans Mono")
REFRESH_MS = 200
SAVE_DELAY_MS = 600
LOG_LINES = 300
ROUTE_HOTKEY_GRACE_S = 1.0

# The fishing cycle shown at the top; idle-chain phases count as casting.
STEPS = ("Atış", "Shake", "Çekme")
STEP_OF_PHASE = {"starting": 0, "cast": 0, "idle": 0, "totem": 0, "potions": 0, "aquarium": 0, "shake": 1, "reel": 2}


def load_palette():
    palette = dict(FALLBACK_PALETTE)
    try:
        palette.update({k: v for k, v in json.loads(PALETTE_FILE.read_text()).items() if k in palette})
    except (OSError, ValueError):
        pass
    return palette


def pick_font(candidates):
    families = set(QFontDatabase.families())
    return next((f for f in candidates if f in families), QFont().defaultFamily())


def stylesheet(p, ui_font, log_font):
    return f"""
    QWidget {{ background: {p['surface']}; color: {p['onSurface']}; font-family: "{ui_font}"; font-size: 13px; }}
    QLabel#title {{ font-size: 19px; font-weight: 500; }}
    QLabel#muted, QLabel#value {{ color: {p['outline']}; }}
    QLabel#value {{ min-width: 64px; }}
    QLabel#section {{ color: {p['onSurfaceVariant']}; font-weight: 500; padding-top: 18px; padding-bottom: 2px; }}
    QLabel#hint {{ color: {p['outline']}; font-size: 12px; }}
    QLabel#step {{ font-size: 22px; font-weight: 500; color: {p['outlineVariant']}; padding-bottom: 4px;
                   border-bottom: 2px solid transparent; }}
    QLabel#step[active="true"] {{ color: {p['onSurface']}; border-bottom: 2px solid {p['primary']}; }}
    QLabel#detail {{ color: {p['onSurfaceVariant']}; }}
    QFrame#rule {{ background: {p['outlineVariant']}; max-height: 1px; }}
    QPushButton {{ background: {p['surfaceContainerHigh']}; border: none; border-radius: 6px; padding: 7px 14px; }}
    QPushButton:hover {{ background: {p['surfaceContainerHighest']}; }}
    QPushButton:focus {{ outline: none; border: 1px solid {p['outline']}; }}
    QPushButton#primary {{ background: {p['primary']}; color: {p['onPrimary']}; font-weight: 500; padding: 8px 20px; }}
    QPushButton#primary:hover {{ background: {p['onSurface']}; }}
    QPushButton#primary[running="true"] {{ background: {p['surfaceContainerHighest']}; color: {p['onSurface']}; }}
    QPushButton#primary:disabled {{ background: {p['surfaceContainerHigh']}; color: {p['outline']}; }}
    QPushButton#segment {{ border-radius: 0; padding: 5px 14px; color: {p['onSurfaceVariant']}; }}
    QPushButton#segment:checked {{ background: {p['primary']}; color: {p['onPrimary']}; }}
    QSlider::groove:horizontal {{ height: 4px; background: {p['outlineVariant']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {p['primary']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {p['onSurface']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
    QSlider:focus::handle:horizontal {{ background: {p['primary']}; }}
    QSpinBox, QLineEdit {{ background: {p['surfaceContainerHigh']}; border: 1px solid transparent; border-radius: 4px;
                          padding: 4px 6px; selection-background-color: {p['primary']}; selection-color: {p['onPrimary']}; }}
    QSpinBox:focus, QLineEdit:focus {{ border: 1px solid {p['outline']}; }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
    QScrollArea {{ border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; }}
    QScrollBar::handle:vertical {{ background: {p['surfaceContainerHighest']}; border-radius: 4px; min-height: 24px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QPlainTextEdit#log {{ background: {p['surfaceContainerLowest']}; color: {p['outline']}; border: none; border-radius: 6px;
                          padding: 6px; font-family: "{log_font}"; font-size: 11px; }}
    """


class Switch(QAbstractButton):
    """A small on/off switch: a pill whose knob slides right when on."""

    def __init__(self, palette, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._p = palette

    def sizeHint(self):
        return QSize(38, 22)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(1, 3, 36, 16)
        on = self.isChecked()
        painter.setPen(QColor(self._p["outline"]) if self.hasFocus() else Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._p["primary"] if on else self._p["surfaceContainerHighest"]))
        painter.drawRoundedRect(track, 8, 8)
        knob = QRectF(21 if on else 3, 5, 12, 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._p["onPrimary"] if on else self._p["outline"]))
        painter.drawEllipse(knob)


class Window(QWidget):
    def __init__(self, cfg, log, palette):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.p = palette
        self.controller = None
        self.events = queue.Queue()
        self.log_lines = queue.Queue()
        log.listeners.append(self.log_lines.put)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(SAVE_DELAY_MS)
        self.save_timer.timeout.connect(self._save)
        self.saved_at = 0.0
        self.error_text = None
        self.recorder = None
        self.route_ended_at = -ROUTE_HOTKEY_GRACE_S

        self.setWindowTitle("FischAI")
        self.resize(480, 820)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 16)
        root.setSpacing(10)
        root.addLayout(self._header())
        root.addLayout(self._cycle())
        self.detail = QLabel(objectName="detail", wordWrap=True)
        self.counters = QLabel(objectName="muted")
        root.addWidget(self.detail)
        root.addWidget(self.counters)
        root.addWidget(QFrame(objectName="rule"))
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self._settings())
        root.addWidget(scroll, 1)
        self.logview = QPlainTextEdit(objectName="log", readOnly=True, maximumBlockCount=LOG_LINES)
        self.logview.setFixedHeight(110)
        root.addWidget(self.logview)
        self.footer = QLabel(objectName="hint")
        root.addWidget(self.footer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(REFRESH_MS)

    # -- layout pieces -------------------------------------------------------

    def _header(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("FischAI", objectName="title"))
        row.addStretch(1)
        self.start_button = QPushButton("Başlat", objectName="primary")
        self.start_button.clicked.connect(self._toggle)
        row.addWidget(self.start_button)
        return row

    def _cycle(self):
        row = QHBoxLayout()
        row.setSpacing(18)
        self.step_labels = []
        for name in STEPS:
            label = QLabel(name, objectName="step")
            self.step_labels.append(label)
            row.addWidget(label)
        row.addStretch(1)
        return row

    def _settings(self):
        page = QWidget()
        self.grid = QGridLayout(page)
        self.grid.setContentsMargins(0, 0, 8, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(1, 1)
        self.row = 0
        c = self.cfg

        self._section("Balık tutma")
        self._switch("Oltayı kendisi atsın", c.fishing, "auto_cast")
        self._slider("Atışta basılı tutma", c.fishing, "cast_hold_ms", 100, 2000, 50, "ms")
        self._slider("Yakaladıktan sonra bekleme", c.fishing, "recast_delay_ms", 100, 5000, 100, "ms")
        self._switch("Shake'e kendisi bassın", c.fishing, "auto_shake")

        self._section("Çekme")
        self._switch("Çubuğu kendisi yönetsin", c.reel, "auto_reel")
        self._choice("Olta", c.reel, "rod", (("normal", "Normal"), ("requiem", "Requiem")))
        self._slider("Takip gücü", c.reel, "aggressiveness_pct", 10, 300, 5, "%",
                     hint="Yüksek: çubuk balığa daha sert koşar.")
        self._slider("Yumuşaklık", c.reel, "damping_pct", 10, 300, 5, "%",
                     hint="Yüksek: daha az savrulur, daha geç yetişir.")
        self._slider("Azami hız artışı", c.reel, "balance_pct", 0, 200, 5, "%")
        self._slider("Algılama hızı", c.reel, "fps", 5, 60, 1, "kare/sn")
        self._switch("Her olta için kendini ayarlasın", c.reel, "learn")

        self._section("Totem")
        self._switch("Totem kullan", c.totem, "enabled")
        self._slot("Gündüz totemi slotu", c.totem, "day_slot")
        self._slot("Gece totemi slotu", c.totem, "night_slot")

        self._section("İksir")
        self._switch("İksir kullan", c.potions, "enabled")
        self._potion_slots()
        self._slider("Kullanım aralığı", c.potions, "interval_min", 10, 240, 5, "dk")
        self._switch("Yalnız hava olayı varken", c.potions, "only_during_weather_event")

        self._section("Akvaryum")
        self._switch("Saatte bir besle", c.aquarium, "enabled",
                     hint="Besleme sırasında pencere birkaç saniye 1680×900 olur.")

        self._section("Bağlantı kopunca")
        self._switch("Fisch'e geri gir", c.recovery, "enabled",
                     hint="Sober kapanır ya da Roblox bağlantı hatası verirse Sober'ı yeniden açar, Fisch'e girer, "
                          "doğduğun yerden balık yerine kayıtlı rotayı yürür ve balığa devam eder.")
        self._route_row()

        self._section("Discord")
        self._text("Webhook adresi", c.webhook, "url", "https://discord.com/api/webhooks/…")
        self._slider("Özet aralığı", c.webhook, "summary_min", 5, 240, 5, "dk")

        self._section("Sorun bildirme")
        folder = str(STATE_DIR / "debug").replace(str(Path.home()), "~", 1)
        self._switch("Ne gördüğünü kaydet", c.general, "debug_snapshots",
                     hint=f"Her tıklamada ve aşama değişiminde gördüğü kareyi, tıkladığı yer işaretli olarak "
                          f"{folder} içine kaydeder (en fazla 400 görüntü).")
        self.grid.setRowStretch(self.row, 1)
        return page

    def _place(self, label, widget, hint=None):
        self.grid.addWidget(QLabel(label), self.row, 0)
        self.grid.addWidget(widget, self.row, 1, 1, 2)
        self.row += 1
        if hint:
            self.grid.addWidget(QLabel(hint, objectName="hint", wordWrap=True), self.row, 0, 1, 3)
            self.row += 1

    def _section(self, title):
        self.grid.addWidget(QLabel(title, objectName="section"), self.row, 0, 1, 3)
        self.row += 1

    def _switch(self, label, target, attr, hint=None):
        switch = Switch(self.p)
        switch.setChecked(getattr(target, attr))
        switch.toggled.connect(lambda on: self._set(target, attr, on))
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(switch)
        box = QWidget()
        box.setLayout(holder)
        holder.setContentsMargins(0, 0, 0, 0)
        self._place(label, box, hint)

    def _slider(self, label, target, attr, lo, hi, step, unit, hint=None):
        slider = QSlider(Qt.Orientation.Horizontal, minimum=lo, maximum=hi, singleStep=step, pageStep=step * 4)
        slider.setValue(getattr(target, attr))
        value = QLabel(objectName="value", alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value.setText(f"{slider.value()} {unit}")

        def changed(v):
            v = lo + round((v - lo) / step) * step
            value.setText(f"{v} {unit}")
            self._set(target, attr, v)
        slider.valueChanged.connect(changed)
        self.grid.addWidget(QLabel(label), self.row, 0)
        self.grid.addWidget(slider, self.row, 1)
        self.grid.addWidget(value, self.row, 2)
        self.row += 1
        if hint:
            self.grid.addWidget(QLabel(hint, objectName="hint", wordWrap=True), self.row, 0, 1, 3)
            self.row += 1

    def _slot_box(self, value):
        box = QSpinBox(minimum=0, maximum=9, specialValueText="kapalı", alignment=Qt.AlignmentFlag.AlignCenter)
        box.setValue(value)
        box.setFixedWidth(72)
        return box

    def _slot(self, label, target, attr):
        box = self._slot_box(getattr(target, attr))
        box.valueChanged.connect(lambda v: self._set(target, attr, v))
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addStretch(1)
        holder.addWidget(box)
        wrapper = QWidget()
        wrapper.setLayout(holder)
        self._place(label, wrapper)

    def _potion_slots(self):
        slots = (list(self.cfg.potions.slots) + [0, 0, 0, 0])[:4]
        self.cfg.potions.slots = slots
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addStretch(1)
        for i, value in enumerate(slots):
            box = self._slot_box(value)

            def changed(v, i=i):
                self.cfg.potions.slots[i] = v
                self._changed()
            box.valueChanged.connect(changed)
            holder.addWidget(box)
        wrapper = QWidget()
        wrapper.setLayout(holder)
        self._place("Slotlar", wrapper, hint="Her slottaki iksir sırayla içilir, sonra olta 1. slota geri alınır.")

    def _choice(self, label, target, attr, options):
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.setSpacing(0)
        holder.addStretch(1)
        group = QButtonGroup(self)
        for key, text in options:
            button = QPushButton(text, objectName="segment", checkable=True)
            button.setChecked(getattr(target, attr) == key)
            button.clicked.connect(lambda _c, key=key: self._set(target, attr, key))
            group.addButton(button)
            holder.addWidget(button)
        wrapper = QWidget()
        wrapper.setLayout(holder)
        self._place(label, wrapper)

    def _text(self, label, target, attr, placeholder):
        edit = QLineEdit(getattr(target, attr), placeholderText=placeholder)
        edit.textChanged.connect(lambda t: self._set(target, attr, t.strip()))
        self._place(label, edit)

    def _route_row(self):
        key = self.cfg.general.hotkey.upper()
        self.route_state = QLabel(objectName="value", alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.route_button = QPushButton("Kaydet")
        self.route_button.clicked.connect(self._route_clicked)
        holder = QHBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addStretch(1)
        holder.addWidget(self.route_state)
        holder.addWidget(self.route_button)
        wrapper = QWidget()
        wrapper.setLayout(holder)
        self._place("Yürüme rotası", wrapper,
                    hint=f"Fisch'e yeniden girip doğduğun yerde dur. Kaydet'e, sonra oyunda {key}'e bas; yalnız "
                         f"W A S D, boşluk ve ok tuşlarıyla balık yerine yürü, suya dön ve {key}'e yine bas.")
        self._show_route()

    def _show_route(self):
        route = load_route(STATE_DIR / ROUTE_FILE)
        self.route_state.setText(f"{route[-1][0]:.1f} sn" if route else "kayıtlı değil")

    # -- behaviour -----------------------------------------------------------

    def _set(self, target, attr, value):
        setattr(target, attr, value)   # the running macro reads these live
        self._changed()

    def _changed(self):
        self.save_timer.start()

    def _save(self):
        try:
            config.save(self.cfg)
        except (config.ConfigError, OSError) as e:
            self.footer.setText(f"Kaydedilemedi: {e}")
            return
        self.saved_at = time.monotonic()

    def attach(self, controller):
        self.controller = controller
        self.start_button.setEnabled(True)

    def hotkey(self):
        # The hotkey also starts and ends a route recording; those presses must not toggle the macro.
        if self.recorder is None and time.monotonic() - self.route_ended_at > ROUTE_HOTKEY_GRACE_S:
            self.events.put("toggle")

    def _toggle(self):
        if self.controller is None or self.recorder is not None:
            return
        self.controller.toggle()
        self.error_text = self.controller.last_error

    def _route_clicked(self):
        if self.recorder is not None:
            self.recorder.cancel()
            return
        if self.controller is None or self.controller.session is not None:
            self.footer.setText("Rota, makro dururken kaydedilir.")
            return
        try:
            self.recorder = RouteRecorder(self.cfg.general.hotkey)
        except PermissionError as e:
            self.footer.setText(str(e))
            return
        self.route_button.setText("Vazgeç")
        self.start_button.setEnabled(False)
        self.route_state.setText(f"oyunda {self.cfg.general.hotkey.upper()}'e bas")
        threading.Thread(target=self._route_thread, args=(self.recorder,), daemon=True, name="route").start()

    def _route_thread(self, recorder):
        try:
            events = recorder.record(on_start=lambda: self.events.put(("route", "started")))
        finally:
            recorder.close()
        self.events.put(("route", events))

    def _route_event(self, payload):
        if payload == "started":
            self.route_state.setText("kaydediliyor…")
            return
        self.recorder = None
        self.route_ended_at = time.monotonic()
        self.route_button.setText("Kaydet")
        self.start_button.setEnabled(self.controller is not None)
        if payload:
            save_route(STATE_DIR / ROUTE_FILE, payload)
            self.log(f"route: saved {len(payload)} key events ({payload[-1][0]:.1f} s)")
        self._show_route()

    def _tick(self):
        while not self.events.empty():
            event = self.events.get_nowait()
            if event == "toggle":
                self._toggle()
            else:
                self._route_event(event[1])
        while not self.log_lines.empty():
            self.logview.appendPlainText(self.log_lines.get_nowait())
        if self.controller is not None:
            had_session = self.controller.session is not None
            self.controller.poll()
            if had_session and self.controller.session is None:
                self.error_text = self.controller.last_error
        self._render_status()

    def _render_status(self):
        session = self.controller.session if self.controller else None
        running = session is not None
        self.start_button.setText("Durdur" if running else "Başlat")
        self.start_button.setProperty("running", running)
        self.start_button.style().polish(self.start_button)
        phase = session.phase if running else None
        active = STEP_OF_PHASE.get(phase)
        for i, label in enumerate(self.step_labels):
            on = i == active
            if label.property("active") != on:
                label.setProperty("active", on)
                label.style().polish(label)
        self.detail.setText(self._detail(session))
        if running:
            s = session.stats
            minutes = int((time.time() - s.started) // 60)
            self.counters.setText(f"{s.reels} balık    {s.casts} atış    {minutes} dk    "
                                  f"algılama {session.bridge.frame_ms:.0f} ms")
        else:
            self.counters.setText(f"Kısayol: oyundayken {self.cfg.general.hotkey.upper()}")
        if time.monotonic() - self.saved_at < 2.0:
            self.footer.setText("Ayarlar kaydedildi ve çalışan makroya uygulandı.")
        elif self.footer.text().startswith("Ayarlar kaydedildi"):
            self.footer.setText("")

    def _detail(self, session):
        if self.controller is None:
            return self.error_text or "Hazırlanıyor…"
        if session is None:
            if self.error_text:
                return _turkish_error(self.error_text)
            return "Hazır. Fisch'te suya dönük dur, olta 1. slotta olsun, sonra Başlat'a bas."
        phase = session.phase
        if phase == "paused":
            return "Duraklatıldı: Sober odakta değil. Oyuna dönünce devam eder."
        if phase == "reel":
            boxes = session.bridge.latest
            seen = [f"{name} %{box.conf * 100:.0f}" for name, box in (("çubuk", boxes.bar), ("balık", boxes.fish)) if box]
            return "Çekiyor. Görülen: " + (", ".join(seen) if seen else "kısa bir an hiçbir şey")
        if phase == "shake":
            return f"Shake: bu atışta {session.macro.cast_shake_clicks} tıklama"
        return {
            "starting": "Başlıyor: olta ele alınıyor",
            "cast": "Atış",
            "totem": "Totem kullanılıyor",
            "potions": "İksir içiliyor",
            "aquarium": "Akvaryum besleniyor",
            "recovery": "Bağlantı koptu: Sober yeniden açılıp Fisch'e giriliyor",
            "idle": "Bekliyor",
        }.get(phase, phase)

    def closeEvent(self, event):
        self.timer.stop()
        if self.recorder is not None:
            self.recorder.cancel()
        if self.save_timer.isActive():
            self.save_timer.stop()
            self._save()
        if self.controller is not None:
            self.controller.close()
        super().closeEvent(event)


def _turkish_error(text):
    if text.startswith("No window"):
        return "Sober penceresi bulunamadı. Oyunu açıp tekrar Başlat'a bas."
    if "closed" in text:
        return "Oyun penceresi kapandı."
    if "error prompt" in text:
        return "Roblox bağlantı hatası verdi."
    if text.startswith("Could not get back"):
        return "Sober üç denemede Fisch'e geri giremedi."
    return f"Durdu: {text}"


def _ask_for_module(window, log):
    """No model yet: let the user pick ExoPanda's Fisch module and install it."""
    QMessageBox.information(
        window, "ExoPanda modülü gerekli",
        "Çekme yapay zekası ExoPanda'nın Fisch modülünden alınıyor. "
        "fisch-<sürüm>.zip dosyasını (Windows modülü) seç; model ve şablonlar bu bilgisayara kurulur.")
    path, _ = QFileDialog.getOpenFileName(window, "ExoPanda Fisch modülü", str(Path.home()),
                                          "ExoPanda Fisch modülü (fisch-*.zip *_Fischer.exe)")
    if not path:
        return False
    from . import assets
    try:
        assets.install(path, DATA_DIR, log=log)
    except SystemExit as e:
        QMessageBox.warning(window, "Kurulamadı", str(e))
        return False
    return assets_installed()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FischAI")
    app.setDesktopFileName("fischai")
    palette = load_palette()
    app.setStyleSheet(stylesheet(palette, pick_font(UI_FONTS), pick_font(LOG_FONTS)))
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        QMessageBox.critical(None, "FischAI", "FischAI yalnız Hyprland oturumunda çalışır.")
        return
    try:
        cfg = config.load()
    except config.ConfigError as e:
        QMessageBox.critical(None, "Ayar dosyası hatalı", f"{config.CONFIG_PATH}\n\n{e}")
        return
    try:
        with single_instance():
            _run(app, cfg, palette)
    except SystemExit as e:
        QMessageBox.warning(None, "FischAI", str(e))


def _run(app, cfg, palette):
    from .inputdev import HotkeyListener

    log = Log(STATE_DIR / "fischai.log", echo=False)
    window = Window(cfg, log, palette)
    window.start_button.setEnabled(False)
    window.show()
    app.processEvents()
    if not assets_installed() and not _ask_for_module(window, log):
        window.error_text = "ExoPanda modülü kurulmadan başlatılamaz."
        app.exec()
        return
    try:
        controller = Controller(cfg, log)
    except (OSError, RuntimeError, SystemExit) as e:
        window.error_text = f"Başlatılamadı: {e}"
        app.exec()
        return
    log(f"FischAI {__version__} settings window: reel detector on {controller.model.backend.name}")
    window.attach(controller)
    try:
        HotkeyListener([cfg.general.hotkey], lambda _key: window.hotkey()).start()
    except PermissionError as e:
        log(str(e))
    app.exec()

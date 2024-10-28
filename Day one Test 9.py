import clr
import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox,
                             QHBoxLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QGroupBox, QFormLayout,
                             QTextEdit, QScrollArea, QGridLayout, QAction, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
import pyqtgraph as pg
import time
import numpy as np
from datetime import datetime
import csv
import os
from scipy import signal
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add reference to the DLL
clr.AddReference(r"C:/Program Files/Alpha/bin/I8Library1.dll")
from I8Devices import Device, Settings  # type: ignore


class DataAcquisitionThread(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, device, sampling_rate, channels, extra_channels, gain, exgain, buffer_size=2500, all_data_queue=None):
        super().__init__()
        self.device = device
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.extra_channels = extra_channels
        self.gain = gain if gain != 0 else 1  # Prevent division by zero
        self.exgain = exgain if exgain != 0 else 1
        self.buffer_size = buffer_size
        self.running = True
        self.channel_data_count = {ch: 0 for ch in range(len(channels) + len(extra_channels))}
        self.all_data_queue = all_data_queue

    def run(self):
        total_channels = len(self.channels) + len(self.extra_channels)
        data_buffer = np.empty((total_channels, 0), dtype=np.float32)
        while self.running:
            try:
                last_data = self.device.getData(0)  # Fetch all available data
                if isinstance(last_data, bool):
                    logging.warning("Device is not in gathering mode or receive buffer is empty.")
                    time.sleep(0.1)
                    continue
                if last_data:
                    new_data = []
                    for sample in last_data:
                        sample_data = []
                        sample_length = len(sample)
                        # Collect main channels data
                        for idx, channel_idx in enumerate(self.channels):
                            sample_idx = 6 + idx
                            if sample_length > sample_idx:
                                sample_value = float(sample[sample_idx]) / self.gain
                                sample_data.append(sample_value)
                            else:
                                sample_data.append(0.0)
                        # Collect extra channels data
                        for idx, ex_channel_idx in enumerate(self.extra_channels):
                            sample_idx = 6 + len(self.channels) + idx
                            if sample_length > sample_idx:
                                sample_value = float(sample[sample_idx]) / self.exgain
                                sample_data.append(sample_value)
                            else:
                                sample_data.append(0.0)
                        new_data.append(sample_data)
                    if new_data:
                        new_data_np = np.array(new_data, dtype=np.float32).T  # Shape: (channels, samples)
                        data_buffer = np.hstack((data_buffer, new_data_np))
                        if data_buffer.shape[1] > self.buffer_size:
                            data_buffer = data_buffer[:, -self.buffer_size:]  # Keep only the latest data
                        self.data_received.emit(data_buffer.copy())

                        # Add new data to the all_data_queue
                        if self.all_data_queue is not None:
                            self.all_data_queue.put(new_data_np)
                else:
                    logging.info("No data received in this cycle.")
            except Exception as e:
                logging.error(f"Error in data acquisition: {str(e)}")
                time.sleep(0.1)
            self.msleep(10)  # 10 ms sleep to prevent CPU overutilization

    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish

class RealTimeEEGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Real-Time EEG Plotter")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize timer label first
        self.timer_label = QLabel("Duration: 00:00:00")
        self.timer_label.setFont(QFont("Courier", 12))
        self.timer_label.setAlignment(Qt.AlignLeft)

        # Initialize timer for updating the duration label
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.update_duration)

        # Initialize device
        self.device = Device()
        self.device.debug_mode = False  # Set to False for production

        # Define channels
        self.channels = list(range(21))
        self.extra_channels = list(range(3))

        # Initialize settings
        self.settings = Settings()
        self.init_settings()

        # Initialize data storage
        self.buffer_size = self.settings.sampling_rate * 10
        self.data = np.zeros((len(self.channels) + len(self.extra_channels), self.buffer_size), dtype=np.float32)
        self.original_data = self.data.copy()  # Store original data
        self.ptr = 0

        # Plot settings
        self.scale = 10
        self.duration = 10

        # Initialize plot
        self.init_plot()

        # Initialize data reception tracking
        self.channel_data_count = {ch: 0 for ch in range(len(self.channels) + len(self.extra_channels))}
        self.last_count_update = time.time()

        # Add filter settings
        self.low_pass_enabled = True
        self.high_pass_enabled = True
        self.notch_enabled = True
        self.low_pass_freq = 50  # Hz
        self.high_pass_freq = 1  # Hz
        self.notch_freq = 50  # Hz
        
        # Initialize filter coefficients
        self.notch_filter = None
        self.highpass_filter = None
        self.lowpass_filter = None
        
        # Design initial filters
        self.design_filters()

        # Add scaling factor
        self.scaling_factor = 100  # microvolts

        # Initialize selected channels
        self.selected_channels = self.channels.copy() + self.extra_channels.copy()

        # Setup UI
        self.init_ui()

        # Initialize thread
        self.thread = None

        # Initialize timer for smooth updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_interval = 50  # Update every 50 ms (20 fps)

        # Initialize time tracking for fixed-speed plotting
        self.last_update_time = time.time()
        self.plot_speed = 1.0  # 1 second of data per second of real time

        self.all_data = []  # To store all data since acquisition start
        self.all_data_queue = queue.Queue()  # Move queue to main class

        # Initialize acquisition duration tracking
        self.acquisition_start_time = None
        self.acquisition_duration = 0  # in seconds

    def init_settings(self):
        self.settings.test_signal = False
        self.settings.sampling_rate = 250
        self.settings.leadoff_mode = False
        self.settings.gain = 1  # Changed from 24 to 1
        self.settings.exgain = 1  # Changed from 24 to 1
        # Initialize channels
        self.settings.channels_on = [True] * 21
        self.settings.exchannels_on = [True if i in self.extra_channels else False for i in range(3)]

    def init_ui(self):
        self.apply_style()  # Apply modern style

        # Create menu bar
        self.create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Add timer label to the top of the left layout
        left_layout.addWidget(self.timer_label)

        # Device control
        device_group = QGroupBox("Device Control")
        device_layout = QVBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_device)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.disconnect_button.setEnabled(False)
        device_layout.addWidget(self.connect_button)
        device_layout.addWidget(self.disconnect_button)
        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        self.test_signal_cb = QCheckBox()
        self.sampling_rate_combo = QComboBox()
        self.sampling_rate_combo.addItems(['250', '500', '1000'])
        self.leadoff_mode_cb = QCheckBox()
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(1, 24)
        self.gain_spin.setValue(1)  # Changed from 24 to 1
        self.exgain_spin = QSpinBox()
        self.exgain_spin.setRange(1, 24)
        self.exgain_spin.setValue(1)  # Changed from 24 to 1
        settings_layout.addRow("Test Signal:", self.test_signal_cb)
        settings_layout.addRow("Sampling Rate:", self.sampling_rate_combo)
        settings_layout.addRow("Lead-off Mode:", self.leadoff_mode_cb)
        settings_layout.addRow("Gain:", self.gain_spin)
        settings_layout.addRow("Extra Gain:", self.exgain_spin)
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)

        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QGridLayout()
        self.channel_checkboxes = []
        total_channels = self.channels + self.extra_channels
        for idx, ch in enumerate(total_channels):
            checkbox = QCheckBox(f"Ch {ch+1}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_selected_channels)
            self.channel_checkboxes.append(checkbox)
            row = idx // 4
            col = idx % 4
            channel_layout.addWidget(checkbox, row, col)
        channel_group.setLayout(channel_layout)
        left_layout.addWidget(channel_group)

        # Acquisition control
        acquisition_group = QGroupBox("Acquisition Control")
        acquisition_layout = QVBoxLayout()
        self.start_button = QPushButton("Start Acquisition")
        self.start_button.clicked.connect(self.start_acquisition)
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("Stop Acquisition")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)
        acquisition_layout.addWidget(self.start_button)
        acquisition_layout.addWidget(self.stop_button)
        acquisition_group.setLayout(acquisition_layout)
        left_layout.addWidget(acquisition_group)

        # Data saving
        save_group = QGroupBox("Data Saving")
        save_layout = QVBoxLayout()
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        self.save_all_checkbox = QCheckBox("Save All Data")
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_all_checkbox)
        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)

        # Add filter options
        filter_group = QGroupBox("Filter Options")
        filter_layout = QFormLayout()

        self.low_pass_cb = QCheckBox("Low-pass Filter")
        self.low_pass_cb.setChecked(True)  # Set checked by default
        self.low_pass_cb.stateChanged.connect(self.update_filter_settings)
        self.low_pass_spin = QSpinBox()
        self.low_pass_spin.setRange(1, 1000)
        self.low_pass_spin.setValue(self.low_pass_freq)
        self.low_pass_spin.valueChanged.connect(self.update_filter_settings)
        filter_layout.addRow(self.low_pass_cb, self.low_pass_spin)

        self.high_pass_cb = QCheckBox("High-pass Filter")
        self.high_pass_cb.setChecked(True)  # Set checked by default
        self.high_pass_cb.stateChanged.connect(self.update_filter_settings)
        self.high_pass_spin = QSpinBox()
        self.high_pass_spin.setRange(0, 1000)
        self.high_pass_spin.setValue(self.high_pass_freq)
        self.high_pass_spin.valueChanged.connect(self.update_filter_settings)
        filter_layout.addRow(self.high_pass_cb, self.high_pass_spin)

        self.notch_cb = QCheckBox("Notch Filter")
        self.notch_cb.setChecked(True)  # Set checked by default
        self.notch_cb.stateChanged.connect(self.update_filter_settings)
        self.notch_spin = QSpinBox()
        self.notch_spin.setRange(0, 1000)
        self.notch_spin.setValue(self.notch_freq)
        self.notch_spin.valueChanged.connect(self.update_filter_settings)
        filter_layout.addRow(self.notch_cb, self.notch_spin)

        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)

        # Add scaling factor control
        scaling_group = QGroupBox("Scaling")
        scaling_layout = QHBoxLayout()
        self.scaling_factor_spin = QSpinBox()
        self.scaling_factor_spin.setRange(1, 10000000)
        self.scaling_factor_spin.setValue(100)  # Changed from 1000000 to 100
        self.scaling_factor_spin.valueChanged.connect(self.update_scaling_factor)
        scaling_layout.addWidget(QLabel("Scaling Factor:"))
        scaling_layout.addWidget(self.scaling_factor_spin)
        scaling_group.setLayout(scaling_layout)
        left_layout.addWidget(scaling_group)

        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.plot_widget, 4)
        central_widget.setLayout(main_layout)

    def apply_style(self):
        # Set Fusion style for a modern look
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

    def create_menu_bar(self):
        # Create menu bar
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        save_action = QAction('Save Data', self)
        save_action.triggered.connect(self.save_data)
        file_menu.addAction(save_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        QMessageBox.about(self, "About", "Advanced Real-Time EEG Plotter\nVersion 1.0")

    def init_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Real-Time EEG Data", color='k', size="20pt")

        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Channel', color='k', size='16pt')
        self.plot_widget.setLabel('bottom', 'Time (s)', color='k', size='16pt')

        num_channels = len(self.channels) + len(self.extra_channels)

        # Increase the y-range to provide more separation between channels
        self.plot_widget.setYRange(-num_channels * 2 + 1, 1, padding=0)  # Adjusted separation
        self.plot_widget.getViewBox().setMouseEnabled(y=False)
        self.plot_widget.getViewBox().setMouseEnabled(x=True)
        self.plot_widget.getViewBox().setMouseMode(self.plot_widget.getViewBox().RectMode)

        self.plot_widget.setXRange(0, self.duration)

        self.curves = []
        colors = [QColor().fromHsv(int(h), 200, 200).name() for h in np.linspace(0, 360, num_channels, endpoint=False)]

        channel_names = [f'Ch {ch+1}' for ch in self.channels] + [f'Ex {ch+1}' for ch in self.extra_channels]

        for idx, name in enumerate(channel_names):
            curve = self.plot_widget.plot(pen=pg.mkPen(color=colors[idx], width=1), name=name)
            self.curves.append(curve)

        # Update channel labels position
        axis = self.plot_widget.getAxis('left')
        ticks = [(-idx * 2, name) for idx, name in enumerate(channel_names)]  # Adjusted separation
        axis.setTicks([ticks])

        self.time_axis = np.linspace(0, self.duration, self.buffer_size)

    def update_selected_channels(self):
        # Update the list of selected channels based on checkboxes
        self.selected_channels = []
        for idx, checkbox in enumerate(self.channel_checkboxes):
            if checkbox.isChecked():
                self.selected_channels.append(idx)
        # Update the plot to show only selected channels
        self.update_channel_visibility()

    def update_channel_visibility(self):
        for idx, curve in enumerate(self.curves):
            if idx in self.selected_channels:
                curve.show()
            else:
                curve.hide()

    def connect_device(self):
        try:
            if self.device.connect():
                logging.info('Connected successfully')
                self.connect_button.setEnabled(False)
                self.disconnect_button.setEnabled(True)
                self.start_button.setEnabled(True)
                QMessageBox.information(self, "Success", "Device connected successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to connect to device.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect to device: {str(e)}")

    def disconnect_device(self):
        try:
            self.stop_acquisition()
            self.device.disconnect()
            logging.info('Disconnected from device')
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.start_button.setEnabled(False)
            QMessageBox.information(self, "Success", "Device disconnected successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to disconnect device: {str(e)}")

    def apply_settings(self):
        try:
            self.settings.test_signal = self.test_signal_cb.isChecked()
            self.settings.sampling_rate = int(self.sampling_rate_combo.currentText())
            self.settings.leadoff_mode = self.leadoff_mode_cb.isChecked()
            self.settings.gain = self.gain_spin.value()
            self.settings.exgain = self.exgain_spin.value()
            self.data = self.original_data.copy()  # Reset to original data
            return self.device.set(self.settings)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
            return False

    def start_acquisition(self):
        if self.apply_settings():
            logging.info('Settings applied successfully')
            try:
                if self.device.start():
                    logging.info('Data acquisition started')
                    self.start_time = time.time()
                    self.last_update_time = self.start_time
                    self.acquisition_start_time = self.start_time  # Track start time
                    self.acquisition_duration = 0  # Reset duration

                    # Start duration timer
                    self.duration_timer.start(1000)  # Update every second

                    status = self.device.getStatus()

                    if status['mode']:
                        self.thread = DataAcquisitionThread(
                            device=self.device,
                            sampling_rate=self.settings.sampling_rate,
                            channels=self.channels,
                            extra_channels=self.extra_channels,
                            gain=self.settings.gain,
                            exgain=self.settings.exgain,
                            buffer_size=self.buffer_size,
                            all_data_queue=self.all_data_queue
                        )
                        self.thread.data_received.connect(self.receive_data)
                        self.thread.start()
                        self.start_button.setEnabled(False)
                        self.stop_button.setEnabled(True)
                        self.save_button.setEnabled(True)
                        self.update_timer.start(self.update_interval)
                        self.all_data = []  # Reset all_data when starting new acquisition

                        # Clear the queue when starting new acquisition
                        while not self.all_data_queue.empty():
                            self.all_data_queue.get()

                        self.acquisition_start_time = time.time()
                    else:
                        QMessageBox.critical(self, "Error", "Device did not enter gathering mode. Please check the configuration.")
                else:
                    QMessageBox.critical(self, "Error", "Failed to start data acquisition.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start data acquisition: {str(e)}")
        else:
            QMessageBox.critical(self, "Error", "Failed to apply settings.")

    def stop_acquisition(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.update_timer.stop()
        self.duration_timer.stop()  # Stop the duration timer
        try:
            self.device.stop()
            logging.info('Data acquisition stopped')
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop data acquisition: {str(e)}")

    @pyqtSlot(np.ndarray)
    def receive_data(self, data_chunk):
        num_samples = data_chunk.shape[1]

        # Update the circular buffer (last 10 seconds of data)
        if self.ptr + num_samples >= self.buffer_size:
            space_left = self.buffer_size - self.ptr
            self.data[:, self.ptr:] = data_chunk[:, :space_left]
            overflow = num_samples - space_left
            if overflow > 0:
                self.data[:, :overflow] = data_chunk[:, -overflow:]
            self.ptr = overflow if overflow > 0 else 0
        else:
            self.data[:, self.ptr:self.ptr + num_samples] = data_chunk
            self.ptr += num_samples

        # Update data reception count
        for i in range(data_chunk.shape[0]):
            if i in self.channel_data_count:
                self.channel_data_count[i] += data_chunk.shape[1]
            else:
                logging.warning(f"Warning: Unexpected channel index {i}")

        # Store all data for saving option
        self.all_data.append(data_chunk)

    def design_filters(self):
        """Design all filters based on current settings"""
        try:
            nyquist = 0.5 * self.settings.sampling_rate
            
            if self.notch_freq > 0:
                notch_low = self.notch_freq - 5  # 5 Hz bandwidth
                notch_high = self.notch_freq + 5
                self.notch_filter = signal.butter(5, np.array([notch_low, notch_high])/nyquist, btype='bandstop')
            
            if self.high_pass_freq > 0:
                self.highpass_filter = signal.butter(5, self.high_pass_freq/nyquist, btype='highpass')
            
            if self.low_pass_freq > 0:
                self.lowpass_filter = signal.butter(5, self.low_pass_freq/nyquist, btype='lowpass')
        except Exception as e:
            logging.error(f"Error designing filters: {str(e)}")

    def update_filter_settings(self):
        """Update filter settings and redesign filters if needed"""
        needs_redesign = False
        
        if self.low_pass_enabled != self.low_pass_cb.isChecked() or \
           self.low_pass_freq != self.low_pass_spin.value():
            needs_redesign = True
        
        if self.high_pass_enabled != self.high_pass_cb.isChecked() or \
           self.high_pass_freq != self.high_pass_spin.value():
            needs_redesign = True
            
        if self.notch_enabled != self.notch_cb.isChecked() or \
           self.notch_freq != self.notch_spin.value():
            needs_redesign = True

        # Update settings
        self.low_pass_enabled = self.low_pass_cb.isChecked()
        self.high_pass_enabled = self.high_pass_cb.isChecked()
        self.notch_enabled = self.notch_cb.isChecked()
        self.low_pass_freq = self.low_pass_spin.value()
        self.high_pass_freq = self.high_pass_spin.value()
        self.notch_freq = self.notch_spin.value()

        # Redesign filters if needed
        if needs_redesign:
            self.design_filters()

    def apply_filters(self, data):
        """Apply the pre-designed filters to the data"""
        try:
            # Apply detrend first
            data = signal.detrend(data)

            if self.notch_enabled and self.notch_filter is not None:
                data = signal.filtfilt(self.notch_filter[0], self.notch_filter[1], data)

            if self.high_pass_enabled and self.highpass_filter is not None:
                data = signal.filtfilt(self.highpass_filter[0], self.highpass_filter[1], data)

            if self.low_pass_enabled and self.lowpass_filter is not None:
                data = signal.filtfilt(self.lowpass_filter[0], self.lowpass_filter[1], data)

            return data
        except Exception as e:
            logging.error(f"Error applying filters: {str(e)}")
            return data  # Return unfiltered data in case of error

    def update_plot(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update time axis
        self.time_axis += elapsed_time
        visible_duration = self.duration  # Duration visible on screen
        start_time = max(0, self.time_axis[-1] - visible_duration)
        end_time = self.time_axis[-1]

        num_channels = len(self.channels) + len(self.extra_channels)

        for idx, curve in enumerate(self.curves):
            if idx in self.selected_channels:
                y_data = self.data[idx, :]
                y_data = self.apply_filters(y_data)  # Apply filters
                # Improve vertical scaling and separation
                y_data = y_data * self.scale / self.scaling_factor - (idx * 2)  # Use self.scaling_factor
                curve.setData(self.time_axis, y_data)
            else:
                curve.clear()

        # Set x-range for scrolling effect
        self.plot_widget.setXRange(start_time, end_time, padding=0)

        # Update channel labels
        self.update_channel_labels()

    def update_channel_labels(self):
        axis = self.plot_widget.getAxis('left')
        num_channels = len(self.channels) + len(self.extra_channels)
        channel_names = [f'Ch {ch+1}' for ch in self.channels] + [f'Ex {ch+1}' for ch in self.extra_channels]

        ticks = []
        for idx, name in enumerate(channel_names):
            if idx in self.selected_channels:
                y_pos = -idx * 2  # Consistent with the plotting offset
                ticks.append((y_pos, name))

        axis.setTicks([ticks])

    def save_data(self):
        try:
            filename = self.generate_filename()
            
            if self.save_all_checkbox.isChecked():
                # Collect all data from the queue
                all_data_chunks = []
                while not self.all_data_queue.empty():
                    all_data_chunks.append(self.all_data_queue.get())
                data_to_save = np.hstack(all_data_chunks) if all_data_chunks else np.array([])
            else:
                data_to_save = self.data

            timestamps = [(i / self.settings.sampling_rate) * 1000 for i in range(data_to_save.shape[1])]
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                header = ["Time (milliseconds)"] + [f"Channel {ch + 1} Data (μV)" for ch in self.channels]
                header += [f"Extra Channel extra_{ch + 1} Data (μV)" for ch in self.extra_channels]
                writer.writerow(header)
                
                for i in range(data_to_save.shape[1]):
                    row = [timestamps[i]] + list(data_to_save[:, i])
                    writer.writerow(row)
            logging.info(f"Data saved to {filename}")
            QMessageBox.information(self, "Success", f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def generate_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Calculate duration if acquisition_start_time exists
        duration_str = ""
        if self.acquisition_start_time is not None:
            duration_seconds = int(time.time() - self.acquisition_start_time)
            duration_str = f"test_{timestamp}_{duration_seconds}sec.csv"
        else:
            duration_str = f"test_{timestamp}.csv"
        
        return os.path.join(os.getcwd(), duration_str)

    def closeEvent(self, event):
        try:
            self.stop_acquisition()
            self.device.stop()
            event.accept()
        except Exception as e:
            logging.error(f"Error during application closure: {str(e)}")
            event.accept()

    def update_scaling_factor(self, value):
        self.scaling_factor = value

    def update_duration(self):
        if self.acquisition_start_time is not None:
            duration = int(time.time() - self.acquisition_start_time)
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            self.timer_label.setText(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = RealTimeEEGPlotter()
    window.show()
    sys.exit(app.exec_())

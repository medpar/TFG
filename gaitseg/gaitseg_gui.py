import sys
import os
import glob
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                             QRadioButton, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d # Added for resampling

try:
    import gaitseg_utils as gal # Using your script name
except ImportError:
    print("Error: Could not import 'gaitseg_utils.py'. Make sure it's in the correct path and named correctly.")
    sys.exit(1)

GUI_BASE_DATA_DIR = gal.root_dir
GUI_OUTPUT_DIR = os.path.expanduser("~/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected")
SUBJECT_DIRS_PATTERN = "S*"
TRIAL_FILE_PATTERN = "S*_A01_T*.raw"
MAGNET_TOLERANCE_SECONDS = 0.1 # Increased tolerance for stronger magnet

os.makedirs(GUI_OUTPUT_DIR, exist_ok=True)

class GaitCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Phase Correction Tool")
        self.setGeometry(100, 100, 1400, 800)

        self.raw_file_paths = []
        self.current_file_index = -1
        self.current_file_path = None

        self.current_t_w = None
        self.current_omega_signal = None
        self.current_mid_swing = []
        self.current_hs = []
        self.current_to = []
        self.current_flat_mask = None
        self.editable_phase_labels = None
        self.data_is_dirty = False

        self.defining_new_region_mode = False
        self.new_region_start_idx = None
        self.temp_vline_start = None
        self.temp_vline_end = None
        self.selected_segment_patch = None
        self.selected_segment_indices = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.on_file_selected_from_list_wrapper)
        left_panel.addWidget(QLabel("Files to Process:"))
        left_panel.addWidget(self.file_list_widget)

        nav_buttons_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous File")
        self.prev_button.clicked.connect(self.prev_file_auto_save)
        self.next_button = QPushButton("Next File")
        self.next_button.clicked.connect(self.next_file_auto_save)
        nav_buttons_layout.addWidget(self.prev_button)
        nav_buttons_layout.addWidget(self.next_button)
        left_panel.addLayout(nav_buttons_layout)
        
        self.status_label = QLabel("Status: Load files to begin.")
        left_panel.addWidget(self.status_label)

        main_layout.addLayout(left_panel, 1)

        right_panel = QVBoxLayout()
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        right_panel.addWidget(self.canvas)

        controls_layout = QHBoxLayout()
        mode_groupbox = QGroupBox("Interaction Mode")
        mode_layout = QVBoxLayout()
        self.modify_mode_radio = QRadioButton("Modify Existing Phase Segment")
        self.define_mode_radio = QRadioButton("Define New Phase Region (2 Clicks)")
        self.modify_mode_radio.setChecked(True)
        self.modify_mode_radio.toggled.connect(self.on_mode_change)
        mode_layout.addWidget(self.modify_mode_radio)
        mode_layout.addWidget(self.define_mode_radio)
        mode_groupbox.setLayout(mode_layout)
        controls_layout.addWidget(mode_groupbox)

        phase_groupbox = QGroupBox("Assign Phase")
        phase_layout = QHBoxLayout()
        self.phase_buttons = {}
        self.phase_names = {0: "Stance", 1: "Swing", 2: "Turn", -1: "Unclassified"}
        for phase_code, phase_name in self.phase_names.items():
            btn = QPushButton(f"{phase_name} ({phase_code})")
            btn.clicked.connect(lambda checked, pc=phase_code: self.assign_phase_to_selection(pc))
            self.phase_buttons[phase_code] = btn
            phase_layout.addWidget(btn)
        phase_groupbox.setLayout(phase_layout)
        controls_layout.addWidget(phase_groupbox)
        
        right_panel.addLayout(controls_layout)

        self.save_button = QPushButton("Save Current Corrections")
        self.save_button.clicked.connect(self.save_current_file_corrections)
        right_panel.addWidget(self.save_button)

        main_layout.addLayout(right_panel, 3)

        self.populate_file_list()
        if self.raw_file_paths:
            self.load_file_data(0)
        else:
            self.status_label.setText("No .raw files found in specified directory structure.")
            
    def populate_file_list(self):
        self.raw_file_paths = []
        subject_dirs = sorted(glob.glob(os.path.join(GUI_BASE_DATA_DIR, SUBJECT_DIRS_PATTERN)))
        for subj_dir in subject_dirs:
            if os.path.isdir(subj_dir):
                trial_files = sorted(glob.glob(os.path.join(subj_dir, TRIAL_FILE_PATTERN)))
                self.raw_file_paths.extend(trial_files)
        
        self.file_list_widget.clear()
        for fp_raw in self.raw_file_paths:
            self.file_list_widget.addItem(os.path.basename(fp_raw))
        print(f"Found {len(self.raw_file_paths)} .raw files.")

    def on_file_selected_from_list_wrapper(self, current_item, previous_item):
        if current_item:
            if self.data_is_dirty and self.current_file_path:
                print(f"Auto-saving changes for {os.path.basename(self.current_file_path)} before switching.")
                self.save_corrected_csv()
                self.data_is_dirty = False
            
            idx = self.file_list_widget.row(current_item)
            if idx != self.current_file_index:
                self.load_file_data(idx)

    def load_file_data(self, file_index):
        if not (0 <= file_index < len(self.raw_file_paths)):
            self.status_label.setText("Invalid file index.")
            return

        self.current_file_index = file_index
        self.current_file_path = self.raw_file_paths[file_index]
        
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.setCurrentRow(file_index)
        self.file_list_widget.blockSignals(False)

        self.status_label.setText(f"Processing: {os.path.basename(self.current_file_path)}")
        QApplication.processEvents()

        try:
            results = gal.compute_velocity_and_events(self.current_file_path)
            if results[0] is None:
                raise ValueError("compute_velocity_and_events returned None.")

            self.current_t_w, _, self.current_omega_signal, \
            self.current_mid_swing, _, self.current_hs, self.current_to, \
            self.current_flat_mask = results
            
            if self.current_t_w is None or self.current_omega_signal is None:
                 raise ValueError("Time or omega signal is None after processing from compute_velocity_and_events.")

            subject_id_from_path = os.path.basename(os.path.dirname(self.current_file_path))
            base_filename_no_ext_raw = os.path.splitext(os.path.basename(self.current_file_path))[0]
            corrected_csv_filename = f"{base_filename_no_ext_raw}_corrected.csv"
            corrected_csv_path = os.path.join(GUI_OUTPUT_DIR, subject_id_from_path, corrected_csv_filename)

            if os.path.exists(corrected_csv_path):
                print(f"Loading previously corrected phases from: {corrected_csv_path}")
                df_corrected = pd.read_csv(corrected_csv_path)
                # Corrected CSV has 'time' and 'phase' at 50Hz.
                # self.current_t_w is at the native IMU processing rate from compute_velocity_and_events.
                # We need to "upsample" the phases from the CSV to match self.current_t_w for display.
                if 'time' in df_corrected.columns and 'phase' in df_corrected.columns and \
                   len(self.current_t_w) > 0 and len(df_corrected['time']) > 0:
                    
                    corrected_time_50hz = df_corrected['time'].values
                    corrected_phases_50hz = df_corrected['phase'].values.astype(int)

                    # Ensure sorted for interpolation
                    sort_indices_corrected = np.argsort(corrected_time_50hz)
                    sorted_corrected_time_50hz = corrected_time_50hz[sort_indices_corrected]
                    sorted_corrected_phases_50hz = corrected_phases_50hz[sort_indices_corrected]

                    phase_upsampler = interp1d(
                        sorted_corrected_time_50hz,
                        sorted_corrected_phases_50hz,
                        kind='nearest',
                        bounds_error=False,
                        fill_value=(sorted_corrected_phases_50hz[0], sorted_corrected_phases_50hz[-1]) # Extrapolate with nearest
                    )
                    self.editable_phase_labels = phase_upsampler(self.current_t_w).astype(int)
                    self.status_label.setText(f"Loaded (resampled prior corrections): {os.path.basename(self.current_file_path)}")
                else:
                    print(f"Warning: Corrected CSV {corrected_csv_path} missing time/phase, or current_t_w is empty. Recomputing initial phases.")
                    self.editable_phase_labels = gal.compute_phase_labels_from_events(
                        len(self.current_t_w), self.current_hs, self.current_to, self.current_flat_mask
                    )
                    self.status_label.setText(f"Loaded (initial phases): {os.path.basename(self.current_file_path)}")
            else:
                self.editable_phase_labels = gal.compute_phase_labels_from_events(
                    len(self.current_t_w), self.current_hs, self.current_to, self.current_flat_mask
                )
                self.status_label.setText(f"Loaded (initial phases): {os.path.basename(self.current_file_path)}")
            
            self.data_is_dirty = False

        except Exception as e:
            self.status_label.setText(f"Error processing {os.path.basename(self.current_file_path)}: {e}")
            print(f"Error processing {self.current_file_path}: {e}")
            self.current_t_w = None
            self.editable_phase_labels = None
            self.data_is_dirty = False
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error loading {os.path.basename(self.current_file_path)}", ha='center', va='center')
            self.canvas.draw()
            return

        self.defining_new_region_mode = self.define_mode_radio.isChecked()
        self.new_region_start_idx = None
        self.selected_segment_indices = None
        if self.selected_segment_patch:
            self.selected_segment_patch.remove()
            self.selected_segment_patch = None
        self.plot_current_data()

    def on_mode_change(self):
        self.defining_new_region_mode = self.define_mode_radio.isChecked()
        self.new_region_start_idx = None
        if self.temp_vline_start: self.temp_vline_start.remove(); self.temp_vline_start = None
        if self.temp_vline_end: self.temp_vline_end.remove(); self.temp_vline_end = None
        if self.selected_segment_patch: self.selected_segment_patch.remove(); self.selected_segment_patch = None
        self.selected_segment_indices = None
        if self.canvas: self.canvas.draw_idle()
        self.status_label.setText(f"Mode: {'Define New Region' if self.defining_new_region_mode else 'Modify Existing Segment'}")

    def plot_current_data(self):
        self.ax.clear()
        if self.current_t_w is None or self.current_omega_signal is None or self.editable_phase_labels is None:
            self.ax.text(0.5, 0.5, "No data to display or error in loading.", ha='center', va='center')
            self.canvas.draw()
            return

        sensor_id = gal.sensors[0] if gal.LEG.upper() == 'L' else gal.sensors[1]
        signal_color = gal.colors.get(sensor_id, 'tab:grey')
        
        plot_t = self.current_t_w
        plot_omega = self.current_omega_signal
        plot_phases = self.editable_phase_labels
        
        min_len_data = len(plot_t)
        if len(plot_omega) != min_len_data:
            print(f"Warning: Omega signal length ({len(plot_omega)}) differs from time vector length ({min_len_data}). Truncating/padding omega.")
            plot_omega_aligned = np.full(min_len_data, np.nan)
            common_len = min(len(plot_omega), min_len_data)
            plot_omega_aligned[:common_len] = plot_omega[:common_len]
            plot_omega = plot_omega_aligned
        if len(plot_phases) != min_len_data:
            print(f"Warning: Phase labels length ({len(plot_phases)}) differs from time vector length ({min_len_data}). Truncating/padding phases.")
            plot_phases_aligned = np.full(min_len_data, -1, dtype=int) # Default to unclassified
            common_len = min(len(plot_phases), min_len_data)
            plot_phases_aligned[:common_len] = plot_phases[:common_len]
            plot_phases = plot_phases_aligned


        self.ax.plot(plot_t, plot_omega, color=signal_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id})')

        phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
        phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
        legend_phases_added = set()

        for ph_val, color_val in phase_colors_map.items():
            mask = (plot_phases == ph_val)
            if not np.any(mask): continue
            diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]

            for seg_start, seg_end in zip(starts, ends):
                if seg_start < seg_end and seg_start < len(plot_t):
                    actual_seg_end_idx = min(seg_end, len(plot_t))
                    if actual_seg_end_idx <= seg_start: continue
                    
                    label_to_use = None
                    if ph_val not in legend_phases_added:
                        label_to_use = phase_legend_labels_map.get(ph_val)
                        legend_phases_added.add(ph_val)
                    
                    end_time_for_span = plot_t[actual_seg_end_idx-1] if actual_seg_end_idx > seg_start else plot_t[seg_start]
                    self.ax.axvspan(plot_t[seg_start], end_time_for_span + gal.dt/2, # gal.dt is fs=50hz, this might need adjustment if plot_t is higher rate
                                    color=color_val, alpha=0.4, label=label_to_use)
        
        event_marker_size = 6
        if self.current_mid_swing:
            valid_ms = [idx for idx in self.current_mid_swing if 0 <= idx < len(plot_t) and 0 <= idx < len(plot_omega)]
            if valid_ms: self.ax.plot(plot_t[valid_ms], plot_omega[valid_ms], '.', color='red', markersize=event_marker_size, alpha=0.7, label='Mid-Swing')
        if self.current_hs:
            valid_hs = [idx for idx in self.current_hs if 0 <= idx < len(plot_t) and 0 <= idx < len(plot_omega)]
            if valid_hs: self.ax.plot(plot_t[valid_hs], plot_omega[valid_hs], 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike')
        if self.current_to:
            valid_to = [idx for idx in self.current_to if 0 <= idx < len(plot_t) and 0 <= idx < len(plot_omega)]
            if valid_to: self.ax.plot(plot_t[valid_to], plot_omega[valid_to], 's', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off')

        self.ax.set_title(f'Interactive Correction - {os.path.basename(self.current_file_path)}')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Angular Velocity (rad/s)')
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='best')
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def _snap_to_event(self, click_time, click_idx_initial):
        if self.current_t_w is None: return click_idx_initial

        all_event_indices = sorted(list(set(self.current_hs + self.current_to)))
        snapped_idx = click_idx_initial
        min_time_diff_to_snap = MAGNET_TOLERANCE_SECONDS

        closest_event_idx = -1
        min_abs_diff = float('inf')

        for event_idx in all_event_indices:
            if 0 <= event_idx < len(self.current_t_w):
                time_diff = self.current_t_w[event_idx] - click_time
                abs_time_diff = abs(time_diff)
                if abs_time_diff < min_time_diff_to_snap:
                    if abs_time_diff < min_abs_diff :
                        min_abs_diff = abs_time_diff
                        closest_event_idx = event_idx
        
        if closest_event_idx != -1:
            print(f"Snap: Click at {click_time:.2f}s (idx {click_idx_initial}) snapped to event at {self.current_t_w[closest_event_idx]:.2f}s (idx {closest_event_idx})")
            return closest_event_idx
        return click_idx_initial


    def on_plot_click(self, event):
        if event.inaxes != self.ax or self.current_t_w is None:
            return
        
        click_time = event.xdata
        click_idx_raw = np.argmin(np.abs(self.current_t_w - click_time))

        if self.selected_segment_patch:
            self.selected_segment_patch.remove()
            self.selected_segment_patch = None

        if self.defining_new_region_mode:
            snapped_click_idx = self._snap_to_event(click_time, click_idx_raw)

            if self.new_region_start_idx is None:
                self.new_region_start_idx = snapped_click_idx
                if self.temp_vline_start: self.temp_vline_start.remove()
                self.temp_vline_start = self.ax.axvline(self.current_t_w[self.new_region_start_idx], color='lime', linestyle='--')
                self.status_label.setText(f"New region start: {self.current_t_w[self.new_region_start_idx]:.2f}s (idx {self.new_region_start_idx}). Click to set end.")
                if self.temp_vline_end: self.temp_vline_end.remove(); self.temp_vline_end = None
            else:
                new_region_end_idx_snapped = snapped_click_idx
                if self.temp_vline_end: self.temp_vline_end.remove()
                self.temp_vline_end = self.ax.axvline(self.current_t_w[new_region_end_idx_snapped], color='orange', linestyle='--')
                
                start_final = min(self.new_region_start_idx, new_region_end_idx_snapped)
                end_final = max(self.new_region_start_idx, new_region_end_idx_snapped)
                
                if start_final == end_final:
                    if start_final < len(self.current_t_w) -1 :
                        end_final = start_final
                    elif start_final > 0:
                        start_final = end_final -1
                    else:
                        end_final = start_final
                self.selected_segment_indices = (start_final, end_final + 1 )
                self.status_label.setText(f"New region defined: [{self.current_t_w[start_final]:.2f}s - {self.current_t_w[end_final]:.2f}s]. Select a phase.")
        else:
            boundary_events = sorted(list(set([0] + self.current_hs + self.current_to + [len(self.current_t_w)-1])))
            unique_boundaries = []
            if boundary_events:
                unique_boundaries.append(boundary_events[0])
                for i in range(1, len(boundary_events)):
                    if boundary_events[i] > boundary_events[i-1]:
                         unique_boundaries.append(boundary_events[i])
            
            seg_start_idx, seg_end_idx = -1, -1
            for i in range(len(unique_boundaries) -1):
                s_idx, e_idx = unique_boundaries[i], unique_boundaries[i+1]
                if s_idx <= click_idx_raw < e_idx:
                    seg_start_idx, seg_end_idx = s_idx, e_idx
                    break
            if seg_start_idx == -1 and unique_boundaries and click_idx_raw >= unique_boundaries[-1] and click_idx_raw < len(self.current_t_w):
                 seg_start_idx = unique_boundaries[-1]
                 seg_end_idx = len(self.current_t_w)

            if seg_start_idx != -1 and seg_end_idx != -1 and seg_start_idx < seg_end_idx:
                self.selected_segment_indices = (seg_start_idx, seg_end_idx)
                self.status_label.setText(f"Selected segment: [{self.current_t_w[seg_start_idx]:.2f}s - {self.current_t_w[seg_end_idx-1]:.2f}s]. Assign new phase.")
                
                # Estimate segment width based on current_t_w time, not gal.dt
                time_start = self.current_t_w[seg_start_idx]
                time_end = self.current_t_w[seg_end_idx-1] 
                segment_width_time = time_end - time_start
                if seg_end_idx < len(self.current_t_w): # Add average dt if not the last sample
                    segment_width_time += (self.current_t_w[seg_end_idx] - self.current_t_w[seg_end_idx-1]) / 2
                else: # Add average dt from previous sample if at the very end
                     segment_width_time += (self.current_t_w[seg_end_idx-1] - self.current_t_w[seg_end_idx-2]) / 2 if seg_end_idx > 1 else gal.dt


                self.selected_segment_patch = patches.Rectangle(
                    (time_start - ( (self.current_t_w[seg_start_idx+1] - self.current_t_w[seg_start_idx])/2 if seg_start_idx+1 < len(self.current_t_w) else gal.dt/2 ) , self.ax.get_ylim()[0]), # adjust start slightly
                    segment_width_time,
                    self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                    linewidth=2, edgecolor='gold', facecolor='yellow', alpha=0.2, zorder=0
                )
                self.ax.add_patch(self.selected_segment_patch)
            else:
                self.selected_segment_indices = None
                self.status_label.setText("Clicked outside defined event segments or invalid segment.")
        
        self.canvas.draw_idle()

    def assign_phase_to_selection(self, phase_code):
        if self.editable_phase_labels is None:
            self.status_label.setText("No data loaded to assign phase.")
            return

        if self.selected_segment_indices:
            start_idx, end_idx_exclusive = self.selected_segment_indices
            if 0 <= start_idx < end_idx_exclusive <= len(self.editable_phase_labels):
                self.editable_phase_labels[start_idx:end_idx_exclusive] = phase_code
                self.data_is_dirty = True
                self.status_label.setText(f"Assigned phase {self.phase_names[phase_code]} to segment "
                                          f"[{self.current_t_w[start_idx]:.2f}s - {self.current_t_w[end_idx_exclusive-1]:.2f}s].")
                self.plot_current_data()
                
                self.selected_segment_indices = None
                if self.selected_segment_patch: self.selected_segment_patch.remove(); self.selected_segment_patch = None
                if self.defining_new_region_mode:
                    self.new_region_start_idx = None
                    if self.temp_vline_start: self.temp_vline_start.remove(); self.temp_vline_start = None
                    if self.temp_vline_end: self.temp_vline_end.remove(); self.temp_vline_end = None
                    self.canvas.draw_idle()
            else:
                self.status_label.setText(f"Error: Invalid segment indices for assignment ({start_idx}, {end_idx_exclusive}). Max index: {len(self.editable_phase_labels)-1}")
        else:
            self.status_label.setText("No segment selected. Click on plot first.")
    
    def save_current_file_corrections(self):
        if self.current_file_path and self.editable_phase_labels is not None:
            self.save_corrected_csv()
        else:
            self.status_label.setText("No current file or data to save.")


    def save_corrected_csv(self):
        if self.current_file_path is None or self.editable_phase_labels is None or self.current_t_w is None:
            self.status_label.setText("No data to save (file, labels, or time missing).")
            print("Save cancelled: file path, labels, or time missing.")
            return

        if len(self.current_t_w) == 0:
            self.status_label.setText("No time data available for phases (time vector is empty). Cannot save.")
            print("Error in save_corrected_csv: self.current_t_w is empty.")
            return
        
        original_phase_timestamps = self.current_t_w
        
        # Align editable_phase_labels with original_phase_timestamps if lengths differ (should not happen if load_file_data is correct)
        if len(self.editable_phase_labels) != len(original_phase_timestamps):
            print(f"Warning: Mismatch between editable_phase_labels ({len(self.editable_phase_labels)}) "
                  f"and current_t_w ({len(original_phase_timestamps)}). Aligning by truncating/padding.")
            min_len = min(len(self.editable_phase_labels), len(original_phase_timestamps))
            current_phase_labels_aligned = np.full(min_len, -1, dtype=int) # default unclassified
            current_phase_labels_aligned[:min(min_len, len(self.editable_phase_labels))] = self.editable_phase_labels[:min(min_len, len(self.editable_phase_labels))]
            original_phase_timestamps = original_phase_timestamps[:min_len]
            if min_len == 0:
                self.status_label.setText("No phase data after alignment. Cannot save.")
                print("Error in save_corrected_csv: No data after aligning phases and timestamps.")
                return
        else:
            current_phase_labels_aligned = self.editable_phase_labels

        base_filename_raw = os.path.basename(self.current_file_path)
        base_filename_no_ext = os.path.splitext(base_filename_raw)[0]
        subject_id_from_path = os.path.basename(os.path.dirname(self.current_file_path))
        
        mot_root_dir = os.path.expanduser("~/Documents/TFG_VIDIMU/VIDIMU/benchmark/jointangles/jointangles_imus")
        mot_path = os.path.join(mot_root_dir, subject_id_from_path, f"ik_{base_filename_no_ext}.mot")

        output_df_data = {}
        target_time_vector = None
        resampled_phases = None

        if os.path.exists(mot_path):
            try:
                dfmot = pd.read_csv(mot_path, skiprows=6, sep='\t')
                if 'time' not in dfmot.columns or len(dfmot['time']) == 0:
                    print(f"Warning: MOT file {mot_path} is missing 'time' column or time data is empty. Falling back to synthetic time.")
                    # Fallback to synthetic time logic (will be handled by the 'else' of 'if os.path.exists(mot_path)' effectively if we set mot_path to non-existent)
                    # For clarity, handle this explicitly or set a flag to use synthetic time
                    raise FileNotFoundError("MOT time column missing or empty") # Trigger fallback
                
                target_time_vector = dfmot['time'].values
                print(f"Using time vector from MOT file: {len(target_time_vector)} points.")

                # Resample phase labels to target_time_vector (50Hz from MOT)
                if len(original_phase_timestamps) == 1:
                    resampled_phases = np.full_like(target_time_vector, current_phase_labels_aligned[0], dtype=int)
                else:
                    # Ensure original_phase_timestamps are sorted for interp1d
                    sort_indices = np.argsort(original_phase_timestamps)
                    sorted_original_timestamps = original_phase_timestamps[sort_indices]
                    sorted_original_labels = current_phase_labels_aligned[sort_indices]

                    phase_interpolator = interp1d(
                        sorted_original_timestamps,
                        sorted_original_labels,
                        kind='nearest',
                        bounds_error=False,
                        fill_value=(sorted_original_labels[0], sorted_original_labels[-1])
                    )
                    resampled_phases = phase_interpolator(target_time_vector).astype(int)
                
                output_df_data = {'time': target_time_vector, 'phase': resampled_phases}
                
                # Add joint angles
                for joint_name_template in gal.JOINTS:
                    actual_col_name_in_dfmot = None
                    for col_dfmot in dfmot.columns:
                        if col_dfmot.lower() == joint_name_template.lower(): # Ignoring case for joint name
                            actual_col_name_in_dfmot = col_dfmot
                            break
                    
                    if actual_col_name_in_dfmot:
                        joint_angle_data = gal.fp.getJointAngleMotAsNP(dfmot, actual_col_name_in_dfmot)
                        if len(joint_angle_data) != len(target_time_vector):
                            print(f"Warning: Joint {actual_col_name_in_dfmot} length ({len(joint_angle_data)}) "
                                  f"differs from MOT time ({len(target_time_vector)}). Aligning by padding/truncating.")
                            if len(joint_angle_data) > len(target_time_vector):
                                output_df_data[joint_name_template] = joint_angle_data[:len(target_time_vector)]
                            else:
                                padded_angles = np.pad(joint_angle_data, (0, len(target_time_vector) - len(joint_angle_data)), 'edge')
                                output_df_data[joint_name_template] = padded_angles
                        else:
                            output_df_data[joint_name_template] = joint_angle_data
                    else:
                        print(f"Warning: Joint {joint_name_template} not found in {mot_path}. Filling with NaNs.")
                        output_df_data[joint_name_template] = np.full(len(target_time_vector), np.nan)

            except Exception as e:
                print(f"Error processing .mot file {mot_path}: {e}. Will attempt to save phases with synthetic 50Hz time.")
                # Force fallback to synthetic time generation if MOT processing fails
                target_time_vector = None # Signal that MOT time is unavailable
        
        if target_time_vector is None: # Fallback if MOT not found, or error reading MOT's time
            print(f"Warning: MOT file not found at {mot_path} or MOT time invalid. Resampling phases to a synthetic 50Hz grid.")
            
            t_start = original_phase_timestamps[0]
            t_end = original_phase_timestamps[-1]
            if t_end < t_start: t_end = t_start # Ensure non-negative duration

            # +1 to include both start and end if t_end-t_start is a multiple of 1/fs
            num_target_samples = int(np.round((t_end - t_start) * gal.fs)) + 1 
            if num_target_samples <= 0: num_target_samples = 1
            
            # Create time vector ensuring exact sample rate
            target_time_vector = t_start + np.arange(num_target_samples) / gal.fs
            print(f"Generated synthetic 50Hz time vector: {len(target_time_vector)} points from {t_start:.2f}s to {target_time_vector[-1]:.2f}s.")


            if len(original_phase_timestamps) == 1:
                resampled_phases = np.full_like(target_time_vector, current_phase_labels_aligned[0], dtype=int)
            else:
                sort_indices = np.argsort(original_phase_timestamps)
                sorted_original_timestamps = original_phase_timestamps[sort_indices]
                sorted_original_labels = current_phase_labels_aligned[sort_indices]

                phase_interpolator = interp1d(
                    sorted_original_timestamps,
                    sorted_original_labels,
                    kind='nearest',
                    bounds_error=False,
                    fill_value=(sorted_original_labels[0], sorted_original_labels[-1])
                )
                resampled_phases = phase_interpolator(target_time_vector).astype(int)
            
            output_df_data = {'time': target_time_vector, 'phase': resampled_phases}
            # No joint angles to add if MOT file was not processed successfully

        output_df = pd.DataFrame(output_df_data)
        
        # Final check for consistent column lengths (should be guaranteed by now if logic is correct)
        expected_len = len(output_df['time'])
        for col, data_col in output_df.items():
            if len(data_col) != expected_len:
                print(f"Critical Error: Column '{col}' has length {len(data_col)}, expected {expected_len}. Dataframe inconsistent before saving.")
                # Pad/truncate as a last resort, though this indicates a flaw in earlier logic
                if len(data_col) > expected_len:
                    output_df[col] = data_col[:expected_len]
                else:
                    # Create a padded series/array
                    padding = np.full(expected_len - len(data_col), np.nan if data_col.dtype.kind == 'f' else -1) # Use -1 for phases
                    output_df[col] = pd.Series(np.concatenate([data_col.values, padding]))


        output_subject_dir = os.path.join(GUI_OUTPUT_DIR, subject_id_from_path)
        os.makedirs(output_subject_dir, exist_ok=True)
        output_filename_csv = f"{base_filename_no_ext}_corrected.csv"
        output_filepath_csv = os.path.join(output_subject_dir, output_filename_csv)
        
        try:
            output_df.to_csv(output_filepath_csv, index=False, float_format='%.5f') # Save with precision
            self.status_label.setText(f"Saved: {output_filename_csv}")
            print(f"Saved corrected data to {output_filepath_csv} with {len(output_df)} rows.")
            self.data_is_dirty = False
        except Exception as e:
            self.status_label.setText(f"Error saving CSV: {e}")
            print(f"Error saving CSV {output_filepath_csv}: {e}")

    def next_file_auto_save(self):
        if self.data_is_dirty and self.current_file_path:
            self.save_corrected_csv()
        if self.current_file_index < len(self.raw_file_paths) - 1:
            self.load_file_data(self.current_file_index + 1)
        else:
            self.status_label.setText("Last file reached.")

    def prev_file_auto_save(self):
        if self.data_is_dirty and self.current_file_path:
            self.save_corrected_csv()
        if self.current_file_index > 0:
            self.load_file_data(self.current_file_index - 1)
        else:
            self.status_label.setText("First file reached.")

    def closeEvent(self, event):
        if self.data_is_dirty:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Save before quitting?",
                                         QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                         QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                self.save_corrected_csv()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
                return
        else:
            event.accept()
        print("Closing Gait Correction Tool.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = GaitCorrectionApp()
    mainWin.show()
    sys.exit(app.exec())
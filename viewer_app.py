#!/usr/bin/env python3
"""
SHAPY Body Viewer - Cross-Platform 3D Visualization Application

A modern desktop application for viewing 3D body meshes and anthropometric measurements.
Works on Windows and Linux.

Requirements:
    pip install PyQt5 pyvista pyvistaqt trimesh numpy

Usage:
    python viewer_app.py
    python viewer_app.py --mesh output/example.obj
    python viewer_app.py --mesh output/example.obj --measurements output/example_measurements.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFrame, QGroupBox, QGridLayout,
    QSplitter, QStatusBar, QMenuBar, QAction, QMessageBox, QStyle,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# PyVista for 3D rendering
import pyvista as pv
from pyvistaqt import QtInteractor

# Trimesh for loading OBJ files
import trimesh


# Dark theme colors
DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
DARK_CARD = "#0f3460"
ACCENT_BLUE = "#00d4ff"
ACCENT_GREEN = "#00ff88"
ACCENT_ORANGE = "#ff9f1c"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#a0a0a0"


class MeasurementsPanel(QGroupBox):
    """Panel to display anthropometric measurements."""
    
    def __init__(self, parent=None):
        super().__init__("📊 Anthropometric Measurements", parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QGridLayout()
        layout.setSpacing(12)
        
        # Measurement labels
        self.labels = {}
        measurements = [
            ("height", "📏 Height", "cm"),
            ("mass", "⚖️ Weight", "kg"),
            ("chest", "📐 Chest", "cm"),
            ("waist", "📐 Waist", "cm"),
            ("hips", "📐 Hips", "cm"),
        ]
        
        for i, (key, name, unit) in enumerate(measurements):
            # Name label
            name_label = QLabel(name)
            name_label.setFont(QFont("Segoe UI", 11))
            name_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
            layout.addWidget(name_label, i, 0)
            
            # Value label
            value_label = QLabel("--")
            value_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setStyleSheet(f"color: {ACCENT_BLUE};")
            layout.addWidget(value_label, i, 1)
            
            # Unit label
            unit_label = QLabel(unit)
            unit_label.setFont(QFont("Segoe UI", 10))
            unit_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
            layout.addWidget(unit_label, i, 2)
            
            self.labels[key] = value_label
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {DARK_CARD};")
        layout.addWidget(separator, len(measurements), 0, 1, 3)
        
        # BMI calculation (if height and weight available)
        bmi_name = QLabel("📈 BMI")
        bmi_name.setFont(QFont("Segoe UI", 11))
        bmi_name.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(bmi_name, len(measurements) + 1, 0)
        
        self.bmi_label = QLabel("--")
        self.bmi_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.bmi_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bmi_label.setStyleSheet(f"color: {ACCENT_GREEN};")
        layout.addWidget(self.bmi_label, len(measurements) + 1, 1)
        
        bmi_unit = QLabel("kg/m²")
        bmi_unit.setFont(QFont("Segoe UI", 10))
        bmi_unit.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(bmi_unit, len(measurements) + 1, 2)
        
        self.setLayout(layout)
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: bold;
                color: {TEXT_PRIMARY};
                border: 2px solid {DARK_CARD};
                border-radius: 10px;
                margin-top: 12px;
                padding: 15px;
                background-color: {DARK_PANEL};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {ACCENT_BLUE};
            }}
        """)
    
    def update_measurements(self, measurements: dict):
        """Update the displayed measurements."""
        height_m = None
        weight_kg = None
        
        for key, label in self.labels.items():
            if key in measurements:
                m = measurements[key]
                if key == "mass":
                    value = m.get("value", 0)
                    label.setText(f"{value:.1f}")
                    weight_kg = value
                else:
                    # Convert to cm if in meters
                    value = m.get("value_cm", m.get("value", 0) * 100)
                    label.setText(f"{value:.1f}")
                    if key == "height":
                        height_m = value / 100
            else:
                label.setText("--")
        
        # Calculate BMI
        if height_m and weight_kg and height_m > 0:
            bmi = weight_kg / (height_m ** 2)
            self.bmi_label.setText(f"{bmi:.1f}")
            
            # Color code BMI
            if bmi < 18.5:
                self.bmi_label.setStyleSheet(f"color: {ACCENT_BLUE};")  # Underweight
            elif bmi < 25:
                self.bmi_label.setStyleSheet(f"color: {ACCENT_GREEN};")  # Normal
            elif bmi < 30:
                self.bmi_label.setStyleSheet(f"color: {ACCENT_ORANGE};")  # Overweight
            else:
                self.bmi_label.setStyleSheet("color: #ff4444;")  # Obese
        else:
            self.bmi_label.setText("--")
    
    def clear(self):
        """Clear all measurements."""
        for label in self.labels.values():
            label.setText("--")
        self.bmi_label.setText("--")


class ControlsPanel(QGroupBox):
    """Panel with view controls."""
    
    def __init__(self, parent=None):
        super().__init__("🎮 View Controls", parent)
        self.plotter = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # View buttons
        views_layout = QGridLayout()
        
        button_style = f"""
            QPushButton {{
                padding: 10px;
                border: 1px solid {DARK_CARD};
                border-radius: 6px;
                background-color: {DARK_CARD};
                color: {TEXT_PRIMARY};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_BLUE};
                color: {DARK_BG};
            }}
            QPushButton:pressed {{
                background-color: #0099cc;
            }}
        """
        
        self.btn_front = QPushButton("▶ Front")
        self.btn_front.setStyleSheet(button_style)
        self.btn_front.clicked.connect(lambda: self.set_view('front'))
        views_layout.addWidget(self.btn_front, 0, 0)
        
        self.btn_back = QPushButton("◀ Back")
        self.btn_back.setStyleSheet(button_style)
        self.btn_back.clicked.connect(lambda: self.set_view('back'))
        views_layout.addWidget(self.btn_back, 0, 1)
        
        self.btn_left = QPushButton("⬅ Left")
        self.btn_left.setStyleSheet(button_style)
        self.btn_left.clicked.connect(lambda: self.set_view('left'))
        views_layout.addWidget(self.btn_left, 1, 0)
        
        self.btn_right = QPushButton("➡ Right")
        self.btn_right.setStyleSheet(button_style)
        self.btn_right.clicked.connect(lambda: self.set_view('right'))
        views_layout.addWidget(self.btn_right, 1, 1)
        
        self.btn_top = QPushButton("⬆ Top")
        self.btn_top.setStyleSheet(button_style)
        self.btn_top.clicked.connect(lambda: self.set_view('top'))
        views_layout.addWidget(self.btn_top, 2, 0)
        
        self.btn_reset = QPushButton("🔄 Reset")
        self.btn_reset.setStyleSheet(button_style)
        self.btn_reset.clicked.connect(lambda: self.set_view('iso'))
        views_layout.addWidget(self.btn_reset, 2, 1)
        
        layout.addLayout(views_layout)
        
        # Screenshot button
        self.btn_screenshot = QPushButton("📷 Save Screenshot")
        self.btn_screenshot.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_BLUE}, stop:1 #0099ff);
                color: {DARK_BG};
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00e5ff, stop:1 #00aaff);
            }}
        """)
        layout.addWidget(self.btn_screenshot)
        
        self.setLayout(layout)
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: bold;
                color: {TEXT_PRIMARY};
                border: 2px solid {DARK_CARD};
                border-radius: 10px;
                margin-top: 12px;
                padding: 15px;
                background-color: {DARK_PANEL};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {ACCENT_BLUE};
            }}
        """)
    
    def set_plotter(self, plotter):
        """Set the plotter reference."""
        self.plotter = plotter
        self.btn_screenshot.clicked.connect(self.save_screenshot)
    
    def set_view(self, view: str):
        """Set camera view with proper orientation for body meshes."""
        if not self.plotter:
            return
        
        try:
            # Get mesh bounds for camera positioning
            bounds = self.plotter.bounds
            if bounds is None:
                return
                
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
            
            # Calculate distance based on mesh size
            size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            distance = size * 2.5
            
            # Camera positions: (position, focal_point, view_up)
            # SMPL-X mesh has Y pointing up, Z pointing forward
            views = {
                'front': (
                    [center[0], center[1], center[2] + distance],
                    center,
                    [0, 1, 0]
                ),
                'back': (
                    [center[0], center[1], center[2] - distance],
                    center,
                    [0, 1, 0]
                ),
                'left': (
                    [center[0] - distance, center[1], center[2]],
                    center,
                    [0, 1, 0]
                ),
                'right': (
                    [center[0] + distance, center[1], center[2]],
                    center,
                    [0, 1, 0]
                ),
                'top': (
                    [center[0], center[1] + distance, center[2]],
                    center,
                    [0, 0, -1]
                ),
                'iso': 'iso',
            }
            
            if view in views:
                if view == 'iso':
                    self.plotter.camera_position = 'iso'
                    self.plotter.reset_camera()
                else:
                    self.plotter.camera_position = views[view]
                    
        except Exception as e:
            print(f"Error setting view: {e}")
            # Fallback to simple reset
            self.plotter.reset_camera()
    
    def save_screenshot(self):
        """Save a screenshot of the current view."""
        if not self.plotter:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if filename:
            self.plotter.screenshot(filename)


class SHAPYViewer(QMainWindow):
    """Main application window."""
    
    def __init__(self, mesh_path=None, measurements_path=None):
        super().__init__()
        self.mesh_path = mesh_path
        self.measurements_path = measurements_path
        self.current_mesh = None
        
        self.setup_ui()
        self.setup_menu()
        
        # Load initial files if provided
        if mesh_path:
            self.load_mesh(mesh_path)
        if measurements_path:
            self.load_measurements(measurements_path)
        elif mesh_path:
            # Try to auto-detect measurements file
            self.auto_load_measurements(mesh_path)
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("SHAPY Body Viewer")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {DARK_CARD};
                width: 2px;
            }}
        """)
        
        # Left panel (3D viewer)
        viewer_container = QWidget()
        viewer_container.setStyleSheet(f"background-color: {DARK_BG};")
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista plotter with dark theme
        self.plotter = QtInteractor(viewer_container)
        self.plotter.set_background('#0d1117')  # Dark background
        self.plotter.add_axes(color='white', line_width=2)
        viewer_layout.addWidget(self.plotter.interactor)
        
        # Add welcome text
        self.plotter.add_text(
            "Open a mesh file to begin",
            position='upper_left',
            font_size=14,
            color='#666666'
        )
        
        splitter.addWidget(viewer_container)
        
        # Right panel (controls and measurements)
        right_panel = QWidget()
        right_panel.setMaximumWidth(340)
        right_panel.setMinimumWidth(300)
        right_panel.setStyleSheet(f"background-color: {DARK_BG};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with gradient effect
        title = QLabel("🏃 SHAPY Body Viewer")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {ACCENT_BLUE};
            padding: 10px;
        """)
        right_layout.addWidget(title)
        
        # Open file button
        self.btn_open = QPushButton("📂 Open Mesh File")
        self.btn_open.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_GREEN}, stop:1 #00cc66);
                color: {DARK_BG};
                border: none;
                padding: 14px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff99, stop:1 #00dd77);
            }}
        """)
        self.btn_open.clicked.connect(self.open_mesh_dialog)
        right_layout.addWidget(self.btn_open)
        
        # Measurements panel
        self.measurements_panel = MeasurementsPanel()
        right_layout.addWidget(self.measurements_panel)
        
        # Controls panel
        self.controls_panel = ControlsPanel()
        self.controls_panel.set_plotter(self.plotter)
        right_layout.addWidget(self.controls_panel)
        
        # Spacer
        right_layout.addStretch()
        
        # File info
        self.file_info = QLabel("No file loaded")
        self.file_info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setWordWrap(True)
        right_layout.addWidget(self.file_info)
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([900, 340])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Open a mesh file to begin")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {DARK_PANEL};
                color: {TEXT_SECONDARY};
                border-top: 1px solid {DARK_CARD};
                padding: 5px;
            }}
        """)
        
        # Menu bar style
        self.menuBar().setStyleSheet(f"""
            QMenuBar {{
                background-color: {DARK_PANEL};
                color: {TEXT_PRIMARY};
                padding: 5px;
            }}
            QMenuBar::item:selected {{
                background-color: {DARK_CARD};
            }}
            QMenu {{
                background-color: {DARK_PANEL};
                color: {TEXT_PRIMARY};
                border: 1px solid {DARK_CARD};
            }}
            QMenu::item:selected {{
                background-color: {ACCENT_BLUE};
                color: {DARK_BG};
            }}
        """)
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_mesh_action = QAction("&Open Mesh...", self)
        open_mesh_action.setShortcut("Ctrl+O")
        open_mesh_action.triggered.connect(self.open_mesh_dialog)
        file_menu.addAction(open_mesh_action)
        
        open_measurements_action = QAction("Open &Measurements...", self)
        open_measurements_action.setShortcut("Ctrl+M")
        open_measurements_action.triggered.connect(self.open_measurements_dialog)
        file_menu.addAction(open_measurements_action)
        
        file_menu.addSeparator()
        
        save_screenshot_action = QAction("&Save Screenshot...", self)
        save_screenshot_action.setShortcut("Ctrl+S")
        save_screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_screenshot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        for view_name in ["Front", "Back", "Left", "Right", "Top", "Reset (Iso)"]:
            action = QAction(view_name, self)
            action.triggered.connect(
                lambda checked, v=view_name.lower().split()[0]: 
                self.controls_panel.set_view(v if v != 'reset' else 'iso')
            )
            view_menu.addAction(action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def open_mesh_dialog(self):
        """Open file dialog to select a mesh file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh File", "",
            "Mesh Files (*.obj *.ply *.stl);;OBJ Files (*.obj);;All Files (*)"
        )
        
        if filename:
            self.load_mesh(filename)
            self.auto_load_measurements(filename)
    
    def open_measurements_dialog(self):
        """Open file dialog to select measurements file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Measurements File", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            self.load_measurements(filename)
    
    def load_mesh(self, filepath: str):
        """Load and display a mesh file."""
        try:
            self.statusBar().showMessage(f"Loading {filepath}...")
            
            # Load with trimesh
            mesh = trimesh.load(filepath, process=False)
            
            # Fix SMPL-X mesh orientation
            # The mesh comes with Y pointing down and facing backwards
            # Rotate 180 degrees around X-axis to flip upright
            # Then rotate 180 degrees around Y-axis to face forward
            vertices = mesh.vertices.copy()
            
            # Rotation matrix for 180° around X-axis: flips Y and Z
            # [1,  0,  0]
            # [0, -1,  0]
            # [0,  0, -1]
            vertices[:, 1] = -vertices[:, 1]  # Flip Y
            vertices[:, 2] = -vertices[:, 2]  # Flip Z
            
            # Convert to PyVista
            faces = np.hstack([[3] + list(f) for f in mesh.faces])
            pv_mesh = pv.PolyData(vertices, faces)
            
            # Clear and add new mesh with better lighting
            self.plotter.clear()
            
            # Add mesh with skin-like appearance
            self.plotter.add_mesh(
                pv_mesh,
                color='#e8beac',  # Skin tone
                smooth_shading=True,
                ambient=0.2,
                diffuse=0.8,
                specular=0.3,
                specular_power=15,
                show_edges=False,
            )
            
            # Add better lighting
            self.plotter.add_light(pv.Light(
                position=(5, 5, 5),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.8
            ))
            self.plotter.add_light(pv.Light(
                position=(-5, 3, -5),
                focal_point=(0, 0, 0),
                color='#aaccff',
                intensity=0.4
            ))
            
            # Add axes
            self.plotter.add_axes(color='white', line_width=2)
            
            # Set camera to front view (Z+ is front for SMPL-X)
            self.plotter.reset_camera()
            bounds = pv_mesh.bounds
            center = pv_mesh.center
            size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            
            # Position camera in front of the body
            self.plotter.camera_position = [
                (center[0], center[1], center[2] + size * 2.5),  # Camera position
                center,  # Focal point
                (0, 1, 0)  # View up (Y is up)
            ]
            
            self.current_mesh = pv_mesh
            self.mesh_path = filepath
            
            # Update UI
            filename = os.path.basename(filepath)
            self.file_info.setText(f"📁 {filename}\n{len(mesh.vertices):,} vertices | {len(mesh.faces):,} faces")
            self.statusBar().showMessage(f"Loaded: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mesh:\n{str(e)}")
            self.statusBar().showMessage("Error loading mesh")
    
    def load_measurements(self, filepath: str):
        """Load and display measurements."""
        try:
            with open(filepath, 'r') as f:
                measurements = json.load(f)
            
            self.measurements_panel.update_measurements(measurements)
            self.measurements_path = filepath
            self.statusBar().showMessage(f"Loaded measurements from {os.path.basename(filepath)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load measurements:\n{str(e)}")
    
    def auto_load_measurements(self, mesh_path: str):
        """Try to automatically find and load matching measurements file."""
        mesh_dir = os.path.dirname(mesh_path)
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        
        # Try common naming patterns
        patterns = [
            f"{mesh_name}_measurements.json",
            f"{mesh_name}.measurements.json",
            f"{mesh_name}_params.json",
        ]
        
        for pattern in patterns:
            meas_path = os.path.join(mesh_dir, pattern)
            if os.path.exists(meas_path):
                self.load_measurements(meas_path)
                return
        
        # Clear measurements if no file found
        self.measurements_panel.clear()
    
    def save_screenshot(self):
        """Save screenshot of current view."""
        if self.current_mesh is None:
            QMessageBox.warning(self, "Warning", "No mesh loaded")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if filename:
            self.plotter.screenshot(filename)
            self.statusBar().showMessage(f"Screenshot saved: {filename}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SHAPY Body Viewer",
            """<h2>SHAPY Body Viewer</h2>
            <p>Version 1.1</p>
            <p>A cross-platform 3D body mesh viewer with anthropometric measurements display.</p>
            <p><b>Controls:</b></p>
            <ul>
                <li>Left Mouse: Rotate</li>
                <li>Middle Mouse: Pan</li>
                <li>Scroll: Zoom</li>
            </ul>
            <p>Part of the SHAPY CPU project.</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        self.plotter.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description='SHAPY Body Viewer - 3D Body Mesh Visualization'
    )
    parser.add_argument('--mesh', '-m', type=str, default=None,
                        help='Path to mesh file (.obj, .ply)')
    parser.add_argument('--measurements', '-M', type=str, default=None,
                        help='Path to measurements JSON file')
    
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(DARK_BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(DARK_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(DARK_CARD))
    palette.setColor(QPalette.ToolTipBase, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(DARK_CARD))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(ACCENT_BLUE))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_BLUE))
    palette.setColor(QPalette.HighlightedText, QColor(DARK_BG))
    app.setPalette(palette)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = SHAPYViewer(
        mesh_path=args.mesh,
        measurements_path=args.measurements
    )
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

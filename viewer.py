'''3D Model Viewer Application using PyQt5 and VTK'''

# Standard library imports
import os # For file path operations
import math # For mathematical operations
import tempfile  # For creating temporary files

# Third-party imports
import vtk # VTK core
import laspy # For LAS loading
import numpy as np # For numerical operations
import pye57 # For E57 loading
import trimesh # For alternative OBJ loading

from PyQt5.QtCore import Qt # For Qt constants
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QAction, QInputDialog, QListWidget, QDockWidget,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QColorDialog, QSpinBox, QWidget, QPlainTextEdit  # Basic PyQt5 imports
)

import vtkmodules.qt  # Ensure VTK Qt support is loaded
# REMOVE legacy QGLWidget override (it can crash on macOS)
# vtkmodules.qt.QVTKRWIBase = 'QGLWidget'

# Qt widget compatibility: prefer NativeWidget, else classic Interactor
try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKOpenGLNativeWidget as QVTKWidgetBase
    VTK_QT_WIDGET = "QVTKOpenGLNativeWidget"
except ImportError:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidgetBase
    VTK_QT_WIDGET = "QVTKRenderWindowInteractor"

DEFAULT_TEXTURE = "assets/default_texture.png"

def create_outline_actor(polydata):
    ''' LAS/E57 - Create an outline actor for the given polydata '''
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(polydata)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline.GetOutputPort())
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(1, 1, 1)  # White box
    return outline_actor


def load_las_as_vtk(filename):
    '''Load LAS file and convert to VTK PolyData'''
    if not isinstance(filename, str) or not os.path.isfile(filename):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        las = laspy.read(filename)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return vtk.vtkPolyData()  # Return empty polydata on error

    if not hasattr(las, 'x') or not hasattr(las, 'y') or not hasattr(las, 'z'):
        print("LAS file missing coordinate data.")
        return vtk.vtkPolyData()

    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Optional: color support if available
    has_color = all(hasattr(las, attr) for attr in ['red', 'green', 'blue'])
    colors = None
    if has_color and las.red is not None and las.green is not None and las.blue is not None:
        try:
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            colors = (colors / 65535.0 * 255).astype(np.uint8)  # Normalize to 0–255
        except Exception as e:
            print(f"Error processing color data: {e}")
            colors = None

    # Convert to VTK points
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    if colors is not None:
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for c in colors:
            vtk_colors.InsertNextTuple3(*c)
        polydata.GetPointData().SetScalars(vtk_colors)
    else:
        print("No color data found or processed for LAS file.")

    return polydata


def load_e57_as_vtk(filename):
    ''' Load E57 file and convert to VTK PolyData '''
    try:
        e57 = pye57.E57(filename)
        scan = e57.read_scan(0)  # Read the first scan
    except Exception as e:
        print(f"Error reading E57 file: {e}")
        return vtk.vtkPolyData()

    try:
        points = np.vstack((scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"])).transpose()
    except Exception as e:
        print(f"E57 missing coordinate data: {e}")
        return vtk.vtkPolyData()

    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Optional: color support if available
    if all(k in scan for k in ("colorRed", "colorGreen", "colorBlue")):
        try:
            colors = np.vstack((scan["colorRed"], scan["colorGreen"], scan["colorBlue"])).transpose()
            # Normalize if data looks 16-bit
            if colors.max() > 255:
                colors = (colors / 65535.0 * 255.0)
            colors = colors.clip(0, 255).astype(np.uint8)
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetNumberOfComponents(3)
            vtk_colors.SetName("Colors")
            for c in colors:
                vtk_colors.InsertNextTuple3(*c)
            polydata.GetPointData().SetScalars(vtk_colors)
        except Exception as e:
            print(f"Error processing E57 color data: {e}")

    return polydata


class CustomQVTKRenderWindowInteractor(QVTKWidgetBase):
    """Custom QVTKRenderWindowInteractor to forward key and mouse events to parent."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def wheelEvent(self, event):
        camera = self.parent.renderer.GetActiveCamera()
        zoom_factor = 0.9 if event.angleDelta().y() < 0 else 1.1  # up=in, down=out on Mac
        camera.Dolly(zoom_factor)
        self.parent.renderer.ResetCameraClippingRange()
        self.parent.vtk_widget.GetRenderWindow().Render()

    def keyPressEvent(self, event):
        if self.parent.dialog_active:
            return
        # Forward B key to parent for background toggle
        if event.key() == Qt.Key_B:
            self.parent.toggle_background()
            return
        # Existing camera rotation logic
        camera = self.parent.renderer.GetActiveCamera()
        rotation_step = 5
        if event.key() == Qt.Key_Left:
            camera.Azimuth(rotation_step)
        elif event.key() == Qt.Key_Right:
            camera.Azimuth(-rotation_step)
        elif event.key() == Qt.Key_Up:
            camera.Elevation(-rotation_step)
        elif event.key() == Qt.Key_Down:
            camera.Elevation(rotation_step)
        camera.OrthogonalizeViewUp()
        self.parent.renderer.ResetCameraClippingRange()
        self.parent.vtk_widget.GetRenderWindow().Render()

        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            camera.Dolly(1.1)  # Zoom in
            self.parent.renderer.ResetCameraClippingRange()
            self.parent.vtk_widget.GetRenderWindow().Render()
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            camera.Dolly(0.9)  # Zoom out
            self.parent.renderer.ResetCameraClippingRange()
            self.parent.vtk_widget.GetRenderWindow().Render()

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.parent.dialog_active:
            return  # Ignore mouse events if a dialog is active
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.parent.dialog_active:
            return  # Ignore mouse events if a dialog is active
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if self.parent.dialog_active:
            return  # Ignore mouse events if a dialog is active
        super().mouseReleaseEvent(event)


class LightAdjustDialog(QDialog):
    '''Dialog for adjusting multiple lights.'''
    def __init__(self, parent, lights):
        super().__init__(parent)
        self.setWindowTitle("Custom Adjust Lights")
        self.lights = lights
        self.parent = parent
        main_layout = QVBoxLayout()
        row_layout = QHBoxLayout()  # Horizontal layout for all light controls
        self.controls = []

        for i, light in enumerate(self.lights):
            group = QVBoxLayout()
            group.addWidget(QLabel(f"Light {i+1}"))

            # Intensity slider
            intensity_slider = QSlider(Qt.Horizontal)
            intensity_slider.setMinimum(0)
            intensity_slider.setMaximum(100)
            intensity_slider.setValue(int(light.GetIntensity() * 100))
            intensity_slider.valueChanged.connect(lambda val, idx=i: self.update_intensity(idx, val))
            group.addWidget(QLabel("Intensity"))
            group.addWidget(intensity_slider)

            # Position controls
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(QLabel("X"))
            x_spin = QSpinBox()
            x_spin.setRange(-10, 10)
            x_spin.setValue(int(light.GetPosition()[0]))
            x_spin.valueChanged.connect(lambda val, idx=i: self.update_position(idx, 0, val))
            pos_layout.addWidget(x_spin)

            pos_layout.addWidget(QLabel("Y"))
            y_spin = QSpinBox()
            y_spin.setRange(-10, 10)
            y_spin.setValue(int(light.GetPosition()[1]))
            y_spin.valueChanged.connect(lambda val, idx=i: self.update_position(idx, 1, val))
            pos_layout.addWidget(y_spin)

            pos_layout.addWidget(QLabel("Z"))
            z_spin = QSpinBox()
            z_spin.setRange(-10, 10)
            z_spin.setValue(int(light.GetPosition()[2]))
            z_spin.valueChanged.connect(lambda val, idx=i: self.update_position(idx, 2, val))
            pos_layout.addWidget(z_spin)

            group.addLayout(pos_layout)

            # Color picker
            color_btn = QPushButton("Set Color")
            color_btn.clicked.connect(lambda _, idx=i: self.pick_color(idx))
            group.addWidget(color_btn)

            row_layout.addLayout(group)  # Add each light's controls to the horizontal row
            self.controls.append((intensity_slider, x_spin, y_spin, z_spin, color_btn))

        main_layout.addLayout(row_layout)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
        self.setLayout(main_layout)


    def update_intensity(self, idx, val):
        '''Update light intensity.'''
        self.lights[idx].SetIntensity(val / 100.0)
        self.parent.vtk_widget.GetRenderWindow().Render()

    def update_position(self, idx, axis, val):
        '''Update light position along specified axis (0=X,1=Y,2=Z).'''
        pos = list(self.lights[idx].GetPosition())
        pos[axis] = val
        self.lights[idx].SetPosition(pos)
        self.parent.vtk_widget.GetRenderWindow().Render()

    def pick_color(self, idx):
        '''Open color dialog to pick light color.'''
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgbF()[:3]
            self.lights[idx].SetColor(*rgb)
            self.parent.vtk_widget.GetRenderWindow().Render()


class ViewerApp(QMainWindow):
    '''Main application window for 3D model viewing.'''
    def __init__(self):
        '''Initialize the main application window and VTK renderer.'''
        super().__init__()
        self.dialog_active = False  # Flag to track if a dialog is open
        self.setWindowTitle("3D Model Viewer")
        self.setGeometry(100, 100, 800, 600) # Set initial window size
        self.current_lighting_preset = "studio"
        # Measurement state
        self.measure_points = []  # Store measurement points
        self.measure_markers = [] # Store marker actors for measurements
        self.measure_lines = []   # Store line actors for measurements
        self.measure_labels = []
        self.measure_history = []
        self.undo_stack = []
        self.redo_stack = []
        self.annotations = []

        # VTK widget
        self.vtk_widget = CustomQVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)

        # Choose render window based on widget type
        if VTK_QT_WIDGET == "QVTKOpenGLNativeWidget":
            # Modern path: use GenericOpenGL render window (no Initialize call)
            ren_win = vtk.vtkGenericOpenGLRenderWindow()
            self.vtk_widget.SetRenderWindow(ren_win)
            self.renderer = vtk.vtkRenderer()
            ren_win.AddRenderer(self.renderer)
            # Interactor from the widget/render window
            self.interactor = self.vtk_widget.GetInteractor() or ren_win.GetInteractor()
        else:
            # Classic path: use the widget's own render window (no GenericOpenGL here)
            ren_win = self.vtk_widget.GetRenderWindow()
            self.renderer = vtk.vtkRenderer()
            ren_win.AddRenderer(self.renderer)
            # The widget acts as interactor; initialize it
            self.interactor = self.vtk_widget
            try:
                # Initialize once; do not call Start() in Qt apps
                self.vtk_widget.Initialize()
            except Exception:
                pass

        self.vtk_widget.setFocus()


        # Enable depth peeling
        self.enable_depth_peeling()

        # Mouse styles
        self.style_switch = vtk.vtkInteractorStyleSwitch()
        self.interactor.SetInteractorStyle(self.style_switch)
        self.mouse_styles = ["trackball", "terrain", "joystick"]
        self.current_mouse_style = "trackball"
        self._terrain_style = vtk.vtkInteractorStyleTerrain()
        self.set_mouse_style(self.current_mouse_style)

        # Keep a terrain style instance alive (used on-demand)
        self._terrain_style = vtk.vtkInteractorStyleTerrain()

        self.set_mouse_style(self.current_mouse_style)

        self.history_list = QListWidget()
        self.history_list.itemSelectionChanged.connect(self.update_measurement_highlight)

        # Create a widget to hold controls and history list
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        # Measurement control buttons
        self.btn_start = QPushButton("Start Measuring")
        self.btn_start.clicked.connect(self.activate_measure_mode)
        history_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop Measuring")
        self.btn_stop.clicked.connect(self.cancel_measurement)
        history_layout.addWidget(self.btn_stop)
        self.btn_clear_last = QPushButton("Clear Last Measurement")
        self.btn_clear_last.clicked.connect(self.clear_last_measurement)
        history_layout.addWidget(self.btn_clear_last)
        btn_clear_all = QPushButton("Clear All Measurements")
        btn_clear_all.clicked.connect(self.clear_all_measurements)
        history_layout.addWidget(btn_clear_all)
        btn_delete_selected = QPushButton("Delete Selected Measurement")
        btn_delete_selected.clicked.connect(self.delete_selected_measurement)
        history_layout.addWidget(btn_delete_selected)
        # Undo/Redo buttons
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo_measurement)
        history_layout.addWidget(btn_undo)
        btn_redo = QPushButton("Redo")
        btn_redo.clicked.connect(self.redo_measurement)
        history_layout.addWidget(btn_redo)
        # Add the history list below the controls
        history_layout.addWidget(self.history_list)

        self.history_dock = QDockWidget("Measurements", self)
        self.history_dock.setWidget(history_widget)
        self.history_dock.setFeatures(QDockWidget.DockWidgetMovable) # Only allow moving, not closing or floating
        self.addDockWidget(Qt.LeftDockWidgetArea, self.history_dock)

        # Model Info dock
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("font-family: Menlo, monospace; font-size: 12px;")
        self.info_dock = QDockWidget("Model Info", self)
        self.info_dock.setWidget(self.info_text)
        self.info_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)

        self.background_colors = [
            (1, 1, 1),        # White
            (0.9, 0.9, 0.9),  # Light Gray
            (0.2, 0.2, 0.2),  # Dark Gray
            (0, 0, 0)         # Black
        ]
        self.background_index = 2
        self.set_background_color(*self.background_colors[self.background_index])

        # Menu
        self.init_menu()
        # Load default model
        default_model = "assets/house.obj"
        if os.path.exists(default_model):
            self.load_model(default_model)

        #self.interactor.Initialize()

    def init_menu(self):
        '''Initialize the menu bar and actions.'''
        menubar = self.menuBar()

        ########### FILE MENU ##############
        file_menu = menubar.addMenu("File")
        # Open file action
        open_action = QAction("Open Model", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        # Save file action
        save_action = QAction("Save Model", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        ########### VIEW MENU ##############
        view_menu = menubar.addMenu("View")
        # Reset view action
        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("R")
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        # Add zoom in/out actions
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(lambda: self.zoom_camera(1.1))  # Zoom in
        view_menu.addAction(zoom_in_action)
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(lambda: self.zoom_camera(0.9))  # Zoom out
        view_menu.addAction(zoom_out_action)
        # Refresh model info action
        refresh_info_action = QAction("Refresh Model Info", self)
        refresh_info_action.setShortcut("I")
        refresh_info_action.triggered.connect(self.update_model_info)
        view_menu.addAction(refresh_info_action)

        ########### MOUSE MENU ##############
        mouse_menu = menubar.addMenu("Mouse")
        act_trackball = QAction("Style: Trackball", self)
        act_trackball.triggered.connect(lambda: self.set_mouse_style("trackball"))
        mouse_menu.addAction(act_trackball)
        act_terrain = QAction("Style: Terrain", self)
        act_terrain.triggered.connect(lambda: self.set_mouse_style("terrain"))
        mouse_menu.addAction(act_terrain)
        act_joystick = QAction("Style: Joystick", self)
        act_joystick.triggered.connect(lambda: self.set_mouse_style("joystick"))
        mouse_menu.addAction(act_joystick)
        act_cycle = QAction("Cycle Mouse Style", self)
        act_cycle.setShortcut("M")
        act_cycle.triggered.connect(self.cycle_mouse_style)
        mouse_menu.addAction(act_cycle)

        ########### WIREFRAME MENU ##############
        wireframe_menu = menubar.addMenu("Wireframe")
        toggle_wireframe = QAction("Toggle Wireframe", self)
        toggle_wireframe.setShortcut("W")  # Optional shortcut
        toggle_wireframe.triggered.connect(self.toggle_wireframe)
        wireframe_menu.addAction(toggle_wireframe)

        ########### MEASURE MENU ##############
        measure_menu = menubar.addMenu("Measure")
        # START MEASURING ACTION
        start_measure_action = QAction("Start Measuring", self)
        start_measure_action.triggered.connect(self.activate_measure_mode)
        measure_menu.addAction(start_measure_action)
        # STOP MEASURING ACTION
        stop_measure_action = QAction("Stop Measuring", self)
        stop_measure_action.triggered.connect(self.cancel_measurement)
        measure_menu.addAction(stop_measure_action)
        # CLEAR LAST MEASUREMENT ACTION
        clear_last_measure_action = QAction("Clear Last Measurement", self)
        clear_last_measure_action.triggered.connect(self.clear_last_measurement)
        measure_menu.addAction(clear_last_measure_action)
        # CLEAR ALL MEASUREMENTS ACTION
        clear_measure_action = QAction("Clear All Measurements", self)
        clear_measure_action.triggered.connect(self.clear_all_measurements)
        measure_menu.addAction(clear_measure_action)
        # UNDO MEASUREMENT ACTION
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_measurement)
        measure_menu.addAction(undo_action)
        # REDO MEASUREMENT ACTION
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo_measurement)
        measure_menu.addAction(redo_action)

        ########### ANNOTATION MENU ##############
        annotation_menu = menubar.addMenu("Annotation")
        add_annotation_action = QAction("Add Annotation", self)
        add_annotation_action.triggered.connect(self.activate_annotation_mode)
        annotation_menu.addAction(add_annotation_action)
        #  CLEAR LAST ANNOTATION ACTION
        clear_last_annotation_action = QAction("Clear Last Annotation", self)
        clear_last_annotation_action.triggered.connect(self.clear_last_annotation)
        annotation_menu.addAction(clear_last_annotation_action)
        #  CLEAR ALL ANNOTATIONS ACTION
        clear_annotations_action = QAction("Clear All Annotations", self)
        clear_annotations_action.triggered.connect(self.clear_all_annotations)
        annotation_menu.addAction(clear_annotations_action)

        ############# LIGHTING MENU ################
        lighting_menu = menubar.addMenu("Lighting")
        # cycle lighting preset
        cycle_lighting = QAction("Cycle Lighting Preset", self)
        cycle_lighting.setShortcut("L")  # Press 'L' to cycle
        cycle_lighting.triggered.connect(self.cycle_lighting_preset)
        lighting_menu.addAction(cycle_lighting)
        # specific lighting presets
        lighting_default = QAction("Lighting: Default", self)
        lighting_default.triggered.connect(lambda: self.apply_lighting_preset("default"))
        lighting_menu.addAction(lighting_default)
        lighting_studio = QAction("Lighting: Studio", self)
        lighting_studio.triggered.connect(lambda: self.apply_lighting_preset("studio"))
        lighting_menu.addAction(lighting_studio)
        lighting_off = QAction("Lighting: Off", self)
        lighting_off.triggered.connect(lambda: self.apply_lighting_preset("off"))
        lighting_menu.addAction(lighting_off)
        # Add another option: "Lighting: Warm"
        lighting_warm = QAction("Lighting: Warm", self)
        lighting_warm.triggered.connect(lambda: self.apply_lighting_preset("warm"))
        lighting_menu.addAction(lighting_warm)
        # light adjustment dialog
        adjust_lights = QAction("Custom Adjust Lights", self)
        adjust_lights.triggered.connect(self.open_light_dialog)
        lighting_menu.addAction(adjust_lights)

        ############# BACKGROUND COLOR MENU ################
        background_menu = menubar.addMenu("Background")
        bg_white = QAction("White", self)
        bg_white.setShortcut("Ctrl+1")
        bg_white.triggered.connect(lambda: self.set_background_color(1, 1, 1))
        background_menu.addAction(bg_white)
        bg_light_gray = QAction("Light Gray", self)
        bg_light_gray.setShortcut("Ctrl+2")
        bg_light_gray.triggered.connect(lambda: self.set_background_color(0.9, 0.9, 0.9))
        background_menu.addAction(bg_light_gray)
        bg_dark_gray = QAction("Dark Gray", self)
        bg_dark_gray.setShortcut("Ctrl+3")
        bg_dark_gray.triggered.connect(lambda: self.set_background_color(0.2, 0.2, 0.2))
        background_menu.addAction(bg_dark_gray)
        bg_black = QAction("Black", self)
        bg_black.setShortcut("Ctrl+4")
        bg_black.triggered.connect(lambda: self.set_background_color(0, 0, 0))
        background_menu.addAction(bg_black)
        # Optionally, add a custom color picker
        bg_custom = QAction("Custom...", self)
        bg_custom.setShortcut("Ctrl+5")
        bg_custom.triggered.connect(self.pick_custom_background)
        background_menu.addAction(bg_custom)
        # Add cycle background option
        bg_cycle = QAction("Cycle Background", self)
        bg_cycle.setShortcut("B")
        bg_cycle.triggered.connect(self.toggle_background)
        background_menu.addAction(bg_cycle)

        ########### DEBUG MENU ##############
        debug_menu = menubar.addMenu("Debug")
        dbg_unlit = QAction("Show Unlit White", self)
        dbg_unlit.triggered.connect(self.debug_unlit_white)
        debug_menu.addAction(dbg_unlit)

        dbg_relit = QAction("Restore Lit Materials", self)
        dbg_relit.triggered.connect(self.debug_restore_lit)
        debug_menu.addAction(dbg_relit)

        dbg_cube = QAction("Add Test Cube", self)
        dbg_cube.triggered.connect(self.debug_add_cube)
        debug_menu.addAction(dbg_cube)

        dbg_dp = QAction("Toggle Depth Peeling", self)
        dbg_dp.triggered.connect(self.toggle_depth_peeling)
        debug_menu.addAction(dbg_dp)


    def save_polydata(self, filename, writer_class, polydata):
        '''Save the given polydata to a file using the specified writer class.'''
        writer = writer_class()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()

    def save_file(self):
        """Save all actors in the scene to a file."""
        # Prompt the user for a save location and file format
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Model File",
            "",
            "3D Model Files (*.obj *.stl *.ply)"
        )

        if not filename:
            return  # User canceled the save dialog

        # Determine the file extension
        ext = os.path.splitext(filename)[1].lower()

        try:
            # Combine all actors' polydata into one
            append_filter = vtk.vtkAppendPolyData()
            actors = self.renderer.GetActors()
            actors.InitTraversal()
            for i in range(actors.GetNumberOfItems()):
                actor = actors.GetNextActor()
                mapper = actor.GetMapper()
                if mapper:
                    polydata = mapper.GetInput()
                    if polydata:
                        append_filter.AddInputData(polydata)

            append_filter.Update()
            combined_polydata = append_filter.GetOutput()

            # Save the combined polydata to the selected file format
            if ext == ".obj":
                # Save as OBJ with textures
                exporter = vtk.vtkOBJExporter()
                exporter.SetFilePrefix(os.path.splitext(filename)[0])  # Exclude extension
                exporter.SetRenderWindow(self.vtk_widget.GetRenderWindow())
                exporter.Write()
            elif ext == ".stl":
                self.save_polydata(filename, vtk.vtkSTLWriter, combined_polydata)
            elif ext == ".ply":
                self.save_polydata(filename, vtk.vtkPLYWriter, combined_polydata)
            else:
                self.statusBar().showMessage("Unsupported file format.")
                return

            self.statusBar().showMessage(f"Model saved to: {filename}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving model: {e}")
            print(f"Error saving model: {e}")

    def enable_depth_peeling(self):
        """Enable depth peeling for proper transparency rendering."""
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.SetAlphaBitPlanes(1)  # Enable alpha bit planes for transparency
        render_window.SetMultiSamples(0)  # Disable multisampling for better transparency

        # Configure depth peeling
        self.renderer.SetUseDepthPeeling(1)
        self.renderer.SetMaximumNumberOfPeels(100)  # Maximum number of depth peels
        self.renderer.SetOcclusionRatio(0.1)  # Occlusion ratio (lower is better quality)

    def zoom_camera(self, factor):
        '''Zoom the camera in or out by a given factor.'''
        camera = self.renderer.GetActiveCamera()
        camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def reset_view(self):
        """Reset the camera to its default position."""
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor_count = actors.GetNumberOfItems()

        if actor_count == 0:
            self.statusBar().showMessage("No actors in the scene to reset view.")
            return

        bounds = self.renderer.ComputeVisiblePropBounds()

        if bounds == (0, 0, 0, 0, 0, 0):
            self.statusBar().showMessage("Scene bounds are invalid. Cannot reset view.")
            return

        # Reset the camera
        self.renderer.ResetCamera()

        # Adjust the camera distance
        camera = self.renderer.GetActiveCamera()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        # Set the focal point to the center of the bounds
        camera.SetFocalPoint(center_x, center_y, center_z)

        # Position the camera at a distance proportional to the size of the scene
        camera.SetPosition(center_x, center_y, center_z + max_dim * 2)  # Adjust the multiplier as needed
        camera.SetViewUp(0, 1, 0)

        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("View reset to default position.")

    def set_mouse_style(self, name: str):
        """Set mouse interaction style: trackball | terrain | joystick."""
        name = name.lower()
        if name == "trackball":
            # Reattach the switch, then select trackball
            self.interactor.SetInteractorStyle(self.style_switch)
            self.style_switch.SetCurrentStyleToTrackballCamera()
        elif name == "joystick":
            # Reattach the switch, then select joystick
            self.interactor.SetInteractorStyle(self.style_switch)
            self.style_switch.SetCurrentStyleToJoystickCamera()
        elif name == "terrain":
            # Not available via switch on this VTK; use terrain style directly
            self.interactor.SetInteractorStyle(self._terrain_style)
        else:
            # Fallback to trackball
            self.interactor.SetInteractorStyle(self.style_switch)
            self.style_switch.SetCurrentStyleToTrackballCamera()
            name = "trackball"

        self.current_mouse_style = name
        self.statusBar().showMessage(f"Mouse style: {name.capitalize()}")

    def cycle_mouse_style(self):
        """Cycle through available mouse styles."""
        i = self.mouse_styles.index(self.current_mouse_style)
        nxt = self.mouse_styles[(i + 1) % len(self.mouse_styles)]
        self.set_mouse_style(nxt)

    def load_texture(self, texture_path):
        '''Load a texture from file and return a vtkTexture object'''
        reader_factory = vtk.vtkImageReader2Factory()
        texture_reader = reader_factory.CreateImageReader2(texture_path)
        if texture_reader:
            texture_reader.SetFileName(texture_path)
            texture_reader.Update()
            texture = vtk.vtkTexture()
            texture.SetInputConnection(texture_reader.GetOutputPort())
            return texture
        else:
            print(f"Failed to load texture: {texture_path}")
            return None

    def trimesh_to_vtk(self, mesh):
        '''Convert a trimesh.Trimesh object to vtkPolyData'''
        # Create VTK points
        points = vtk.vtkPoints()
        for vertex in mesh.vertices:
            points.InsertNextPoint(vertex)

        # Create VTK cells (triangles)
        cells = vtk.vtkCellArray()
        for face in mesh.faces:
            triangle = vtk.vtkTriangle()
            for i, vertex_index in enumerate(face):
                triangle.GetPointIds().SetId(i, vertex_index)
            cells.InsertNextCell(triangle)

        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Add vertex colors if available
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            for color in mesh.visual.vertex_colors[:, :3]:  # Ignore alpha channel
                colors.InsertNextTuple3(*color)
            polydata.GetPointData().SetScalars(colors)

        return polydata

    def load_point_cloud(self, filename, loader_function):
        '''Load a point cloud file using the specified loader function.'''
        polydata = loader_function(filename)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        outline_actor = create_outline_actor(polydata)
        self.renderer.AddActor(outline_actor)
        actor.GetProperty().SetPointSize(2)
        self.renderer.AddActor(actor)
        self.update_model_info()
        self.reset_view()

    def deactivate_all_modes(self):
        '''Deactivate all active modes.'''
        if hasattr(self, "left_click_observer"):
            self.vtk_widget.RemoveObserver(self.left_click_observer)
            del self.left_click_observer
        if hasattr(self, "right_click_observer"):
            self.vtk_widget.RemoveObserver(self.right_click_observer)
            del self.right_click_observer
        if hasattr(self, "annotation_observer"):
            self.vtk_widget.RemoveObserver(self.annotation_observer)
            del self.annotation_observer

    def keyPressEvent(self, event):
        """Forward key press events to the VTK widget's handler."""
        self.vtk_widget.keyPressEvent(event)


    def update_model_info(self):
        """Collect scene stats and show in the Model Info dock."""
        try:
            actors = self.renderer.GetActors()
            actors.InitTraversal()
            n_actors = actors.GetNumberOfItems()

            total_points = 0
            total_polys = 0
            total_lines = 0
            total_strips = 0
            textured = 0

            for _ in range(n_actors):
                a = actors.GetNextActor()
                if a is None:
                    continue
                mapper = a.GetMapper()
                if mapper is None:
                    continue
                data = mapper.GetInput()
                if isinstance(data, vtk.vtkPolyData):
                    total_points += int(data.GetNumberOfPoints() or 0)
                    total_polys  += int(data.GetNumberOfPolys() or 0)
                    total_lines  += int(data.GetNumberOfLines() or 0)
                    total_strips += int(data.GetNumberOfStrips() or 0)
                if a.GetTexture() is not None:
                    textured += 1

            bounds = self.renderer.ComputeVisiblePropBounds()
            bx = (bounds[0], bounds[1])
            by = (bounds[2], bounds[3])
            bz = (bounds[4], bounds[5])
            sx = bx[1] - bx[0]
            sy = by[1] - by[0]
            sz = bz[1] - bz[0]

            info = []
            info.append("Model Info")
            info.append("-----------------------------")
            info.append(f"Actors:            {n_actors}")
            info.append(f"Textured actors:   {textured}")
            info.append("")
            info.append(f"Points (vertices): {total_points}")
            info.append(f"Polys (faces):     {total_polys}")
            info.append(f"Lines:             {total_lines}")
            info.append(f"Strips:            {total_strips}")
            info.append("")
            info.append(f"Bounds X: [{bx[0]:.3f}, {bx[1]:.3f}]  size: {sx:.3f}")
            info.append(f"Bounds Y: [{by[0]:.3f}, {by[1]:.3f}]  size: {sy:.3f}")
            info.append(f"Bounds Z: [{bz[0]:.3f}, {bz[1]:.3f}]  size: {sz:.3f}")

            self.info_text.setPlainText("\n".join(info))
            self.statusBar().showMessage("Model info refreshed.")
        except Exception as e:
            self.info_text.setPlainText(f"Model Info\n-----------------------------\nError: {e}")


    ############### MEASUREMENT METHODS #################

    def activate_measure_mode(self):
        '''Activate measurement mode to measure distances.'''
        self.deactivate_all_modes()  # Ensure no other mode is active
        self.btn_start.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        self.statusBar().showMessage("Click points to measure. Right-click to finish.")
        self.left_click_observer = self.vtk_widget.AddObserver("LeftButtonPressEvent", self.on_measure_click)
        self.right_click_observer = self.vtk_widget.AddObserver("RightButtonPressEvent", self.finish_measure)

    def cancel_measurement(self):
        '''Cancel the current measurement mode, but keep visuals.'''
        if hasattr(self, "left_click_observer"):
            self.vtk_widget.RemoveObserver(self.left_click_observer)
            del self.left_click_observer
        if hasattr(self, "right_click_observer"):
            self.vtk_widget.RemoveObserver(self.right_click_observer)
            del self.right_click_observer
        # Do NOT remove marker/line actors or clear lists here!
        # Do NOT clear measure_points here; keep indices consistent for deletion logic
        # Only clear temporary measurement points if needed:
        # self.measure_points.clear()
        self.statusBar().showMessage("Measurement cancelled.")
        self.btn_start.setStyleSheet("")

    def on_measure_click(self, obj, event):
        '''Handle click to add measurement point.'''
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        pos = picker.GetPickPosition()
        if pos != (0.0, 0.0, 0.0):
            self.measure_points.append(pos)
            # Add marker (small sphere)
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pos)
            sphere.SetRadius(0.5 * max(1.0, self.renderer.GetActiveCamera().GetDistance() / 100.0))
            sphere.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0, 1, 0)  # Green marker
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            self.renderer.AddActor(actor)
            self.measure_markers.append(actor)
            self.vtk_widget.GetRenderWindow().Render()

            n = len(self.measure_points)
            if n > 1:
                p1 = self.measure_points[-2]
                p2 = self.measure_points[-1]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                self.measure_history.append(dist)
                self.undo_stack.append(('add', dist))
                self.history_list.addItem(f"Segment {n-1}: {dist:.3f}")
                self.statusBar().showMessage(f"Last segment: {dist:.3f} | Total: {self.measure_total():.3f}")
                # Draw line
                line = vtk.vtkLineSource()
                line.SetPoint1(p1)
                line.SetPoint2(p2)
                line.Update()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(line.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1, 0, 0)
                actor.GetProperty().SetLineWidth(2)
                self.renderer.AddActor(actor)
                self.measure_lines.append(actor)

                # Add distance label at the line midpoint
                mx = 0.5 * (p1[0] + p2[0])
                my = 0.5 * (p1[1] + p2[1])
                mz = 0.5 * (p1[2] + p2[2])
                label = vtk.vtkBillboardTextActor3D()
                label.SetInput(f"{dist:.3f}")
                label.SetPosition(mx, my, mz)
                label.GetTextProperty().SetColor(1, 1, 0)  # yellow
                label.GetTextProperty().SetFontSize(18)
                self.renderer.AddActor(label)
                self.measure_labels.append(label)

                self.vtk_widget.GetRenderWindow().Render()

    def finish_measure(self, obj, event):
        '''Finish measurement mode.'''
        if len(self.measure_points) > 1:
            total = self.measure_total()
            self.history_list.addItem(f"Total: {total:.3f}")
            self.measure_history.append(total)
            self.undo_stack.append(('add', total))
            self.statusBar().showMessage(f"Measurement finished. Total path length: {total:.3f}")
        else:
            self.statusBar().showMessage("Measurement cancelled or not enough points.")
        # Remove observers (guard with hasattr)
        if hasattr(self, "left_click_observer"):
            self.vtk_widget.RemoveObserver(self.left_click_observer)
            del self.left_click_observer
        if hasattr(self, "right_click_observer"):
            self.vtk_widget.RemoveObserver(self.right_click_observer)
            del self.right_click_observer
        self.btn_start.setStyleSheet("")

    def measure_total(self):
        '''Calculate total path length of measurements.'''
        total = 0.0
        for i in range(1, len(self.measure_points)):
            p1 = self.measure_points[i-1]
            p2 = self.measure_points[i]
            total += math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        return total

    def clear_all_measurements(self):
        '''Clear all measurements.'''
        # Remove marker actors
        if hasattr(self, "measure_markers"):
            for actor in list(self.measure_markers):
                self.renderer.RemoveActor(actor)
            self.measure_markers.clear()
        # Remove line actors
        if hasattr(self, "measure_lines"):
            for actor in list(self.measure_lines):
                self.renderer.RemoveActor(actor)
            self.measure_lines.clear()
        # Remove label actors
        if hasattr(self, "measure_labels"):
            for actor in list(self.measure_labels):
                self.renderer.RemoveActor(actor)
            self.measure_labels.clear()
        # Clear points/history/stacks/UI
        if hasattr(self, "measure_points"):
            self.measure_points.clear()
        self.measure_history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.history_list.clear()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("All measurements cleared.")

    def clear_last_measurement(self):
        '''Clear the last measurement segment.'''
        # Remove last marker
        if hasattr(self, "measure_markers") and self.measure_markers:
            actor = self.measure_markers.pop()
            self.renderer.RemoveActor(actor)
        # Remove last line
        if hasattr(self, "measure_lines") and self.measure_lines:
            actor = self.measure_lines.pop()
            self.renderer.RemoveActor(actor)
        # Remove last label
        if hasattr(self, "measure_labels") and self.measure_labels:
            actor = self.measure_labels.pop()
            self.renderer.RemoveActor(actor)
        # Remove last point
        if hasattr(self, "measure_points") and self.measure_points:
            self.measure_points.pop()
        # Remove last measurement from history and history list
        if self.measure_history:
            self.measure_history.pop()
        if self.history_list.count() > 0:
            self.history_list.takeItem(self.history_list.count() - 1)
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Last measurement cleared.")


    def delete_selected_measurement(self):
        '''Delete selected measurement rows; remove correct actors; skip "Total" rows.'''
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            self.statusBar().showMessage("No measurement selected.")
            return

        # Map rows -> segment index (skip "Total")
        row_to_seg = {}
        seg_idx = 0
        for i in range(self.history_list.count()):
            text = self.history_list.item(i).text()
            if text.startswith("Total"):
                row_to_seg[i] = None
            else:
                row_to_seg[i] = seg_idx
                seg_idx += 1

        rows = sorted([self.history_list.row(item) for item in selected_items], reverse=True)
        for row in rows:
            item = self.history_list.item(row)
            text = item.text() if item else ""
            self.history_list.takeItem(row)

            seg = row_to_seg.get(row, None)
            if seg is None:
                # Skip totals entirely; do not touch measure_history
                continue

            # Remove segment visuals
            total_lines = len(self.measure_lines)
            has_prev = (seg - 1) >= 0
            has_next = (seg + 1) < total_lines
            if seg < len(self.measure_lines):
                self.renderer.RemoveActor(self.measure_lines[seg])
                del self.measure_lines[seg]
            if seg < len(self.measure_labels):
                self.renderer.RemoveActor(self.measure_labels[seg])
                del self.measure_labels[seg]
            if (seg + 1) < len(self.measure_markers) and not has_next:
                self.renderer.RemoveActor(self.measure_markers[seg + 1])
                del self.measure_markers[seg + 1]
                if (seg + 1) < len(self.measure_points):
                    del self.measure_points[seg + 1]
            if seg < len(self.measure_markers) and not has_prev:
                self.renderer.RemoveActor(self.measure_markers[seg])
                del self.measure_markers[seg]
                if seg < len(self.measure_points):
                    del self.measure_points[seg]
        self.vtk_widget.GetRenderWindow().Render()
        self.update_measurement_highlight()
        self.statusBar().showMessage("Selected measurement(s) deleted.")


    def undo_measurement(self):
        '''Undo the last measurement.'''
        if self.undo_stack:
            action, value = self.undo_stack.pop()
            if action == 'add' and self.measure_history:
                removed = self.measure_history.pop()
                self.redo_stack.append(('add', removed))
                self.history_list.takeItem(self.history_list.count() - 1)
                self.statusBar().showMessage("Undo last measurement.")
            # Optionally, remove last marker/line actor here

    def redo_measurement(self):
        '''Redo the last undone measurement.'''
        if self.redo_stack:
            action, value = self.redo_stack.pop()
            if action == 'add':
                self.measure_history.append(value)
                self.undo_stack.append(('add', value))
                # Keep list format consistent
                self.history_list.addItem(f"Segment: {value:.3f}")
                self.statusBar().showMessage("Redo last measurement.")


    def update_measurement_highlight(self):
        '''Update the highlight of measurement segments.'''
        # Reset
        for line in self.measure_lines:
            p = line.GetProperty()
            p.SetColor(1, 0, 0); p.SetLineWidth(2.0)
        # Map rows to segments (skip 'Total')
        row_to_seg, seg = {}, 0
        for i in range(self.history_list.count()):
            txt = self.history_list.item(i).text()
            if txt.startswith("Total"):
                row_to_seg[i] = None
            else:
                row_to_seg[i] = seg; seg += 1
        # Highlight selected
        for item in self.history_list.selectedItems():
            r = self.history_list.row(item)
            s = row_to_seg.get(r)
            if s is not None and 0 <= s < len(self.measure_lines):
                p = self.measure_lines[s].GetProperty()
                p.SetColor(1, 1, 0); p.SetLineWidth(4.0)
        self.vtk_widget.GetRenderWindow().Render()


    ############### ANNOTATION METHODS #################

    def activate_annotation_mode(self):
        '''Activate annotation mode to add text annotations.'''
        self.deactivate_all_modes()  # Ensure no other mode is active

        self.statusBar().showMessage("Click a point to add annotation.")
        self.annotation_observer = self.vtk_widget.AddObserver("LeftButtonPressEvent", self.on_annotation_click)

    def on_annotation_click(self, obj, event):
        """Handle click to add annotation."""
        print("Annotation click detected.")
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        pos = picker.GetPickPosition()
        if pos != (0.0, 0.0, 0.0):
            print("Opening annotation dialog...")
            self.dialog_active = True  # Disable key and mouse events
            print(f"dialog_active set to {self.dialog_active}")
            text, ok = QInputDialog.getText(self, "Add Annotation", "Enter annotation text:")
            self.dialog_active = False  # Re-enable key and mouse events
            print(f"dialog_active set to {self.dialog_active}")
            if ok and text:
                text_actor = vtk.vtkBillboardTextActor3D()
                text_actor.SetInput(text)
                text_actor.SetPosition(pos)
                text_actor.GetTextProperty().SetColor(1, 1, 0)
                text_actor.GetTextProperty().SetFontSize(18)
                self.renderer.AddActor(text_actor)
                self.annotations.append(text_actor)
                self.vtk_widget.GetRenderWindow().Render()
            # Remove observer using stored ID
            if hasattr(self, "annotation_observer"):
                self.vtk_widget.RemoveObserver(self.annotation_observer)
                del self.annotation_observer
            self.statusBar().showMessage("Annotation added.")

    def clear_all_annotations(self):
        '''Clear all annotations.'''
        for actor in getattr(self, "annotations", []):
            self.renderer.RemoveActor(actor)
        self.annotations.clear()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("All annotations cleared.")

    def clear_last_annotation(self):
        '''Clear the last added annotation.'''
        if self.annotations:
            actor = self.annotations.pop()
            self.renderer.RemoveActor(actor)
            self.vtk_widget.GetRenderWindow().Render()
            self.statusBar().showMessage("Last annotation cleared.")
        else:
            self.statusBar().showMessage("No annotations to clear.")


    ###############  BACKGROUND METHODS #################

    def toggle_background(self):
        '''Cycle background color between white, light gray, dark gray, and black.'''
        self.background_index = (self.background_index + 1) % len(self.background_colors)
        color = self.background_colors[self.background_index]
        self.set_background_color(*color)

    def set_background_color(self, r, g, b):
        '''Set the background color of the renderer.'''
        self.renderer.SetBackground(r, g, b)
        # Update index to match the color if it's in the list
        for i, col in enumerate(self.background_colors):
            if all(abs(c1 - c2) < 0.01 for c1, c2 in zip(col, (r, g, b))):
                self.background_index = i
                break
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Background color set.")

    def pick_custom_background(self):
        '''Open a color picker dialog to select a custom background color.'''
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgbF()[:3]
            self.set_background_color(*rgb)


    ############### LIGHTING METHODS #################

    def open_light_dialog(self):
        '''Open a dialog to adjust light properties.'''
        if hasattr(self, "lights") and self.lights:
            dlg = LightAdjustDialog(self, self.lights)
            dlg.exec_()
        else:
            self.statusBar().showMessage("No lights to adjust in this preset.")

    def cycle_lighting_preset(self):
        '''Cycle through predefined lighting presets.'''
        presets = ["studio", "default", "warm", "off"]
        current_index = presets.index(self.current_lighting_preset)
        next_index = (current_index + 1) % len(presets)
        next_preset = presets[next_index]
        self.apply_lighting_preset(next_preset)

    def apply_lighting_preset(self, preset="studio"):
        '''Apply a predefined lighting preset.'''
        self.current_lighting_preset = preset
        self.renderer.RemoveAllLights()
        self.lights = []
        if hasattr(self.renderer, "SetTwoSidedLighting"):
            self.renderer.SetTwoSidedLighting(1)
        if hasattr(self.renderer, "SetLightFollowCamera"):
            self.renderer.SetLightFollowCamera(1)

        if preset == "default":
            # Four lights around the model (horizontal plane)
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                light = vtk.vtkLight()
                light.SetLightTypeToSceneLight()
                light.SetPosition(3 * math.cos(rad), 3 * math.sin(rad), 1)
                light.SetFocalPoint(0, 0, 0)
                light.SetIntensity(0.25)
                light.SetColor(1.0, 1.0, 1.0)
                self.renderer.AddLight(light)
                self.lights.append(light)

            # Two lights from above and below
            top_light = vtk.vtkLight()
            top_light.SetLightTypeToSceneLight()
            top_light.SetPosition(0, 0, 4)
            top_light.SetFocalPoint(0, 0, 0)
            top_light.SetIntensity(0.18)
            top_light.SetColor(1.0, 1.0, 1.0)
            self.renderer.AddLight(top_light)
            self.lights.append(top_light)

            bottom_light = vtk.vtkLight()
            bottom_light.SetLightTypeToSceneLight()
            bottom_light.SetPosition(0, 0, -4)
            bottom_light.SetFocalPoint(0, 0, 0)
            bottom_light.SetIntensity(0.12)
            bottom_light.SetColor(1.0, 1.0, 1.0)
            self.renderer.AddLight(bottom_light)
            self.lights.append(bottom_light)

            # Persistent soft headlight to avoid black scenes
            headlight = vtk.vtkLight()
            headlight.SetLightTypeToHeadlight()
            headlight.SetIntensity(0.18)  # slightly stronger to test
            headlight.SetColor(1.0, 1.0, 1.0)
            self.renderer.AddLight(headlight)
            self.lights.append(headlight)


        elif preset == "studio":
            for angle in [45, 135, 225, 315]:
                rad = math.radians(angle)
                light = vtk.vtkLight()
                light.SetPosition(2 * math.cos(rad), 2 * math.sin(rad), 1)
                light.SetFocalPoint(0, 0, 0)
                light.SetColor(1.0, 1.0, 0.9)  # Slightly warm white
                self.renderer.AddLight(light)
                self.lights.append(light)

            # Persistent soft headlight to avoid black scenes
            headlight = vtk.vtkLight()
            headlight.SetLightTypeToHeadlight()
            headlight.SetIntensity(0.18)  # slightly stronger to test
            headlight.SetColor(1.0, 1.0, 1.0)
            self.renderer.AddLight(headlight)
            self.lights.append(headlight)

        elif preset == "warm":
            # New warm lighting option
            for angle in [30, 150, 210, 330]:
                rad = math.radians(angle)
                light = vtk.vtkLight()
                light.SetPosition(2.5 * math.cos(rad), 2.5 * math.sin(rad), 1.5)
                light.SetFocalPoint(0, 0, 0)
                light.SetIntensity(0.22)
                light.SetColor(1.0, 0.85, 0.7)  # Warm tone
                self.renderer.AddLight(light)
                self.lights.append(light)

            # Persistent soft headlight to avoid black scenes
            headlight = vtk.vtkLight()
            headlight.SetLightTypeToHeadlight()
            headlight.SetIntensity(0.18)  # slightly stronger to test
            headlight.SetColor(1.0, 1.0, 1.0)
            self.renderer.AddLight(headlight)
            self.lights.append(headlight)

        elif preset == "off":
            self.lights = []
            # No lights added

        print(f"[Lights] Active lights: {len(self.lights)} (preset: {preset})")

        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Lighting preset: {preset}")


    ############### FILE LOAD/SAVE METHODS #################
    def open_file(self):
        '''Open a file dialog to select and load a 3D model file.'''
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Model File",
            "",
            "3D Model Files (*.stl *.obj *.ply *.las *.e57)"
        )
        if filename:
            self.load_model(filename)


    ###############  WIREFRAME METHODS #################

    def toggle_wireframe(self):
        '''Toggle between wireframe and solid rendering modes.'''
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            current = actor.GetProperty().GetRepresentation()
            if current == vtk.VTK_SURFACE:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()
        self.vtk_widget.GetRenderWindow().Render()


    ###############  DEBUG METHODS #################

    def debug_unlit_white(self):
        '''Apply unlit white material to all actors.'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            p = a.GetProperty()
            p.LightingOff()
            p.SetColor(1, 1, 1)
        self.vtk_widget.GetRenderWindow().Render()
        print("[DBG] Unlit white applied")

    def debug_restore_lit(self):
        '''Restore lit materials to all actors.'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            mp = a.GetMapper()
            if mp:
                mp.ScalarVisibilityOff()
            p = a.GetProperty()
            p.LightingOn()
            p.SetAmbient(0.35); p.SetDiffuse(0.65)
            p.SetSpecular(0.2); p.SetSpecularPower(20)
            p.SetColor(1, 1, 1)
        self.apply_lighting_preset(self.current_lighting_preset)
        print("[DBG] Lit materials restored")

    def debug_add_cube(self):
        '''Add a test cube to the scene.'''
        cube = vtk.vtkCubeSource(); cube.SetXLength(1); cube.SetYLength(1); cube.SetZLength(1)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.7, 0.2)
        self.renderer.AddActor(actor)
        self.reset_view()
        print("[DBG] Test cube added")

    def toggle_depth_peeling(self):
        '''Toggle depth peeling for transparent objects.'''
        rw = self.vtk_widget.GetRenderWindow()
        using = bool(self.renderer.GetUseDepthPeeling())
        self.renderer.SetUseDepthPeeling(0 if using else 1)
        rw.SetAlphaBitPlanes(1 if not using else 0)
        rw.SetMultiSamples(0 if not using else 8)
        self.vtk_widget.GetRenderWindow().Render()
        print(f"[DBG] Depth peeling: {'ON' if not using else 'OFF'}")

    ###############  LOADING METHODS #################

    def _find_mtl_and_texture_dir(self, obj_path: str):
        """Parse OBJ for mtllib and choose a sensible texture directory."""
        obj_dir = os.path.dirname(obj_path)
        mtl_path = None

        # Parse mtllib from OBJ (use first that exists)
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith("mtllib"):
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            candidate = parts[1].strip()
                            # Resolve relative to OBJ directory
                            cand_path = os.path.normpath(os.path.join(obj_dir, candidate))
                            if os.path.isfile(cand_path):
                                mtl_path = cand_path
                                break
        except Exception as e:
            print(f"Warning: failed to read OBJ for mtllib: {e}")

        # Fallback to base.mtl if no mtllib found
        if mtl_path is None:
            base = os.path.splitext(os.path.basename(obj_path))[0]
            guess = os.path.join(obj_dir, base + ".mtl")
            if os.path.isfile(guess):
                mtl_path = guess

        # Choose texture path: prefer alongside MTL, else OBJ dir, else textures/ subdir
        texture_dir = os.path.dirname(mtl_path) if mtl_path else obj_dir
        textures_sub = os.path.join(texture_dir, "textures")
        if os.path.isdir(textures_sub):
            texture_dir = textures_sub

        return mtl_path, texture_dir


    def _rewrite_obj_mtllib(self, obj_path: str, patched_mtl_path: str) -> str:
        """Write a temporary OBJ that points to the patched MTL (single mtllib)."""
        import tempfile
        lines = []
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Drop existing mtllib lines
                    if line.strip().lower().startswith("mtllib"):
                        continue
                    lines.append(line)
        except Exception as e:
            print(f"OBJ read failed: {e}")
        # Prepend our mtllib
        mtllib_line = f"mtllib {os.path.basename(patched_mtl_path)}\n"
        lines.insert(0, mtllib_line)

        tmp = tempfile.NamedTemporaryFile(prefix="patched_", suffix=".obj", delete=False)
        tmp.write("".join(lines).encode("utf-8"))
        tmp.flush(); tmp.close()
        return tmp.name

    def _scan_obj_materials(self, obj_path: str):
        used = []
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.lower().startswith("usemtl"):
                        parts = s.split(None, 1)
                        if len(parts) == 2:
                            used.append(parts[1].strip())
        except Exception as e:
            print(f"OBJ scan failed: {e}")
        return used

    def _parse_mtl(self, mtl_path: str):
        defined = set()
        maps = {}
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                cur = None
                for line in f:
                    s = line.strip()
                    sl = s.lower()
                    if sl.startswith("newmtl "):
                        cur = s.split(None, 1)[1].strip()
                        defined.add(cur)
                    elif cur and sl.startswith("map_kd"):
                        parts = s.split(None, 1)
                        if len(parts) == 2:
                            maps[cur] = parts[1].strip()
        except Exception as e:
            print(f"MTL parse failed: {e}")
        return defined, maps

    def _resolve_texture_file(self, tex_name: str, texture_dir: str):
        """Resolve a texture by trying:
           - exact path under texture_dir,
           - basename in texture_dir,
           - alternate extensions,
           - textures/ subfolder,
           - recursive search with fuzzy match on basename stem.
           Returns a path RELATIVE to texture_dir if found, else None.
        """
        import re

        def norm_stem(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            return "".join(ch.lower() for ch in stem if ch.isalnum())

        def stem_tokens(p):
            s = os.path.splitext(os.path.basename(p))[0].lower()
            return [t for t in re.split(r"[^a-z0-9]+", s) if len(t) >= 3]

        abs_dir = os.path.abspath(texture_dir)
        raw = tex_name.strip().strip('"').strip("'")
        base = os.path.basename(raw)
        stem, ext = os.path.splitext(base)

        # Candidate direct paths
        candidates = [
            os.path.join(abs_dir, raw),
            os.path.join(abs_dir, base),
        ]
        # Broaden supported extensions
        alt_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".tga",
                    ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF", ".TGA"]
        for alt in ([ext] if ext else []) + alt_exts:
            if alt:
                candidates.append(os.path.join(abs_dir, stem + alt))

        # Try textures/ subfolder
        tex_sub = os.path.join(abs_dir, "textures")
        for alt in ([ext] if ext else []) + alt_exts:
            if alt:
                candidates.append(os.path.join(tex_sub, stem + alt))

        for c in candidates:
            if os.path.isfile(c):
                return os.path.relpath(c, abs_dir)

        # Recursive search: exact match on filename or stem, else fuzzy token match
        target_base = base.lower()
        target_norm = norm_stem(base)
        target_tokens = set(stem_tokens(base))

        best_match = None
        best_score = -1
        for root, _, files in os.walk(abs_dir):
            for f in files:
                fl = f.lower()
                fstem_norm = norm_stem(f)
                if fl == target_base or fstem_norm == target_norm:
                    return os.path.relpath(os.path.join(root, f), abs_dir)
                # Fuzzy: share any meaningful token (e.g., 'wall', 'roof')
                ftokens = set(stem_tokens(f))
                score = len(target_tokens & ftokens)
                if score > best_score and score > 0:
                    best_score = score
                    best_match = os.path.join(root, f)

        if best_match:
            return os.path.relpath(best_match, abs_dir)
        return None


    def _rewrite_mtl(self, obj_path: str, mtl_path: str, texture_dir: str) -> str:
        """Patch MTL: fix map_Kd paths and add stubs for all used-but-undefined materials."""
        used = self._scan_obj_materials(obj_path)
        defined, _ = self._parse_mtl(mtl_path) if mtl_path else (set(), {})
        out_lines = []
        if mtl_path and os.path.isfile(mtl_path):
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                out_lines = f.readlines()

        # Fix map_Kd paths (try alternate extensions/locations)
        replaced = 0
        missing_list = []
        for i, line in enumerate(out_lines):
            s = line.strip()
            if s.lower().startswith("map_kd"):
                parts = s.split(None, 1)
                tex_raw = parts[1].strip() if len(parts) == 2 else ""
                resolved = self._resolve_texture_file(tex_raw, texture_dir)
                if resolved:
                    out_lines[i] = f"map_Kd {resolved}\n"
                    replaced += 1
                else:
                    # Use fallback texture if available
                    fallback = self._resolve_texture_file(DEFAULT_TEXTURE, texture_dir)
                    if fallback:
                        out_lines[i] = f"map_Kd {fallback}\n"
                    else:
                        out_lines[i] = f"# map_Kd {tex_raw}  # missing\n"
                    missing_list.append(tex_raw)
        if missing_list:
            self.statusBar().showMessage(
                f"Missing textures: {', '.join(missing_list)}"
            )
            print(f"Textures not found for map_Kd in {texture_dir}: {missing_list}")
        if replaced:
            self.statusBar().showMessage(
                f"Patched {replaced} map_Kd path(s) in MTL."
            )
            print(f"Patched {replaced} map_Kd path(s) in MTL.")

        # Add stubs for used-but-undefined materials
        missing = [m for m in used if m not in defined]
        if missing:
            out_lines.append("\n# --- Auto-added stub materials ---\n")
            for name in missing:
                out_lines.append(f"newmtl {name}\nKd 0.8 0.8 0.8\nKa 0 0 0\nKs 0 0 0\nNs 0\nillum 1\n")

        tmp = tempfile.NamedTemporaryFile(prefix="patched_", suffix=".mtl", delete=False)
        tmp.write("".join(out_lines).encode("utf-8"))
        tmp.flush(); tmp.close()
        return tmp.name


    def load_obj_with_textures(self, obj_path: str):
        '''Load an OBJ with MTL/PNG/JPG textures and add to the existing renderer.'''
        ren_win = self.vtk_widget.GetRenderWindow()

        # Track existing renderers and actors on our main renderer
        before = []
        rens = ren_win.GetRenderers(); rens.InitTraversal()
        for _ in range(rens.GetNumberOfItems()):
            before.append(rens.GetNextItem())
        # Track actors already on our renderer (importer may reuse it)
        def iter_actors(renderer):
            acts = renderer.GetActors(); acts.InitTraversal()
            for _ in range(acts.GetNumberOfItems()):
                yield acts.GetNextActor()
        actors_before_ids = {id(a) for a in iter_actors(self.renderer)}

        mtl_path, texture_dir = self._find_mtl_and_texture_dir(obj_path)
        patched_mtl = self._rewrite_mtl(obj_path, mtl_path, texture_dir)
        patched_obj = self._rewrite_obj_mtllib(obj_path, patched_mtl)

        print(f"[OBJ] Texture dir: {os.path.abspath(texture_dir)}")

        importer = vtk.vtkOBJImporter()
        importer.SetRenderWindow(ren_win)
        importer.SetFileName(patched_obj)          # force use of our patched OBJ
        importer.SetFileNameMTL(patched_mtl)       # and our patched MTL
        importer.SetTexturePath(os.path.abspath(texture_dir))

        # Debug: renderer counts before import
        rens = ren_win.GetRenderers(); rens.InitTraversal()
        before_count = rens.GetNumberOfItems()
        print(f"[OBJ] Renderers before import: {before_count}")

        try:
            importer.Update()
        except Exception as e:
            self.statusBar().showMessage(f"Failed to import OBJ: {e}")
            return

        # Debug: renderer counts after import
        rens = ren_win.GetRenderers(); rens.InitTraversal()
        after_count = rens.GetNumberOfItems()
        print(f"[OBJ] Renderers after import: {after_count}")

        # Move imported props/lights to our renderer
        after = []
        rens = ren_win.GetRenderers(); rens.InitTraversal()
        for _ in range(rens.GetNumberOfItems()):
            after.append(rens.GetNextItem())
        new_renderers = [r for r in after if r not in before and r is not self.renderer]

        # Helper: configure an actor for proper shading/materials
        def configure_actor(a: vtk.vtkActor):
            mapper = a.GetMapper()
            pd = None
            if mapper is not None:
                in_conn = mapper.GetInputConnection(0, 0)
                normals = vtk.vtkPolyDataNormals()
                if in_conn is not None:
                    normals.SetInputConnection(in_conn)
                else:
                    polydata = mapper.GetInput()
                    if isinstance(polydata, vtk.vtkPolyData):
                        normals.SetInputData(polydata)
                        pd = polydata
                    else:
                        normals = None
                if normals is not None:
                    normals.SplittingOff()
                    normals.ConsistencyOn()
                    normals.AutoOrientNormalsOn()
                    mapper.SetInputConnection(normals.GetOutputPort())
                mapper.ScalarVisibilityOff()
                # Try to get polydata after hooking normals
                if pd is None:
                    try:
                        alg = mapper.GetInputAlgorithm()
                        if alg:
                            alg.Update()
                            pd = mapper.GetInput()
                    except Exception:
                        pass

            # Handle textures: detach if invalid image or no UVs
            tex = a.GetTexture()
            tex_valid = False
            has_tcoords = False
            if isinstance(pd, vtk.vtkPolyData):
                tcoords = pd.GetPointData().GetTCoords()
                has_tcoords = bool(tcoords) and int(tcoords.GetNumberOfTuples() or 0) > 0
            if tex is not None:
                img = tex.GetInput()
                if img is not None:
                    dims = img.GetDimensions()
                    tex_valid = (dims[0] > 0 and dims[1] > 0)
                if not tex_valid or not has_tcoords:
                    # Detach bad texture so Kd/material shading works
                    a.SetTexture(None)
                    print(f"[DBG] Dropped texture: valid={tex_valid}, tcoords={has_tcoords}")

            if a.GetTexture() is not None:
                a.GetTexture().InterpolateOn()

            p = a.GetProperty()
            p.SetLighting(True)
            p.BackfaceCullingOff()
            p.FrontfaceCullingOff()
            p.SetAmbient(0.35); p.SetDiffuse(0.65)
            p.SetSpecular(0.2); p.SetSpecularPower(20)
            p.SetColor(1.0, 1.0, 1.0)  # ensure not multiplied by black
            if hasattr(p, "SetInterpolationToPhong"):
                p.SetInterpolationToPhong()

        moved_actors = 0
        # Case 1: importer created separate renderers; move and configure actors
        for r in new_renderers:
            actors = r.GetActors(); actors.InitTraversal()
            for _ in range(actors.GetNumberOfItems()):
                a = actors.GetNextActor()
                configure_actor(a)
                self.renderer.AddActor(a)
                moved_actors += 1
        # Case 2: importer injected actors directly into our renderer; configure the new ones
        newly_added = [a for a in iter_actors(self.renderer) if id(a) not in actors_before_ids]
        for a in newly_added:
            configure_actor(a)
        if newly_added:
            print(f"[OBJ] Actors added to main renderer by importer: {len(newly_added)}")

        print(f"[OBJ] Moved actors from importer: {moved_actors}")

        # Debug: dump a few actors
        dump = 0
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(min(5, acts.GetNumberOfItems())):
            a = acts.GetNextActor()
            mp = a.GetMapper()
            pd = mp.GetInput() if mp else None
            if pd is None and mp and mp.GetInputConnection(0, 0):
                try:
                    alg = mp.GetInputAlgorithm()
                    if alg:
                        alg.Update()
                        pd = mp.GetInput()
                except Exception:
                    pass
            bounds = a.GetBounds() if a else ()
            print(f"[DBG] Actor: tex={a.GetTexture() is not None}, "
                  f"pts={(pd.GetNumberOfPoints() if isinstance(pd, vtk.vtkPolyData) else 'n/a')}, "
                  f"bounds={bounds}")

        # Remove importer-created renderers so they don’t overdraw your scene
        for r in new_renderers:
            try:
                r.RemoveAllViewProps()
            except Exception:
                pass
            self.vtk_widget.GetRenderWindow().RemoveRenderer(r)

        rens = ren_win.GetRenderers(); rens.InitTraversal()
        final_count = rens.GetNumberOfItems()
        print(f"[OBJ] Renderers after cleanup: {final_count}")

        # Cleanup temp files
        for tmpf in [patched_obj, patched_mtl]:
            try:
                if tmpf and os.path.exists(tmpf):
                    os.remove(tmpf)
            except Exception:
                pass

        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        ren_win.Render()
        self.statusBar().showMessage("OBJ loaded with patched materials.")

    ######################## LOAD MODEL METHOD ###########################
    def load_model(self, filename):
        '''Load a 3D model from the specified file.'''
        # Clear current scene
        self.renderer.RemoveAllViewProps()
        ext = os.path.splitext(filename)[1].lower()

        try:

            if ext == ".obj":
                # Use VTK OBJ importer to load MTL/PNG textures correctly
                self.load_obj_with_textures(filename)

            elif ext in [".stl", ".ply"]:
                mesh = trimesh.load(filename, force='mesh')
                polydata = self.trimesh_to_vtk(mesh)

                # Compute normals for better lighting/shading
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(polydata)
                normals.SplittingOff()
                normals.ConsistencyOn()
                normals.AutoOrientNormalsOn()
                normals.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                self.renderer.AddActor(actor)

                self.reset_view()


            elif ext == ".las":
                self.load_point_cloud(filename, load_las_as_vtk)

            elif ext == ".e57":
                self.load_point_cloud(filename, load_e57_as_vtk)

            else:
                self.statusBar().showMessage(f"Unsupported file format: {ext}")
                print(f"Unsupported file format: {ext}")
                return

            # Apply current lighting and finalize view
            self.apply_lighting_preset(self.current_lighting_preset)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)}")
            self.renderer.ResetCameraClippingRange()
            self.update_model_info()
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {e}")
            print(f"Error loading model: {e}")

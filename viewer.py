from PyQt5.QtWidgets import QMainWindow, QFileDialog, QAction # Basic PyQt5 imports
# Additional PyQt5 widgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QColorDialog, QSpinBox
from PyQt5.QtCore import Qt # For Qt constants
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.qt # Ensure VTK Qt support is loaded
vtkmodules.qt.QVTKRWIBase = 'QGLWidget' # Ensure using QGLWidget
import vtk # VTK core
import os # For file path operations
import math # For mathematical operations
import laspy # For LAS loading
import numpy as np # For numerical operations
import pye57 # For E57 loading
import trimesh # For alternative OBJ loading
import tempfile  # For creating temporary files


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
    ''' Load LAS file and convert to VTK PolyData '''
    las = laspy.read(filename)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Optional: color support if available
    has_color = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
    colors = None
    if has_color:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = (colors / 65535.0 * 255).astype(np.uint8)  # Normalize to 0–255

    # Convert to VTK points
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    if has_color:
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for c in colors:
            vtk_colors.InsertNextTuple3(*c)
        polydata.GetPointData().SetScalars(vtk_colors)

    return polydata


def load_e57_as_vtk(filename):
    ''' Load E57 file and convert to VTK PolyData '''
    e57 = pye57.E57(filename)
    scan = e57.read_scan(0)  # Read the first scan
    points = np.vstack((scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"])).transpose()

    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Optional: color support if available
    if "colorRed" in scan and "colorGreen" in scan and "colorBlue" in scan:
        colors = np.vstack((scan["colorRed"], scan["colorGreen"], scan["colorBlue"])).transpose()
        colors = colors.astype(np.uint8)
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for c in colors:
            vtk_colors.InsertNextTuple3(*c)
        polydata.GetPointData().SetScalars(vtk_colors)

    return polydata


class CustomQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    """Custom QVTKRenderWindowInteractor to forward key and mouse events to parent."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def keyPressEvent(self, event):
        """Handle arrow key presses to rotate the model."""
        if self.parent.dialog_active:
            return  # Ignore key events if a dialog is active

        camera = self.parent.renderer.GetActiveCamera()

        # Rotation step size (in degrees)
        rotation_step = 5

        if event.key() == Qt.Key_Left:
            camera.Azimuth(rotation_step)
        elif event.key() == Qt.Key_Right:
            camera.Azimuth(-rotation_step)
        elif event.key() == Qt.Key_Up:
            camera.Elevation(-rotation_step)
        elif event.key() == Qt.Key_Down:
            camera.Elevation(rotation_step)

        # Update the camera view
        camera.OrthogonalizeViewUp()
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
        self.current_lighting_preset = "default"
        self.measure_points = [] # Store measurement points
        self.annotations = []  # Store annotation actors
        # VTK widget
        self.vtk_widget = CustomQVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)
        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_widget.setFocus() # Ensure VTK widget has focus for key events
        # Enable depth peeling
        self.enable_depth_peeling()
        # Menu
        self.init_menu()
        # Load default model
        default_model = "assets/Tiger1MidHullThickMachineGun.stl"
        if os.path.exists(default_model):
            self.load_model(default_model)

        self.interactor.Initialize()

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
        reset_view_action.setShortcut("R")  # Optional shortcut
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

        ########### WIREFRAME MENU ##############
        wireframe_menu = menubar.addMenu("Wireframe")
        toggle_wireframe = QAction("Toggle Wireframe", self)
        toggle_wireframe.setShortcut("W")  # Optional shortcut
        toggle_wireframe.triggered.connect(self.toggle_wireframe)
        wireframe_menu.addAction(toggle_wireframe)

        ########### MEASURE MENU ##############
        measure_menu = menubar.addMenu("Measure")
        # MEASURE DISTANCE ACTION
        measure_action = QAction("Measure Distance", self)
        measure_action.triggered.connect(self.activate_measure_mode)
        measure_menu.addAction(measure_action)
        # CLEAR LAST MEASUREMENT ACTION
        clear_last_measure_action = QAction("Clear Last Measurement", self)
        clear_last_measure_action.triggered.connect(self.clear_last_measurement)
        measure_menu.addAction(clear_last_measure_action)
        #  CLEAR All MEASUREMENTS ACTION
        clear_measure_action = QAction("Clear All Measurements", self)
        clear_measure_action.triggered.connect(self.clear_all_measurements)
        measure_menu.addAction(clear_measure_action)

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
        # default lighting
        lighting_default = QAction("Lighting: Default", self)
        lighting_default.triggered.connect(lambda: self.apply_lighting_preset("default"))
        lighting_menu.addAction(lighting_default)
        # studio lighting
        lighting_studio = QAction("Lighting: Studio", self)
        lighting_studio.triggered.connect(lambda: self.apply_lighting_preset("studio"))
        lighting_menu.addAction(lighting_studio)
        # lighting off
        lighting_off = QAction("Lighting: Off", self)
        lighting_off.triggered.connect(lambda: self.apply_lighting_preset("off"))
        lighting_menu.addAction(lighting_off)
        # light adjustment dialog
        adjust_lights = QAction("Custom Adjust Lights", self)
        adjust_lights.triggered.connect(self.open_light_dialog)
        lighting_menu.addAction(adjust_lights)

        ############# BACKGROUND COLOR MENU ################
        background_menu = menubar.addMenu("Background")
        bg_white = QAction("White", self)
        bg_white.triggered.connect(lambda: self.set_background_color(1, 1, 1))
        background_menu.addAction(bg_white)
        bg_light_gray = QAction("Light Gray", self)
        bg_light_gray.triggered.connect(lambda: self.set_background_color(0.9, 0.9, 0.9))
        background_menu.addAction(bg_light_gray)
        bg_dark_gray = QAction("Dark Gray", self)
        bg_dark_gray.triggered.connect(lambda: self.set_background_color(0.2, 0.2, 0.2))
        background_menu.addAction(bg_dark_gray)
        bg_black = QAction("Black", self)
        bg_black.triggered.connect(lambda: self.set_background_color(0, 0, 0))
        background_menu.addAction(bg_black)
        # Optionally, add a custom color picker
        bg_custom = QAction("Custom...", self)
        bg_custom.triggered.connect(self.pick_custom_background)
        background_menu.addAction(bg_custom)

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

    def reset_view(self):
        """Reset the camera to its default position."""
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor_count = actors.GetNumberOfItems()
        print(f"Number of actors in the scene: {actor_count}")

        if actor_count == 0:
            self.statusBar().showMessage("No actors in the scene to reset view.")
            return

        bounds = self.renderer.ComputeVisiblePropBounds()
        print(f"Scene bounds: {bounds}")

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

    def activate_measure_mode(self):
        '''Activate measurement mode to measure distances.'''
        self.deactivate_all_modes()  # Ensure no other mode is active

        # Remove previous observers if they exist
        if hasattr(self, "left_click_observer"):
            self.vtk_widget.RemoveObserver(self.left_click_observer)
            del self.left_click_observer
        if hasattr(self, "right_click_observer"):
            self.vtk_widget.RemoveObserver(self.right_click_observer)
            del self.right_click_observer

        # Add new observers
        self.measure_points = []
        self.measure_lines = []
        self.measure_markers = []
        self.statusBar().showMessage("Click points to measure. Right-click to finish.")
        self.left_click_observer = self.vtk_widget.AddObserver("LeftButtonPressEvent", self.on_measure_click)
        self.right_click_observer = self.vtk_widget.AddObserver("RightButtonPressEvent", self.finish_measure)

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
                self.vtk_widget.GetRenderWindow().Render()
        
    def finish_measure(self, obj, event):
        '''Finish measurement mode.'''
        if len(self.measure_points) > 1:
            total = self.measure_total()
            self.statusBar().showMessage(f"Measurement finished. Total path length: {total:.3f}")
        else:
            self.statusBar().showMessage("Measurement cancelled or not enough points.")
        # Remove observers
        self.vtk_widget.RemoveObserver(self.left_click_observer)
        self.vtk_widget.RemoveObserver(self.right_click_observer)
        # Optionally, clear measurement after a delay or on next activation

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
            for actor in self.measure_markers:
                self.renderer.RemoveActor(actor)
            self.measure_markers.clear()
        # Remove line actors
        if hasattr(self, "measure_lines"):
            for actor in self.measure_lines:
                self.renderer.RemoveActor(actor)
            self.measure_lines.clear()
        # Clear points
        if hasattr(self, "measure_points"):
            self.measure_points.clear()
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
        # Remove last point
        if hasattr(self, "measure_points") and self.measure_points:
            self.measure_points.pop()
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage("Last measurement cleared.")

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
            from PyQt5.QtWidgets import QInputDialog
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

    def keyPressEvent(self, event):
        """Handle arrow key presses to rotate the model."""
        camera = self.renderer.GetActiveCamera()

        # Rotation step size (in degrees)
        rotation_step = 5

        if event.key() == Qt.Key_Left:
            # Rotate left (around the Y-axis)
            camera.Azimuth(rotation_step)
        elif event.key() == Qt.Key_Right:
            # Rotate right (around the Y-axis)
            camera.Azimuth(-rotation_step)
        elif event.key() == Qt.Key_Up:
            # Rotate up (around the X-axis)
            camera.Elevation(-rotation_step)
        elif event.key() == Qt.Key_Down:
            # Rotate down (around the X-axis)
            camera.Elevation(rotation_step)

        # Update the camera view
        camera.OrthogonalizeViewUp()
        self.vtk_widget.GetRenderWindow().Render()

    def set_background_color(self, r, g, b):
        '''Set the background color of the renderer.'''
        self.renderer.SetBackground(r, g, b)
        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Background color set.")

    def pick_custom_background(self):
        '''Open a color picker dialog to select a custom background color.'''
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgbF()[:3]
            self.set_background_color(*rgb)

    def open_light_dialog(self):
        '''Open a dialog to adjust light properties.'''
        if hasattr(self, "lights") and self.lights:
            dlg = LightAdjustDialog(self, self.lights)
            dlg.exec_()
        else:
            self.statusBar().showMessage("No lights to adjust in this preset.")
    
    ############### LIGHTING METHODS #################
    def cycle_lighting_preset(self):
        '''Cycle through predefined lighting presets.'''
        presets = ["default", "studio", "off"]
        current_index = presets.index(self.current_lighting_preset)
        next_index = (current_index + 1) % len(presets)
        next_preset = presets[next_index]
        self.apply_lighting_preset(next_preset)

    def apply_lighting_preset(self, preset="default"):
        '''Apply a predefined lighting preset.'''
        self.current_lighting_preset = preset
        self.renderer.RemoveAllLights()
        self.lights = []

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

            # Dim headlight for ambient fill
            headlight = vtk.vtkLight()
            headlight.SetLightTypeToHeadlight()
            headlight.SetIntensity(0.08)
            headlight.SetColor(1, 1, 1)
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

        elif preset == "off":
            self.lights = []
            # No lights added
            pass

        self.vtk_widget.GetRenderWindow().Render()
        self.statusBar().showMessage(f"Lighting preset: {preset}")

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
    

    ######################## LOAD MODEL METHOD ###########################
    def load_model(self, filename):
        '''Load a 3D model from the specified file.'''
        self.renderer.RemoveAllViewProps()
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in [".obj", ".stl", ".ply"]:
                # Load the model using trimesh
                mesh = trimesh.load(filename, force='mesh')

                # Convert to VTK
                polydata = self.trimesh_to_vtk(mesh)

                # Create VTK mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # Only check for textures if the file format supports them
                if ext == ".obj" and hasattr(mesh, "visual") and mesh.visual.kind == "texture":
                    texture_data = mesh.visual.material.image
                    try:
                        if isinstance(texture_data, str) and os.path.exists(texture_data):
                            # If texture_data is a file path, load it directly
                            texture = self.load_texture(texture_data)
                            actor.SetTexture(texture)
                        elif texture_data is not None:
                            # If texture_data is an in-memory image, save it to a temporary file
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                                texture_data.save(temp_file.name)  # Save the image to the temp file
                                texture = self.load_texture(temp_file.name)
                                actor.SetTexture(texture)
                        else:
                            print("Texture data is missing or invalid. Using default material.")
                    except Exception as e:
                        print(f"Error loading texture: {e}. Using default material.")
                elif ext == ".obj":
                    print("No texture data found. Using default material.")

                self.renderer.AddActor(actor)
                self.reset_view()
                self.renderer.SetBackground(0.9, 0.9, 0.9)  # Light gray

            elif ext == ".las":
                self.load_point_cloud(filename, load_las_as_vtk)

            elif ext == ".e57":
                self.load_point_cloud(filename, load_e57_as_vtk)

            else:
                print(f"Unsupported file format: {ext}")
                return

            self.apply_lighting_preset(self.current_lighting_preset)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)}")

            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {e}")
            print(f"Error loading model: {e}")

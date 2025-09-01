# 3D Model Viewer

A cross-platform desktop application for viewing 3D models and point clouds using PyQt5 and VTK.

## Features

- **Supported Formats:**  
  - Mesh: `.stl`, `.obj`, `.ply`
  - Point Cloud: `.las`, `.e57`
- **Material & Texture Support:**  
  - Loads OBJ materials (`.mtl`) and textures.
- **Lighting Presets:**  
  - Choose between Default, Studio, and Off.
  - Cycle presets with the `L` key.
  - **Interactive Light Adjustment:** Change light positions, colors, and intensities via a dialog.
- **Wireframe Toggle:**  
  - Toggle wireframe mode with the `W` key.
- **Camera Controls:**  
  - Automatic centering and zoom for loaded models.
- **Point Cloud Visualization:**  
  - LAS and E57 files display colored points if color data is present.
- **Outline Box:**  
  - Point clouds are shown with a bounding box for context.
- **Status Bar:**  
  - Displays current lighting preset and material info.


## 🏁 Prerequisites
Here's how to run your Python-based 3D model viewer project step by step:

Make sure you have the following installed:
• 	Python 3.8+
• 	pip (Python package manager)
• 	A code editor like VS Code or PyCharm

> **Note:** Large models and point clouds may require a modern CPU and dedicated GPU for smooth performance. Systems with limited RAM or integrated graphics may experience slow rendering or lag with very large files.


## 📦 Installation of required libraries
Open a terminal or command prompt and run:
```bash
pip install PyQt5 vtk laspy numpy pye57 trimesh
```
or
```bash
pip3 install PyQt5 vtk laspy numpy pye57 trimesh
```
or install each package individually:
```bash
pip3 install PyQt5
pip3 install vtk
pip3 install laspy
pip3 install numpy
pip3 install trimesh
pip3 install pye57
```

### Mac Users: Installing pye57
If you are on macOS, you may need to install the native library `xerces-c` before installing `pye57`:
```sh
brew install xerces-c
```
If installation fails with an error about `libxerces-c-3.2.dylib`, check for the file in `/usr/local/lib`. If it’s missing, create a symlink to the available version:
```sh
ln -s /usr/local/lib/libxerces-c-3.3.dylib /usr/local/lib/libxerces-c-3.2.dylib
```

---

This installs:
- PyQt5 for the GUI
- vtk for 3D rendering
- laspy, pye57, trimesh, numpy for file format handling


## 🧪 Usage
Place a compatible file in the `assets/` folder. You can get one from...
https://science.nasa.gov/3d-resources/
https://www.thingiverse.com/

1. **Run the main python file:**
    ```sh
    python main.py
    ```
    Run the desktop app from the terminal similar to below i.e
```bash
python main.py
```

- This will launch a desktop window with a default 3D model loaded
- Use the mouse to rotate, zoom, and pan the model
- If nothing shows up, make sure the file path is correct and the model isn't empty

2. **Open a model:**  
   Use `File > Open Model` to load a supported file.
3. **Lighting:**  
   Use `View > Lighting: Default/Studio/Off` or press `L` to cycle.
4. **Adjust Lights:**  
   Use `View > Adjust Lights` to interactively change light properties.
5. **Wireframe:**  
   Use `View > Toggle Wireframe` or press `W` to toggle wireframe mode.

## File Format Details
- **OBJ:** Loads associated `.mtl` and texture files if present.
- **PLY:** Supports mesh and point cloud data.
- **LAS/E57:** Displays colored point clouds if color info is available.

## Customization

- **Lighting:**  
  - Adjust via the dialog for position, color, and intensity.
- **Background:**  
  - Default is light gray; change in code if desired.

## Measuring distance
Select "Measure Distance" from the menu.
Click multiple points (left mouse button).
Each measured point will be marked with a green sphere.
Lines are drawn between points; segment and total length are shown in the status bar.
Right-click to finish and see the total path length.
Users can select "Clear Measurements" from the menu to remove all measurement markers and lines.

## Troubleshooting
🛠️ Troubleshooting Tips
- If you see errors about missing modules, ensure all dependencies are installed.
- For large point clouds, performance may vary depending on your hardware.
- If the window opens but no model appears, check the console for errors
- If PyQt5 or VTK fails to install, try upgrading pip:
```bash
pip install --upgrade pip
```

## 📁 Files
Inside 3d_model_viewer are...
- `main.py` — the entry point
- `viewer.py` — the viewer logic
- `assets/` — store your 3D models and textures here

## Credits

- [VTK](https://vtk.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [laspy](https://laspy.readthedocs.io/)
- [pye57](https://github.com/asmaloney/pye57)

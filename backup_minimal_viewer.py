"""Minimal 3D Model Viewer (PyQt5 + VTK)"""

import os
from typing import Optional

import vtk
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QAction, QFileDialog, QDockWidget,
    QPlainTextEdit, QActionGroup, QToolBar, QStyle
)

from PyQt5.QtGui import QKeySequence

import vtkmodules.vtkIOImage  # ensure PNG/JPG readers are registered

# Optional (guarded)
try:
    import trimesh
except Exception:
    trimesh = None

class ViewerApp(QMainWindow):
    '''Main application window for 3D model viewing.'''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal 3D Viewer")
        self.resize(1000, 700)
        self.unlit = False
        self._shared_texture = None
        self.transparency_mode = "opaque"  # opaque | cutout | blend
        # Transform applied to the whole model (all actors)
        self.model_transform = vtk.vtkTransform()
        # Per-actor material and texture cache
        self._actor_material = {}
        self._texture_cache = {}

        # Central VTK widget
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)

        # Renderer / window / interactor
        self.ren_win = self.vtk_widget.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()

        try:
            if hasattr(self.renderer, "UseSrgbColorSpaceOn"):
                self.renderer.UseSrgbColorSpaceOn()
        except:
            pass

        self.ren_win.AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_widget.Initialize()


         # Mouse styles
        self.mouse_styles = ["trackball", "terrain", "joystick"]
        self.mouse_style = "trackball"
        self._interactor_style = None
        self.set_mouse_style(self.mouse_style)

        try:
            self.ren_win.SetAlphaBitPlanes(1)
            self.ren_win.SetMultiSamples(0)
            if hasattr(self.renderer, "SetUseDepthPeeling"):
                self.renderer.SetUseDepthPeeling(False)  # enable later only for blend
        except:
            pass

        # Simple state
        self.bg_colors = [(1,1,1),(0.9,0.9,0.9),(0.2,0.2,0.2),(0,0,0)]
        self.bg_idx = 2
        self.brightness_mode = "bright"  # normal | bright
        self.current_file: Optional[str] = None

        # UI
        self._init_menu()
        self._init_toolbar()
        self._init_info_dock()

        # Lighting + initial background
        self.apply_lighting("headlight")
        self.set_background(*self.bg_colors[self.bg_idx])

        # Load default if exists
        default_obj = "assets/house.obj"
        if os.path.isfile(default_obj):
            self.load_model(default_obj)

        self._bind_debug_keys()


    def _init_toolbar(self):
        """Create a slim top toolbar with icon-only actions."""
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setAllowedAreas(Qt.TopToolBarArea)
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(Qt.TopToolBarArea, tb)

        st = self.style()
        # File actions
        act_open = QAction(st.standardIcon(QStyle.SP_DialogOpenButton), "Open", self)
        act_open.setToolTip("Open model")
        act_open.triggered.connect(self.open_file)
        tb.addAction(act_open)

        act_save = QAction(st.standardIcon(QStyle.SP_DialogSaveButton), "Save As", self)
        act_save.setToolTip("Save model as OBJ")
        act_save.triggered.connect(self.save_model_as)
        tb.addAction(act_save)

        tb.addSeparator()
        # View controls
        act_reset = QAction(st.standardIcon(QStyle.SP_BrowserReload), "Reset View", self)
        act_reset.setToolTip("Reset view")
        act_reset.triggered.connect(self.reset_view)
        tb.addAction(act_reset)

        act_wire = QAction(st.standardIcon(QStyle.SP_TitleBarShadeButton), "Wireframe", self)
        act_wire.setToolTip("Toggle wireframe")
        act_wire.triggered.connect(self.toggle_wireframe)
        tb.addAction(act_wire)

        act_bg = QAction(st.standardIcon(QStyle.SP_DesktopIcon), "Background", self)
        act_bg.setToolTip("Cycle background")
        act_bg.triggered.connect(self.cycle_background)
        tb.addAction(act_bg)

        tb.addSeparator()
        # Zoom controls
        act_zoomin = QAction(st.standardIcon(QStyle.SP_ArrowUp), "Zoom In", self)
        act_zoomin.setToolTip("Zoom in")
        act_zoomin.triggered.connect(lambda: self.zoom(+1, fast=False))
        tb.addAction(act_zoomin)

        act_zoomout = QAction(st.standardIcon(QStyle.SP_ArrowDown), "Zoom Out", self)
        act_zoomout.setToolTip("Zoom out")
        act_zoomout.triggered.connect(lambda: self.zoom(-1, fast=False))
        tb.addAction(act_zoomout)

        tb.addSeparator()
        # Unlit toggle
        self.act_unlit.setIcon(st.standardIcon(QStyle.SP_DialogNoButton))
        self.act_unlit.setText("Unlit")
        self.act_unlit.setToolTip("Toggle unlit (fullbright) textures")
        tb.addAction(self.act_unlit)


    def dump_actor_textures(self):
        ''' Debug: print info about actors, textures, properties'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        i=0
        while i < acts.GetNumberOfItems():
            a = acts.GetNextActor()
            if a:
                m = a.GetMapper()
                tex = a.GetTexture()
                p = a.GetProperty()
                sv = m.GetScalarVisibility() if m else None
                col = p.GetColor()
                print(f"[ACT]{i} tex={bool(tex)} scalarVis={sv} color={tuple(round(c,3) for c in col)} amb={round(p.GetAmbient(),3)} diff={round(p.GetDiffuse(),3)} lightOn={p.GetLighting() if hasattr(p,'GetLighting') else 'n/a'}")
            i+=1


    def set_transparency_mode(self, mode: str):
        '''Set transparency mode: "opaque", "cutout", "blend".'''
        if mode not in ("opaque","cutout","blend"):
            return
        if mode == self.transparency_mode:
            return
        self.transparency_mode = mode
        print(f"[ALPHA] mode -> {mode}")
        try:
            if hasattr(self.renderer, "SetUseDepthPeeling"):
                self.renderer.SetUseDepthPeeling(mode == "blend")
        except:
            pass
        
        # Invalidate caches and reapply textures
        self._shared_texture = None
        self._texture_cache.clear()
        if self.current_file and self.current_file.lower().endswith(".obj"):
            mtl_path = os.path.splitext(self.current_file)[0] + ".mtl"
            if not self._apply_textures_from_mtl(self.current_file, mtl_path):
                self._apply_manual_atlas(self.current_file, mtl_path)
        self.ren_win.Render()


    def diagnose_textures(self):
        ''' Debug: check actors for UVs, texture formats, alpha channel'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        i = 0
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            m = a.GetMapper()
            pd = m.GetInput() if m else None
            tcoords_ok = False
            tcoord_range = None
            if pd and pd.GetPointData():
                tc = pd.GetPointData().GetTCoords()
                if tc:
                    tcoords_ok = True
                    # sample first 3 tuples
                    n = min(3, tc.GetNumberOfTuples())
                    samples = []
                    for k in range(n):
                        u,v = tc.GetTuple(k)[:2]
                        samples.append((round(u,4), round(v,4)))
                    tcoord_range = samples
            tex = a.GetTexture()
            dims = None
            comps = None
            alpha_channel_nonzero = None
            if tex and tex.GetInput():
                img = tex.GetInput()
                dims = img.GetDimensions()
                comps = img.GetNumberOfScalarComponents()
                if comps in (2,4):
                    scalars = img.GetPointData().GetScalars()
                    if scalars:
                        # crude check: sample middle pixel alpha
                        midx = dims[0]//2; midy = dims[1]//2
                        # GetArray pointer: safer to use GetTuple
                        alpha_val = scalars.GetTuple((midy * dims[0] + midx))[comps-1]
                        alpha_channel_nonzero = alpha_val > 0
                elif comps == 1:  # maybe we only have alpha
                    alpha_channel_nonzero = True  # treat as mask
            p = a.GetProperty()
            print(f"[DIAG] A{i} UVs={tcoords_ok} UV_samples={tcoord_range} tex_dims={dims} comps={comps} alphaNonZero={alpha_channel_nonzero} amb={round(p.GetAmbient(),2)} diff={round(p.GetDiffuse(),2)} color={tuple(round(c,2) for c in p.GetColor())}")
            i += 1
            if i >= 5: break  # limit output


    def force_fullbright(self):
        ''' Debug: force all actors to fullbright (unlit)'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            m = a.GetMapper()
            if m: m.ScalarVisibilityOff()
            tex = a.GetTexture()
            p = a.GetProperty()
            p.SetColor(1,1,1)
            p.SetAmbient(1.0)
            p.SetDiffuse(0.0)
            p.SetSpecular(0.0)
            if hasattr(p,"LightingOff"): p.LightingOff()
            if tex:
                try:
                    if hasattr(vtk.vtkTexture,"VTK_TEXTURE_BLENDING_MODE_REPLACE") and hasattr(tex,"SetBlendingMode"):
                        tex.SetBlendingMode(vtk.vtkTexture.VTK_TEXTURE_BLENDING_MODE_REPLACE)
                except: pass
        self.ren_win.Render()
        print("[FIX] Forced fullbright applied.")

    def reapply_texture_fix(self):
        ''' Debug: reapply texture fix (remove textures from actors lacking UVs)'''
        # Call after load if still dark
        self.force_fullbright()


    # ---------- UI ----------
    def _init_menu(self):
        '''Initialize menu bar and actions.'''
        mb = self.menuBar()

        fmenu = mb.addMenu("File")
        act_open = QAction("Open Model", self)
        act_open.triggered.connect(self.open_file)
        fmenu.addAction(act_open)

        act_save = QAction("Save Model As...", self)
        act_save.setShortcut(QKeySequence.SaveAs)
        act_save.triggered.connect(self.save_model_as)
        fmenu.addAction(act_save)

        vmenu = mb.addMenu("View")
        act_reset = QAction("Reset View", self)
        act_reset.triggered.connect(self.reset_view)
        vmenu.addAction(act_reset)

        self.act_unlit = QAction("Unlit Textures", self)
        self.act_unlit.setCheckable(True)
        self.act_unlit.setChecked(self.unlit)
        self.act_unlit.triggered.connect(self.toggle_unlit)
        vmenu.addAction(self.act_unlit)

        act_bg = QAction("Cycle Background (B)", self)
        act_bg.setShortcut("B")
        act_bg.triggered.connect(self.cycle_background)
        vmenu.addAction(act_bg)

        act_wire = QAction("Toggle Wireframe (W)", self)
        act_wire.setShortcut("W")
        act_wire.triggered.connect(self.toggle_wireframe)
        vmenu.addAction(act_wire)

        # Mouse menu
        mmenu = mb.addMenu("Mouse")
        mgroup = QActionGroup(self); mgroup.setExclusive(True)
        for name, label in (("trackball","Trackball"), ("terrain","Terrain"), ("joystick","Joystick")):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(self.mouse_style == name)
            act.triggered.connect(lambda checked, n=name: checked and self.set_mouse_style(n))
            mgroup.addAction(act)
            mmenu.addAction(act)

        # Transparency menu
        tmenu = mb.addMenu("Transparency")
        group = QActionGroup(self); group.setExclusive(True)
        for mode in ("opaque","cutout","blend"):
            act = QAction(mode.capitalize(), self)
            act.setCheckable(True)
            act.setData(mode)
            if mode == self.transparency_mode:
                act.setChecked(True)
            act.triggered.connect(lambda checked, m=mode: checked and self.set_transparency_mode(m))
            group.addAction(act)
            tmenu.addAction(act)

        # Brightness menu
        bmenu = mb.addMenu("Brightness")
        act_norm = QAction("Normal", self)
        act_norm.setCheckable(True)
        act_bright = QAction("Bright", self)
        act_bright.setCheckable(True)
        act_norm.setChecked(self.brightness_mode == "normal")
        act_bright.setChecked(self.brightness_mode == "bright")
        def set_brightness(mode):
            def fn():
                self.brightness_mode = mode
                act_norm.setChecked(mode=="normal")
                act_bright.setChecked(mode=="bright")
                self.apply_brightness()
            return fn
        act_norm.triggered.connect(set_brightness("normal"))
        act_bright.triggered.connect(set_brightness("bright"))
        bmenu.addAction(act_norm)
        bmenu.addAction(act_bright)

        # Lighting menu
        lmenu = mb.addMenu("Lighting")
        act_head = QAction("Headlight", self)
        act_head.triggered.connect(lambda: self.apply_lighting("headlight"))
        lmenu.addAction(act_head)
        act_studio = QAction("Studio", self)
        act_studio.triggered.connect(lambda: self.apply_lighting("studio"))
        lmenu.addAction(act_studio)
        act_off = QAction("Off", self)
        act_off.triggered.connect(lambda: self.apply_lighting("off"))
        lmenu.addAction(act_off)

        # Info menu
        imenu = mb.addMenu("Info")
        act_refresh = QAction("Refresh Model Info (I)", self)
        act_refresh.setShortcut("I")
        act_refresh.triggered.connect(self.update_info)
        imenu.addAction(act_refresh)

    def _on_key(self, obj, evt):
        ''' Handle keypress events for debugging. '''
        key = self.iren.GetKeySym().lower()

        # Zoom: + / - (also '=' and '_' and keypad variants)
        if key in ('plus','equal','kp_add','minus','underscore','kp_subtract'):
            fast = bool(self.iren.GetShiftKey())
            direction = +1 if key in ('plus','equal','kp_add') else -1
            self.zoom(direction, fast=fast)
            return

        # Movement step (Shift = x5)
        if key in ('left','right','up','down','prior','next'):  # prior=PageUp, next=PageDown
            fast = bool(self.iren.GetShiftKey())
            step = self._move_step(fast=fast)
            dx = dy = dz = 0.0
            if key == 'left':  dx = -step
            if key == 'right': dx = +step
            if key == 'up':    dy = +step
            if key == 'down':  dy = -step
            if key == 'prior':     dz = +step
            if key == 'next':      dz = -step
            self.nudge_model(dx, dy, dz)
            return

        if key == 't':
            self.dump_actor_textures()
        elif key == 'f':
            self.force_fullbright()
        elif key == 'r':
            self.reapply_texture_fix()
        elif key == 'd':
            self.diagnose_textures()

    def save_model_as(self):
        ''' Export current scene to Wavefront OBJ (+ MTL + textures). '''
        if not self.renderer or self.renderer.GetActors().GetNumberOfItems() == 0:
            self.statusBar().showMessage("Nothing to save")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model As", "",
            "Wavefront OBJ (*.obj)"
        )
        if not path:
            return
        # Ensure .obj extension
        if not path.lower().endswith(".obj"):
            path += ".obj"
        prefix = os.path.splitext(path)[0]
        try:
            exp = vtk.vtkOBJExporter()
            exp.SetRenderWindow(self.ren_win)
            exp.SetFilePrefix(prefix)
            exp.Write()
            self.statusBar().showMessage(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            self.statusBar().showMessage(f"Save failed: {e}")
            print("Save error:", e)

    def _bind_debug_keys(self):
        ''' Bind debug key events. '''
        self.iren.AddObserver("KeyPressEvent", self._on_key)


    def _process_alpha(self, img: vtk.vtkImageData):
        ''' Process image alpha channel according to transparency_mode.'''
        if not img or img.GetNumberOfScalarComponents() < 4:
            return img
        mode = self.transparency_mode
        if mode == "blend":
            return img
        if mode == "opaque":
            rgb_extract = vtk.vtkImageExtractComponents()
            rgb_extract.SetInputData(img)
            rgb_extract.SetComponents(0,1,2)
            rgb_extract.Update()
            return rgb_extract.GetOutput()
        # cutout: binary mask >128 -> 255 else 0
        alpha_extract = vtk.vtkImageExtractComponents()
        alpha_extract.SetInputData(img)
        alpha_extract.SetComponents(3)
        alpha_extract.Update()
        alpha = alpha_extract.GetOutput()
        thresh = vtk.vtkImageThreshold()
        thresh.SetInputData(alpha)
        thresh.ThresholdByUpper(128)
        thresh.ReplaceInOn();  thresh.SetInValue(0)
        thresh.ReplaceOutOn(); thresh.SetOutValue(255)
        thresh.Update()
        alpha_processed = thresh.GetOutput()
        rgb_extract = vtk.vtkImageExtractComponents()
        rgb_extract.SetInputData(img)
        rgb_extract.SetComponents(0,1,2)
        rgb_extract.Update()
        append = vtk.vtkImageAppendComponents()
        append.AddInputConnection(rgb_extract.GetOutputPort())
        append.AddInputData(alpha_processed)
        append.Update()
        return append.GetOutput()


    def toggle_unlit(self, enabled: bool):
        ''' Toggle unlit (fullbright) rendering mode. '''
        self.unlit = enabled
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: 
                continue
            p = a.GetProperty()
            if enabled:
                # Store previous (ambient,diffuse)
                if not hasattr(a, "_old_lighting"):
                    a._old_lighting = (p.GetAmbient(), p.GetDiffuse(), p.GetSpecular())
                p.SetAmbient(1.0)
                p.SetDiffuse(0.0)
                p.SetSpecular(0.0)
                if hasattr(p, "LightingOff"):
                    p.LightingOff()
                tex = a.GetTexture()
                if tex and hasattr(vtk.vtkTexture,"VTK_TEXTURE_BLENDING_MODE_REPLACE") and hasattr(tex,"SetBlendingMode"):
                    tex.SetBlendingMode(vtk.vtkTexture.VTK_TEXTURE_BLENDING_MODE_REPLACE)
            else:
                if hasattr(a, "_old_lighting"):
                    amb,dif,spec = a._old_lighting
                    p.SetAmbient(amb)
                    p.SetDiffuse(dif)
                    p.SetSpecular(spec)
                if hasattr(p, "LightingOn"): 
                    p.LightingOn()
        self.statusBar().showMessage(f"Unlit: {'ON' if enabled else 'OFF'}")
        self.ren_win.Render()

    def apply_brightness(self):
        ''' Apply brightness mode to all actors. '''
        factor = 1.0 if self.brightness_mode == "normal" else 1.25
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a:
                continue
            p = a.GetProperty()
            if self.unlit:
                p.SetAmbient(min(1.0, 1.0 * factor))
                p.SetDiffuse(0.0)
            else:
                # Base values brighter than before
                amb = 0.50 * factor
                dif = 0.85 * factor
                p.SetAmbient(min(1.0, amb))
                p.SetDiffuse(min(1.0, dif))
                p.SetSpecular(0.05)
        # Boost headlight if present
        self.ren_win.Render()

    def _init_info_dock(self):
        ''' Initialize the info dock widget. '''
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        dock = QDockWidget("Model Info", self)
        dock.setWidget(self.info)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ---------- Core Loading ----------
    def open_file(self):
        ''' Open file dialog to select and load a 3D model. '''
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "",
            "3D Models (*.obj *.stl *.ply)"
        )
        if fn:
            self.load_model(fn)

    def load_model(self, filename: str):
        ''' Load a 3D model from file. '''
        self.renderer.RemoveAllViewProps()
        self._shared_texture = None  # reset cached texture
        self._texture_cache.clear()
        self._actor_material.clear()
        self.model_transform.Identity()  # reset model transform
        self.current_file = filename
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".obj":
                actors = self._load_obj_manual(filename)
                for a in actors:
                    self.renderer.AddActor(a)
                # Try per-material textures from MTL; fallback to manual atlas if none found
                mtl_path = os.path.splitext(filename)[0] + ".mtl"
                if not self._apply_textures_from_mtl(filename, mtl_path):
                    self._apply_manual_atlas(filename, mtl_path)

            elif ext in (".stl", ".ply"):
                self._load_mesh_simple(filename)
            else:
                self.statusBar().showMessage(f"Unsupported: {ext}")
                return

            # Attach transform to all actors and reset view
            self._attach_model_transform()
            self.reset_view()

            self.update_info()
            self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)}")
        except Exception as e:
            self.statusBar().showMessage(f"Load failed: {e}")
            print("Load error:", e)


    def _load_png(self, path: str):
        ''' Load PNG image as vtkImageData; returns None if failed.'''
        if not os.path.isfile(path):
            return None
        r = vtk.vtkPNGReader()
        r.SetFileName(path)
        try:
            r.Update()
        except Exception:
            return None
        img = r.GetOutput()
        if not img or img.GetDimensions() == (0,0,0):
            return None
        return img


    def _extract_mtl_map(self, tokens):
        """Pick the actual filename from a map_* line (skip options like -o, -s, -bm)."""
        if not tokens:
            return None
        # Known image extensions VTK can read out of the box
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".tga")
        # Scan from the end; choose the first token that looks like a file
        for tok in reversed(tokens):
            t = tok.strip().strip('"')
            if "." in t and t.lower().endswith(exts):
                return t
        # Fallback: last token
        return tokens[-1].strip().strip('"')


    def _compose_rgba(self, rgb_img, alpha_img):
        ''' Combine RGB image and Alpha image into RGBA image.'''
        if not (rgb_img and alpha_img): return rgb_img
        if alpha_img.GetNumberOfScalarComponents() not in (1,4):
            return rgb_img
        # Extract single alpha component if needed
        if alpha_img.GetNumberOfScalarComponents() > 1:
            extract = vtk.vtkImageExtractComponents()
            extract.SetInputData(alpha_img)
            extract.SetComponents(3 if alpha_img.GetNumberOfScalarComponents()==4 else 0)
            extract.Update()
            alpha_single = extract.GetOutput()
        else:
            alpha_single = alpha_img
        append = vtk.vtkImageAppendComponents()
        append.AddInputData(rgb_img)
        append.AddInputData(alpha_single)
        append.Update()
        return append.GetOutput()


    def _apply_manual_atlas(self, obj_path: str, mtl_path: str):
        ''' Apply a manual texture atlas named after the model: <name>-RGB(A).png, <name>-Alpha.png. '''
        base_dir = os.path.dirname(obj_path)
        base_name = os.path.splitext(os.path.basename(obj_path))[0]

        # Prefer model-specific atlas filenames
        rgb_path   = os.path.join(base_dir, f"{base_name}-RGB.png")
        rgba_path  = os.path.join(base_dir, f"{base_name}-RGBA.png")
        alpha_path = os.path.join(base_dir, f"{base_name}-Alpha.png")

        rgba_img = self._load_png(rgba_path)
        rgb_img  = self._load_png(rgb_path)
        alpha_img= self._load_png(alpha_path)

        # Optional: keep legacy "house-*.png" ONLY if the model is actually "house"
        if not (rgba_img or rgb_img):
            if base_name.lower() == "house":
                rgb_path   = os.path.join(base_dir, "house-RGB.png")
                rgba_path  = os.path.join(base_dir, "house-RGBA.png")
                alpha_path = os.path.join(base_dir, "house-Alpha.png")
                rgba_img = self._load_png(rgba_path)
                rgb_img  = self._load_png(rgb_path)
                alpha_img= self._load_png(alpha_path)

        if not rgba_img:
            if rgb_img and alpha_img:
                rgba_img = self._compose_rgba(rgb_img, alpha_img)
            elif rgb_img:
                rgba_img = rgb_img
            else:
                print(f"[TEX] No atlas images found for '{base_name}'. Skipping atlas fallback.")
                return

        rgba_img = self._process_alpha(rgba_img)

        if self._shared_texture is None:
            tex = vtk.vtkTexture()
            tex.SetInputData(rgba_img)
            try: tex.EdgeClampOn()
            except: pass
            if hasattr(tex,"SetUseSRGBColorSpace"):
                try: tex.SetUseSRGBColorSpace(True)
                except: pass
            self._shared_texture = tex
        tex = self._shared_texture
        try: tex.InterpolateOn()
        except: pass

        acts = self.renderer.GetActors(); acts.InitTraversal()
        replaced = 0
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            if a.GetTexture():  # don't override MTL-applied textures
                continue
            a.SetTexture(tex)
            replaced += 1
            p = a.GetProperty()
            p.SetColor(1,1,1)
            if self.unlit:
                p.SetAmbient(1.0); p.SetDiffuse(0.0); p.SetSpecular(0.0)
            else:
                p.SetAmbient(0.50); p.SetDiffuse(0.85); p.SetSpecular(0.05)
            if self.transparency_mode != "blend":
                p.SetOpacity(1.0)
        if replaced == 0:
            print(f"[TEX] Atlas fallback skipped (all actors already textured).")

        self.apply_brightness()
        print(f"[TEX] Atlas applied to {replaced} actors using prefix '{base_name}-*.png'.")
        self.ren_win.Render()


    def _load_mesh_simple(self, path: str):
        ''' Load STL or PLY using trimesh if available, else VTK reader. '''
        # STL / PLY simple path
        if trimesh:
            mesh = trimesh.load(path, force='mesh')
            pts = vtk.vtkPoints()
            for v in mesh.vertices:
                pts.InsertNextPoint(*v)
            cells = vtk.vtkCellArray()
            for tri in mesh.faces:
                t = vtk.vtkTriangle()
                t.GetPointIds().SetId(0, tri[0])
                t.GetPointIds().SetId(1, tri[1])
                t.GetPointIds().SetId(2, tri[2])
                cells.InsertNextCell(t)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetPolys(cells)
        else:
            # Fallback readers
            ext = os.path.splitext(path)[1].lower()
            if ext == ".stl":
                r = vtk.vtkSTLReader()
            else:
                r = vtk.vtkPLYReader()
            r.SetFileName(path)
            r.Update()
            poly = r.GetOutput()

        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(poly)
        nf.AutoOrientNormalsOn()
        nf.SplittingOn()
        nf.Update()

        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(nf.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.85, 0.85, 0.9)
        a.GetProperty().SetAmbient(0.2); a.GetProperty().SetDiffuse(0.8)
        try:
            a.GetProperty().BackfaceCullingOn()
        except:
            pass
        self.renderer.AddActor(a)

    # ---------- Appearance ----------
    def apply_lighting(self, preset: str):
        ''' Apply lighting preset: "headlight", "studio", "off". '''
        self.renderer.RemoveAllLights()
        if preset == "off":
            self.statusBar().showMessage("Lighting: off")
            self.ren_win.Render()
            return
        if preset == "studio":
            # Simple 3-point + headlight
            def add(pos, intensity):
                l = vtk.vtkLight()
                l.SetLightTypeToSceneLight()
                l.SetPosition(*pos)
                l.SetFocalPoint(0,0,0)
                l.SetIntensity(intensity)
                self.renderer.AddLight(l)
            add((4,2,4), 0.9)    # key
            add((-4,2,1), 0.5)   # fill
            add((2,-3, -4), 0.4) # back
        # Headlight always
        hl = vtk.vtkLight()
        hl.SetLightTypeToHeadlight()
        hl.SetIntensity(0.5 if preset == "studio" else 1.1)
        # Gentle ambient fill (scene light)
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(0, 5, 0)
        fill.SetFocalPoint(0,0,0)
        fill.SetIntensity(0.25 if preset == "studio" else 0.18)
        self.renderer.AddLight(fill)
        self.renderer.AddLight(hl)
        self.statusBar().showMessage(f"Lighting: {preset}")
        self.ren_win.Render()

    def _load_obj_manual(self, obj_path: str):
        """Minimal OBJ parser supporting v, vt, f (tri or n-gon), ignoring normals & materials.
           Builds unique (v,vt) pairs so texture coordinates map correctly."""
        
        if not os.path.isfile(obj_path):
            print(f"[OBJ-MANUAL] File not found: {obj_path}")
            return []

        verts = []
        uvs = []
        current_mat = "default"
        mat_faces = {}  # material -> list of faces

        def add_face(tokens):
            # tokens like v/vt or v// or v/vt/vn
            corner_pairs = []
            for tok in tokens:
                parts = tok.split('/')
                v_i = int(parts[0])
                vt_i = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                # Handle negative indices
                if v_i < 0: v_i = len(verts) + 1 + v_i
                if vt_i < 0: vt_i = len(uvs) + 1 + vt_i
                corner_pairs.append((v_i-1, vt_i-1 if vt_i>0 else -1))
            # Triangulate fan if needed
            if len(corner_pairs) < 3:
                return
            for i in range(1, len(corner_pairs)-1):
                tri = [corner_pairs[0], corner_pairs[i], corner_pairs[i+1]]
                mat_faces.setdefault(current_mat, []).append(tri)

        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                ls = line.strip().split()
                if not ls: continue
                tag = ls[0]
                if tag == 'v' and len(ls) >= 4:
                    verts.append(tuple(map(float, ls[1:4])))
                elif tag == 'vt' and len(ls) >= 3:
                    uvs.append((float(ls[1]), float(ls[2])))
                elif tag == 'f' and len(ls) >= 4:
                    add_face(ls[1:])
                elif tag == 'usemtl' and len(ls) >= 2:
                    current_mat = ls[1]

        # Build VTK data per material
        actors = []
        for mat_name, tris in mat_faces.items():
            # Map (v,vt) -> new index
            pair_to_new = {}
            points = vtk.vtkPoints()
            tcoords = vtk.vtkFloatArray()
            tcoords.SetNumberOfComponents(2)
            tcoords.SetName("TCoords")
            polys = vtk.vtkCellArray()

            def get_index(v_idx, vt_idx):
                key = (v_idx, vt_idx)
                if key in pair_to_new:
                    return pair_to_new[key]
                new_id = points.InsertNextPoint(*verts[v_idx])
                if 0 <= vt_idx < len(uvs):
                    u,v = uvs[vt_idx]
                else:
                    u,v = 0.0, 0.0
                tcoords.InsertNextTuple((u, v))
                pair_to_new[key] = new_id
                return new_id

            for tri in tris:
                ids = [get_index(v, vt) for (v, vt) in tri]
                polys.InsertNextCell(3)
                for pid in ids:
                    polys.InsertCellPoint(pid)

            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(polys)
            poly.GetPointData().SetTCoords(tcoords)

            mapper = vtk.vtkPolyDataMapper()

            if not self.unlit:
                nf = vtk.vtkPolyDataNormals()
                nf.SetInputData(poly)
                nf.AutoOrientNormalsOn()
                nf.SplittingOff()
                nf.Update()
                mapper.SetInputConnection(nf.GetOutputPort())
            else:
                mapper.SetInputData(poly)

            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(1,1,1)
            prop.SetAmbient(0.4)
            prop.SetDiffuse(0.6)
            actors.append(actor)
            # Track material for this actor
            self._actor_material[actor] = mat_name

        print(f"[OBJ-MANUAL] verts={len(verts)} uvs={len(uvs)} materials={len(mat_faces)} actors={len(actors)}")
        return actors

    def toggle_wireframe(self):
        ''' Toggle wireframe/solid rendering mode for all actors. '''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            p = a.GetProperty()
            if p.GetRepresentation() == vtk.VTK_SURFACE:
                p.SetRepresentationToWireframe()
            else:
                p.SetRepresentationToSurface()
        self.ren_win.Render()


    def _load_image(self, path: str):
        '''Load an image (png/jpg/jpeg/bmp/tif/tiff/tga) as vtkImageData; returns None if failed.'''
        if not path:
            return None
        # Normalize and check existence (VTK readers can handle absolute/relative)
        path = os.path.normpath(path)
        if not os.path.isabs(path) and self.current_file:
            base_dir = os.path.dirname(self.current_file)
            path = os.path.normpath(os.path.join(base_dir, path))
        if not os.path.isfile(path):
            return None
        # Try factory (covers PNG, JPEG, BMP, TIFF, TGA, etc.)
        try:
            factory = vtk.vtkImageReader2Factory()
            reader = factory.CreateImageReader2(path)
        except Exception:
            reader = None
        if reader is None:
            # Fallback by extension
            ext = os.path.splitext(path)[1].lower()
            if ext == ".png":
                reader = vtk.vtkPNGReader()
            elif ext in (".jpg", ".jpeg"):
                reader = vtk.vtkJPEGReader()
            elif ext == ".bmp":
                reader = vtk.vtkBMPReader()
            elif ext in (".tif", ".tiff"):
                reader = vtk.vtkTIFFReader()
            elif ext == ".tga":
                reader = vtk.vtkImageReader2()  # generic (may need factory)
            else:
                return None
        reader.SetFileName(path)
        try:
            reader.Update()
        except Exception:
            return None
        img = reader.GetOutput()
        return img if img and img.GetDimensions() != (0,0,0) else None

    def _parse_mtl(self, mtl_path: str):
        '''Parse minimal MTL: returns dict name -> {map_Kd, map_d, Ka, Kd}.'''
        mats = {}
        if not os.path.isfile(mtl_path):
            return mats
        cur = None
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ls = line.strip().split()
                if not ls: continue
                tag = ls[0].lower()
                if tag == "newmtl" and len(ls) >= 2:
                    cur = ls[1]
                    mats[cur] = {}
                elif cur:
                    if tag == "map_kd" and len(ls) >= 2:
                        mats[cur]["map_Kd"] = self._extract_mtl_map(ls[1:])
                    elif tag in ("map_d", "map_tr") and len(ls) >= 2:
                        mats[cur]["map_d"] = self._extract_mtl_map(ls[1:])
                    elif tag == "kd" and len(ls) >= 4:
                        mats[cur]["Kd"] = tuple(map(float, ls[1:4]))
                    elif tag == "ka" and len(ls) >= 4:
                        mats[cur]["Ka"] = tuple(map(float, ls[1:4]))
        return mats

    def _apply_textures_from_mtl(self, obj_path: str, mtl_path: str) -> bool:
        '''Apply textures per material from MTL. Returns True if any texture applied.'''
        mats = self._parse_mtl(mtl_path)
        if not mats:
            return False
        base_dir = os.path.dirname(obj_path)
        any_applied = False

        def norm_path(p):
            if not p: return None
            p = p.strip().strip('"')
            return os.path.normpath(p if os.path.isabs(p) else os.path.join(base_dir, p))

        # Image->texture cache (per alpha mode)
        def get_texture(img_path: Optional[str], alpha_path: Optional[str]):
            img_full = norm_path(img_path) if img_path else None
            a_full = norm_path(alpha_path) if alpha_path else None
            key = (img_full, a_full, self.transparency_mode)
            if key in self._texture_cache:
                return self._texture_cache[key]
            rgb = self._load_image(img_full) if img_full else None
            if not rgb:
                return None
            rgba = rgb
            if a_full:
                aimg = self._load_image(a_full)
                if aimg:
                    rgba = self._compose_rgba(rgb, aimg)
            rgba = self._process_alpha(rgba)
            tex = vtk.vtkTexture()
            tex.SetInputData(rgba)
            try: tex.EdgeClampOn()
            except: pass
            if hasattr(tex, "SetUseSRGBColorSpace"):
                try: tex.SetUseSRGBColorSpace(True)
                except: pass
            try: tex.InterpolateOn()
            except: pass
            self._texture_cache[key] = tex
            return tex

        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            mat = self._actor_material.get(a)
            info = mats.get(mat) if mat else None
            if not info: continue
            map_kd = info.get("map_Kd")
            map_d  = info.get("map_d")
            tex = get_texture(map_kd, map_d)
            if tex:
                a.SetTexture(tex)
                any_applied = True
                p = a.GetProperty()
                if "Kd" in info:
                    p.SetDiffuseColor(*info["Kd"])
                if self.unlit:
                    p.SetAmbient(1.0); p.SetDiffuse(0.0); p.SetSpecular(0.0)
                else:
                    p.SetAmbient(0.50); p.SetDiffuse(0.85); p.SetSpecular(0.05)
                if self.transparency_mode != "blend":
                    p.SetOpacity(1.0)

        if any_applied:
            self.apply_brightness()
            self.ren_win.Render()
        return any_applied


    def cycle_background(self):
        ''' Cycle through predefined background colors. '''
        self.bg_idx = (self.bg_idx + 1) % len(self.bg_colors)
        self.set_background(*self.bg_colors[self.bg_idx])

    def set_background(self, r, g, b):
        ''' Set background color. '''
        self.renderer.SetBackground(r, g, b)
        self.ren_win.Render()

    def reset_view(self):
        '''Robust reset: frame all visible props and position camera back.'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        if acts.GetNumberOfItems() == 0:
            return
        bounds = self.renderer.ComputeVisiblePropBounds()
        # bounds = (xmin,xmax,ymin,ymax,zmin,zmax)
        if not bounds or all(v == 0 for v in bounds):
            return
        xmin,xmax,ymin,ymax,zmin,zmax = bounds
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        cz = (zmin + zmax) * 0.5
        span_x = max(1e-6, xmax - xmin)
        span_y = max(1e-6, ymax - ymin)
        span_z = max(1e-6, zmax - zmin)
        max_span = max(span_x, span_y, span_z)

        cam = self.renderer.GetActiveCamera()
        # View direction (diagonally down to give depth)
        vx, vy, vz = (1.0, 0.8, 1.0)
        # Normalize
        mag = (vx*vx + vy*vy + vz*vz) ** 0.5
        vx /= mag; vy /= mag; vz /= mag

        dist = max_span * 2.2
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + vx * dist, cy + vy * dist, cz + vz * dist)
        cam.SetViewUp(0, 1, 0)
        cam.SetClippingRange(dist * 0.01, dist * 10.0)

        # Also call VTK's own reset to adjust dolly heuristics (safe)
        self.renderer.ResetCameraClippingRange()
        self.ren_win.Render()


    # ---------- Model movement ----------
    def _attach_model_transform(self):
        '''Attach the shared transform to all current actors.'''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            a.SetUserTransform(self.model_transform)

    def _move_step(self, fast: bool = False) -> float:
        '''Compute movement step from scene bounds.'''
        b = self.renderer.ComputeVisiblePropBounds()
        if not b or all(v == 0 for v in b):
            base = 1.0
        else:
            span = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
            base = max(1e-3, 0.02 * span)  # 2% of scene size
        return base * (5.0 if fast else 1.0)

    def nudge_model(self, dx: float, dy: float, dz: float):
        '''Translate the whole model by (dx,dy,dz) in world space.'''
        self.model_transform.PostMultiply()
        self.model_transform.Translate(dx, dy, dz)
        self.model_transform.Modified()
        self.ren_win.Render()

    # ---------- Zoom ----------
    def zoom(self, direction: int, fast: bool = False):
        '''Zoom camera: direction +1=in, -1=out. Shift (fast) makes it stronger.'''
        cam = self.renderer.GetActiveCamera()
        if not cam:
            return
        step = 1.2 if fast else 1.08
        factor = step if direction > 0 else 1.0/step
        if cam.GetParallelProjection():
            # Parallel: adjust scale directly
            scale = cam.GetParallelScale()
            cam.SetParallelScale(scale / factor)
        else:
            # Perspective: dolly around focal point
            cam.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.ren_win.Render()

    # ---------- Mouse style ----------
    def set_mouse_style(self, style_name: str):
        style_name = (style_name or "").lower()
        styles = {
            "trackball": vtk.vtkInteractorStyleTrackballCamera,
            "terrain": vtk.vtkInteractorStyleTerrain,
            "joystick": vtk.vtkInteractorStyleJoystickCamera,
        }
        cls = styles.get(style_name)
        if not cls:
            return
        style = cls()
        style.SetDefaultRenderer(self.renderer)
        if hasattr(style, "SetCurrentRenderer"):
            style.SetCurrentRenderer(self.renderer)
        self.iren.SetInteractorStyle(style)
        self._interactor_style = style  # keep reference alive
        self.mouse_style = style_name
        self.statusBar().showMessage(f"Mouse: {style_name}")

    # ---------- Info ----------
    def update_info(self):
        ''' Update model info in the dock widget. '''
        acts = self.renderer.GetActors(); acts.InitTraversal()
        n = acts.GetNumberOfItems()
        total_pts = total_polys = textured = 0
        for _ in range(n):
            a = acts.GetNextActor()
            if not a: continue
            m = a.GetMapper()
            if not m: continue
            pd = m.GetInput()
            if isinstance(pd, vtk.vtkPolyData):
                total_pts += pd.GetNumberOfPoints()
                total_polys += pd.GetNumberOfPolys()
            if a.GetTexture(): textured += 1
        lines = [
            "Model Info",
            "-----------",
            f"File: {os.path.basename(self.current_file) if self.current_file else '-'}",
            f"Actors: {n}",
            f"Textured: {textured}",
            f"Points: {total_pts}",
            f"Polys: {total_polys}"
        ]
        self.info.setPlainText("\n".join(lines))


# Standalone launch helper
def main():
    '''Launch the viewer application.'''
    import sys
    app = QApplication(sys.argv)
    w = ViewerApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

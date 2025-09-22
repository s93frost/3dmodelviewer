"""Minimal 3D Model Viewer (PyQt5 + VTK)"""

import os

import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QAction, QFileDialog, QDockWidget,
    QPlainTextEdit, QWidget, QVBoxLayout
)

import vtkmodules.vtkIOImage  # ensure PNG/JPG readers are registered

from typing import Optional

# Optional (guarded)
try:
    import trimesh
except Exception:
    trimesh = None



class ViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal 3D Viewer")
        self.resize(1000, 700)
        self.unlit = False
        self._shared_texture = None
        

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
        except: pass

        self.ren_win.AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.vtk_widget.Initialize()

        # Simple state
        self.pixelated = False
        self.bg_colors = [(1,1,1),(0.9,0.9,0.9),(0.2,0.2,0.2),(0,0,0)]
        self.bg_idx = 2
        self.current_file: Optional[str] = None

        # UI
        self._init_menu()
        self._init_info_dock()

        
        

        # Lighting + initial background
        self.apply_lighting("headlight")
        self.set_background(*self.bg_colors[self.bg_idx])

        # Load default if exists
        default_obj = "assets/house.obj"
        if os.path.isfile(default_obj):
            self.load_model(default_obj)

        self._bind_debug_keys()


    def dump_actor_textures(self):
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

    
    def diagnose_textures(self):
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
                        idx = (midy * dims[0] + midx) * comps + (comps-1)
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
        # Call after load if still dark
        self.force_fullbright()
    

    # ---------- UI ----------
    def _init_menu(self):
        mb = self.menuBar()

        fmenu = mb.addMenu("File")
        act_open = QAction("Open...", self)
        act_open.triggered.connect(self.open_file)
        fmenu.addAction(act_open)

        vmenu = mb.addMenu("View")
        act_reset = QAction("Reset View", self)
        act_reset.triggered.connect(self.reset_view)
        vmenu.addAction(act_reset)

        act_unlit = QAction("Unlit Textures", self)
        act_unlit.setCheckable(True)
        act_unlit.triggered.connect(self.toggle_unlit)
        vmenu.addAction(act_unlit)

        act_bg = QAction("Cycle Background (B)", self)
        act_bg.setShortcut("B")
        act_bg.triggered.connect(self.cycle_background)
        vmenu.addAction(act_bg)

        act_wire = QAction("Toggle Wireframe (W)", self)
        act_wire.setShortcut("W")
        act_wire.triggered.connect(self.toggle_wireframe)
        vmenu.addAction(act_wire)

        act_pix = QAction("Pixelated Textures", self)
        act_pix.setCheckable(True)
        act_pix.triggered.connect(self.toggle_pixelated)
        vmenu.addAction(act_pix)

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

        imenu = mb.addMenu("Info")
        act_refresh = QAction("Refresh Model Info (I)", self)
        act_refresh.setShortcut("I")
        act_refresh.triggered.connect(self.update_info)
        imenu.addAction(act_refresh)

    def _on_key(self, obj, evt):
        key = self.iren.GetKeySym().lower()
        if key == 't':
            self.dump_actor_textures()
        elif key == 'f':
            self.force_fullbright()
        elif key == 'r':
            self.reapply_texture_fix()
        elif key == 'd':
            self.diagnose_textures()
        
    def _bind_debug_keys(self):
        self.iren.AddObserver("KeyPressEvent", self._on_key)
        
    def disable_textures_if_no_uv(self):
        acts = self.renderer.GetActors(); acts.InitTraversal()
        removed = 0
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            m = a.GetMapper()
            pd = m.GetInput() if m else None
            has_uv = False
            if pd and pd.GetPointData() and pd.GetPointData().GetTCoords():
                has_uv = True
            if not has_uv and a.GetTexture():
                a.SetTexture(None)
                removed += 1
        print(f"[FIX] Removed textures from {removed} actors lacking UVs.")
        self.ren_win.Render()

    def toggle_unlit(self, enabled: bool):
        self.unlit = enabled
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            p = a.GetProperty()
            if enabled:
                # Store previous (ambient,diffuse)
                if not hasattr(a, "_old_lighting"):
                    a._old_lighting = (p.GetAmbient(), p.GetDiffuse(), p.GetSpecular())
                p.SetAmbient(1.0); p.SetDiffuse(0.0); p.SetSpecular(0.0)
                if hasattr(p, "LightingOff"): p.LightingOff()
                tex = a.GetTexture()
                if tex and hasattr(vtk.vtkTexture,"VTK_TEXTURE_BLENDING_MODE_REPLACE") and hasattr(tex,"SetBlendingMode"):
                    tex.SetBlendingMode(vtk.vtkTexture.VTK_TEXTURE_BLENDING_MODE_REPLACE)
            else:
                if hasattr(a, "_old_lighting"):
                    amb,dif,spec = a._old_lighting
                    p.SetAmbient(amb); p.SetDiffuse(dif); p.SetSpecular(spec)
                if hasattr(p, "LightingOn"): p.LightingOn()
        self.statusBar().showMessage(f"Unlit: {'ON' if enabled else 'OFF'}")
        self.ren_win.Render()

    def _init_info_dock(self):
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        dock = QDockWidget("Model Info", self)
        dock.setWidget(self.info)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ---------- Core Loading ----------
    def open_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "",
            "3D Models (*.obj *.stl *.ply)"
        )
        if fn:
            self.load_model(fn)

    def load_model(self, filename: str):
        self.renderer.RemoveAllViewProps()
        self.current_file = filename
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".obj":
                self._load_obj(filename)
            elif ext in (".stl", ".ply"):
                self._load_mesh_simple(filename)
            else:
                self.statusBar().showMessage(f"Unsupported: {ext}")
                return
            self.reset_view()
            self.update_info()
            self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)}")
        except Exception as e:
            self.statusBar().showMessage(f"Load failed: {e}")
            print("Load error:", e)
    

    def _load_png(self, path: str):
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
    
    def _compose_rgba(self, rgb_img, alpha_img):
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
        base_dir = os.path.dirname(obj_path)
        rgb_path   = os.path.join(base_dir, "house-RGB.png")
        rgba_path  = os.path.join(base_dir, "house-RGBA.png")
        alpha_path = os.path.join(base_dir, "house-Alpha.png")

        rgba_img = self._load_png(rgba_path)
        rgb_img  = self._load_png(rgb_path)
        alpha_img= self._load_png(alpha_path)

        # Prefer existing RGBA, else compose from RGB+Alpha
        if not rgba_img:
            if rgb_img and alpha_img:
                rgba_img = self._compose_rgba(rgb_img, alpha_img)
            elif rgb_img:
                rgba_img = rgb_img  # will be treated as RGB
            else:
                print("[TEX] No atlas images found.")
                return

        tex = vtk.vtkTexture()
        if isinstance(rgba_img, vtk.vtkImageData):
            tex.SetInputData(rgba_img)
        tex.InterpolateOn()
        if self.pixelated:
            tex.InterpolateOff()
        try:
            tex.EdgeClampOn()
        except: pass
        if hasattr(tex,"SetUseSRGBColorSpace"):
            try: tex.SetUseSRGBColorSpace(True)
            except: pass
        # Assign to every actor whose current texture is missing or empty
        acts = self.renderer.GetActors(); acts.InitTraversal()
        replaced = 0
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if not a: continue
            cur = a.GetTexture()
            need = True
            if cur and hasattr(cur,"GetInput"):
                img = cur.GetInput()
                if img and img.GetDimensions() != (0,0,0):
                    need = False
            if need:
                a.SetTexture(tex)
                replaced += 1
            p = a.GetProperty()
            p.SetColor(1,1,1)
            p.SetAmbient(0.4)
            p.SetDiffuse(0.6)
            p.SetSpecular(0.0)
            if self.unlit:
                p.SetAmbient(1.0); p.SetDiffuse(0.0)
        print(f"[TEX] Manual atlas applied to {replaced} actors.")
        self.ren_win.Render()


    def _load_obj(self, obj_path: str):
        """Try vtkOBJImporter; if UVs degenerate, fallback to manual parser with proper TCoords."""
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        used_importer = True
        failed_uv = False

        # 1) Try importer
        imp = vtk.vtkOBJImporter()
        imp.SetFileName(obj_path)
        if os.path.isfile(mtl_path):
            imp.SetFileNameMTL(mtl_path)
        imp.SetTexturePath(os.path.dirname(obj_path))
        imp.SetRenderWindow(self.ren_win)
        try:
            imp.Update()
        except Exception as e:
            print("[OBJ] Importer failed:", e)
            used_importer = False

        if used_importer:
            # Move actors (same as before)
            rens = self.ren_win.GetRenderers(); rens.InitTraversal()
            extra = []
            for _ in range(rens.GetNumberOfItems()):
                ren = rens.GetNextItem()
                if ren is not self.renderer:
                    extra.append(ren)
            for er in extra:
                acts = er.GetActors(); acts.InitTraversal()
                for _ in range(acts.GetNumberOfItems()):
                    a = acts.GetNextActor()
                    if a: self.renderer.AddActor(a)
                self.ren_win.RemoveRenderer(er)

            # Check UV quality (sample first actor)
            acts = self.renderer.GetActors(); acts.InitTraversal()
            sample_actor = acts.GetNextActor()
            if sample_actor:
                m = sample_actor.GetMapper()
                pd = m.GetInput() if m else None
                if pd and pd.GetPointData() and pd.GetPointData().GetTCoords():
                    tc = pd.GetPointData().GetTCoords()
                    if tc.GetNumberOfTuples() == 0:
                        failed_uv = True
                    else:
                        # If first 50 tuples all (0,0) treat as failed
                        limit = min(50, tc.GetNumberOfTuples())
                        all_zero = True
                        for i in range(limit):
                            u,v = tc.GetTuple(i)[:2]
                            if abs(u) > 1e-6 or abs(v) > 1e-6:
                                all_zero = False
                                break
                        failed_uv = all_zero
                else:
                    failed_uv = True
            else:
                failed_uv = True

        if (not used_importer) or failed_uv:
            print("[OBJ] Falling back to manual OBJ loader (UV rebuild).")
            self.renderer.RemoveAllViewProps()
            actors = self._load_obj_manual(obj_path)
            for a in actors:
                self.renderer.AddActor(a)
            # Manual texture atlas
            self._apply_manual_atlas(obj_path, mtl_path)

        # Camera & lighting baseline
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        cam = self.renderer.GetActiveCamera()
        b = self.renderer.ComputeVisiblePropBounds()
        if not all(v == 0 for v in b):
            cx=(b[0]+b[1])/2; cy=(b[2]+b[3])/2; cz=(b[4]+b[5])/2
            cam.SetFocalPoint(cx,cy,cz)
        self.ren_win.Render()

    def _load_mesh_simple(self, path: str):
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
        self.renderer.AddActor(a)

    # ---------- Appearance ----------
    def apply_lighting(self, preset: str):
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
        hl.SetIntensity(0.35 if preset == "studio" else 1.0)
        self.renderer.AddLight(hl)
        self.statusBar().showMessage(f"Lighting: {preset}")
        self.ren_win.Render()

    def _load_obj_manual(self, obj_path: str):
        """Minimal OBJ parser supporting v, vt, f (tri or n-gon), ignoring normals & materials.
           Builds unique (v,vt) pairs so texture coordinates map correctly."""
        verts = []
        uvs = []
        faces = []      # list of lists of (v_idx, vt_idx)
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
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(1,1,1)
            prop.SetAmbient(0.4)
            prop.SetDiffuse(0.6)
            actors.append(actor)

        print(f"[OBJ-MANUAL] verts={len(verts)} uvs={len(uvs)} materials={len(mat_faces)} actors={len(actors)}")
        return actors

    def toggle_wireframe(self):
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

    def toggle_pixelated(self, enabled: bool):
        self.pixelated = enabled
        acts = self.renderer.GetActors(); acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            t = a.GetTexture()
            if not t: continue
            try:
                if self.pixelated: t.InterpolateOff()
                else: t.InterpolateOn()
            except: pass
        self.statusBar().showMessage(f"Pixelated textures: {'ON' if self.pixelated else 'OFF'}")
        self.ren_win.Render()

    def cycle_background(self):
        self.bg_idx = (self.bg_idx + 1) % len(self.bg_colors)
        self.set_background(*self.bg_colors[self.bg_idx])

    def set_background(self, r, g, b):
        self.renderer.SetBackground(r, g, b)
        self.ren_win.Render()

    def reset_view(self):
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.ren_win.Render()

    # ---------- Info ----------
    def update_info(self):
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
    import sys
    app = QApplication(sys.argv)
    w = ViewerApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

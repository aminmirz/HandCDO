import os
import sys
import numpy as np
import bpy  # type: ignore
import bmesh  # type: ignore
import math
from scipy.spatial import cKDTree  # type: ignore

mm_to_m = 0.001


def create_pad_cube_colliders(pad_obj, collider_base_name, scene_collection=None, sub_collection=None):
    """
    Create triangular prism colliders for a pad mesh by dividing it into small prisms.
    
    Each quad face of the bottom layer is split into two triangles along a diagonal.
    The diagonal is chosen to connect to the vertex with the largest z value on the 
    top layer (i.e., the highest point after bump displacement). Each triangle is then
    extruded vertically to connect with the corresponding triangle on the top layer,
    creating a triangular prism collider.
    
    Corresponding vertices of top and bottom layers have the same x,y values 
    (within ~0.001mm tolerance) in the local coordinate frame of the pad.
    
    Args:
        pad_obj: The viz_pad_ Blender object
        collider_base_name: Base name for colliders (e.g. "collider_pad_PalmBody")
        scene_collection: Scene collection to link colliders to (if sub_collection is None)
        sub_collection: Sub-collection to link colliders to (preferred)
    
    Returns:
        List of created collider objects
    """
    if pad_obj is None or pad_obj.type != "MESH":
        return []
    
    # Get collider material
    coll_mat = bpy.data.materials.get("colliders")
    
    # Create bmesh from pad mesh
    bm = bmesh.new()
    bm.from_mesh(pad_obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    if len(bm.verts) == 0 or len(bm.faces) == 0:
        bm.free()
        return []
    
    # Identify top and bottom surfaces by z-coordinate
    all_z = [v.co.z for v in bm.verts]
    z_min = min(all_z)
    z_max = max(all_z)
    z_range = z_max - z_min
    
    if z_range < 1e-9:
        # Flat mesh, can't create cubes
        bm.free()
        return []
    
    # Tolerance for identifying top/bottom vertices (0.001mm in local coords)
    z_tol = max(z_range * 0.1, 1e-6)  # 10% of height or 1e-6m
    z_tol = 0.1 * mm_to_m  # 0.1 mm
    
    # Classify vertices as top, bottom, or side
    bottom_verts = [v for v in bm.verts if v.co.z <= z_min + z_tol]
    top_verts = [v for v in bm.verts if v.co.z >= z_min + z_tol]
    
    if len(bottom_verts) == 0 or len(top_verts) == 0:
        bm.free()
        return []
    
    # Build KDTree for top vertices using x,y coordinates
    top_xy = np.array([[v.co.x, v.co.y] for v in top_verts])
    top_indices = {i: v for i, v in enumerate(top_verts)}
    top_tree = cKDTree(top_xy)
    
    # Map bottom vertices to their corresponding top vertices
    bottom_to_top = {}
    xy_tol = 0.001 * mm_to_m  # 0.001mm tolerance for xy matching
    
    for bv in bottom_verts:
        bxy = np.array([bv.co.x, bv.co.y])
        dist, idx = top_tree.query(bxy, k=1)
        if dist <= xy_tol:
            bottom_to_top[bv.index] = top_indices[idx]
    
    # Find bottom faces (all vertices are bottom vertices)
    bottom_vert_set = set(v.index for v in bottom_verts)
    bottom_faces = []
    for f in bm.faces:
        if all(v.index in bottom_vert_set for v in f.verts):
            bottom_faces.append(f)
    
    # Create triangular prism colliders (each quad split into 2 triangular prisms along diagonal)
    colliders = []
    collider_idx = 0
    
    for bottom_face in bottom_faces:
        # Get bottom face vertices in order
        bottom_face_verts = list(bottom_face.verts)
        
        # Find corresponding top vertices
        top_face_verts = []
        valid = True
        for bv in bottom_face_verts:
            if bv.index in bottom_to_top:
                top_face_verts.append(bottom_to_top[bv.index])
            else:
                valid = False
                break
        
        if not valid or len(top_face_verts) != len(bottom_face_verts):
            continue
        
        n_verts = len(bottom_face_verts)
        
        # For quads (4 vertices), split into 2 triangular prisms along diagonal
        # For triangles (3 vertices), create 1 triangular prism
        # For other polygons, triangulate by fan from the vertex with largest z on top layer
        if n_verts < 3:
            continue
        
        # Find the index of the top layer vertex with the largest z value
        # This vertex will be the pivot for diagonal splitting (fan triangulation)
        max_z_idx = 0
        max_z_val = top_face_verts[0].co.z
        for i, tv in enumerate(top_face_verts):
            if tv.co.z > max_z_val:
                max_z_val = tv.co.z
                max_z_idx = i
        
        # Generate triangle index sets (fan triangulation from the highest z vertex)
        # Reorder indices so that max_z_idx is the pivot (first vertex in each triangle)
        triangles = []
        for i in range(1, n_verts - 1):
            # Map indices relative to max_z_idx
            idx0 = max_z_idx
            idx1 = (max_z_idx + i) % n_verts
            idx2 = (max_z_idx + i + 1) % n_verts
            triangles.append((idx0, idx1, idx2))
        
        for tri_indices in triangles:
            # Create triangular prism mesh
            prism_bm = bmesh.new()
            
            # Add bottom triangle vertices
            prism_bottom_verts = []
            for idx in tri_indices:
                bv = bottom_face_verts[idx]
                nv = prism_bm.verts.new((bv.co.x, bv.co.y, bv.co.z))
                prism_bottom_verts.append(nv)
            
            # Add top triangle vertices (same order)
            prism_top_verts = []
            for idx in tri_indices:
                tv = top_face_verts[idx]
                nv = prism_bm.verts.new((tv.co.x, tv.co.y, tv.co.z))
                prism_top_verts.append(nv)
            
            prism_bm.verts.ensure_lookup_table()
            
            try:
                # Bottom face (reversed winding for outward normal pointing down)
                prism_bm.faces.new(list(reversed(prism_bottom_verts)))
                
                # Top face
                prism_bm.faces.new(prism_top_verts)
                
                # 3 side faces (quads connecting bottom and top edges)
                for i in range(3):
                    i_next = (i + 1) % 3
                    side_verts = [
                        prism_bottom_verts[i],
                        prism_bottom_verts[i_next],
                        prism_top_verts[i_next],
                        prism_top_verts[i],
                    ]
                    prism_bm.faces.new(side_verts)
            except ValueError:
                # Face creation failed (e.g., degenerate geometry)
                prism_bm.free()
                continue
            
            prism_bm.normal_update()
            
            # Create mesh and object
            collider_name = f"{collider_base_name}_{collider_idx}"
            prism_mesh = bpy.data.meshes.new(f"{collider_name}_mesh")
            prism_bm.to_mesh(prism_mesh)
            prism_bm.free()
            
            coll_obj = bpy.data.objects.new(collider_name, prism_mesh)
            
            # Link to collection
            if sub_collection is not None:
                try:
                    sub_collection.objects.link(coll_obj)
                except Exception:
                    if scene_collection is not None:
                        scene_collection.objects.link(coll_obj)
            elif scene_collection is not None:
                scene_collection.objects.link(coll_obj)
            else:
                bpy.context.scene.collection.objects.link(coll_obj)
            
            # Parent to pad object and align with pad's world transform
            coll_obj.parent = pad_obj
            coll_obj.matrix_parent_inverse = pad_obj.matrix_world.inverted()
            coll_obj.matrix_world = pad_obj.matrix_world.copy()
            
            # Apply collider material
            if coll_mat is not None and coll_obj.data is not None:
                coll_obj.data.materials.clear()
                coll_obj.data.materials.append(coll_mat)
            
            colliders.append(coll_obj)
            collider_idx += 1
    
    bm.free()
    return colliders

class PalmMesh:
    def __init__(self, data_path: str):
        self.name = "PalmBody"
        self.viz_name = f"viz_{self.name}"
        self.root_name = f"{self.name}_root"
        self.pad_name = f"viz_pad_{self.name}"
        self.pad_collider_name = f"collider_pad_{self.name}"
        self.data_path = data_path
        self.data = np.load(data_path, allow_pickle=True)
        self.outline = np.array(self.data["outline"], dtype=float)
        self.inward_offset_outline = np.array(self.data["inward_offset_outline"], dtype=float)
        self.thumb_squares = np.array(self.data["thumb_squares"], dtype=float)

        self.palm_thickness_m = float(self.data.get("palm_thickness_mm", None)) * mm_to_m
        self.palm_wall_thickness_m = float(self.data.get("palm_wall_thickness_mm", None)) * mm_to_m
        self.palm_extrude_height_m = float(self.data.get("palm_extrude_height_mm", None)) * mm_to_m
        self.finger_palm_base_offset_m = float(self.data.get("finger_palm_base_offset_mm", None)) * mm_to_m

        self.thumb_mount_size_m = np.array(self.data.get("thumb_mount_size_mm", None), dtype=float) * mm_to_m
        self.thumb_mount_normal_offset_m = float(self.data.get("thumb_mount_normal_offset_mm", None)) * mm_to_m
        self.thumb_mount_height_m = float(self.data.get("thumb_mount_height_mm", None)) * mm_to_m

        self.palm_pad_resolution_level = int(self.data.get("palm_pad_resolution_level", 7))
        self.palm_bump_edge_zeroing_spread = float(self.data.get("palm_bump_edge_zeroing_spread", 0.3))
        
        # Load palm bump config parameters
        self.palm_bump_type = str(self.data.get("palm_bump_type", "gaussian"))
        self.palm_bumps_number = int(self.data.get("palm_bumps_number", 0))
        self.palm_bump_max_height_mm = float(self.data.get("palm_bump_max_height_intensity_mm", 0.0))
        self.palm_bump_height_intensity_list = np.array(self.data.get("palm_bump_height_intensity_list", []), dtype=float)
        self.palm_bumps_spread_list = np.array(self.data.get("palm_bumps_spread_list", []), dtype=float)
        self.palm_bumps_aspect_ratio_list = np.array(self.data.get("palm_bumps_aspect_ratio_list", []), dtype=float)
        self.palm_bump_rotation_deg_list = np.array(self.data.get("palm_bump_rotation_deg_list", []), dtype=float)
        self.palm_bump_center_angle_deg_list = np.array(self.data.get("palm_bump_center_angle_deg_list", []), dtype=float)
        self.palm_bump_center_offset_list = np.array(self.data.get("palm_bump_center_offset_list", []), dtype=float)
        
        # Load detailed viz flag (controls whether screw holes, mounts, etc. are added)
        self.detailed_viz = bool(self.data.get("detailed_viz", False))
        cm = self.data.get("collision_mesh", None)
        self.collision_mesh = bool(cm) if cm is not None else False
        
        if self.outline.ndim != 2 or self.outline.shape[1] != 2 or self.outline.shape[0] < 3:
            print("Invalid palm outline data; need at least 3 points with XY coordinates.")
            return

    def create_main_extruded_mesh(self):
        """
        create a 3D palm main body mesh in Blender
        by extruding the outline with a given thickness
        """

        # Build a 2D polygon from the outline at Z = 0 and extrude it along +Z
        bm = bmesh.new()
        verts = []
        for x_mm, y_mm in self.outline:
            v = bm.verts.new((float(x_mm) * mm_to_m, float(y_mm) * mm_to_m, 0.0))
            verts.append(v)

        bm.verts.ensure_lookup_table()
        try:
            face = bm.faces.new(verts)
        except ValueError:
            # If a face with these verts already exists, reuse it
            face = bm.faces.get(verts)

        bm.normal_update()

        # Extrude the polygon face to create a solid body
        res = bmesh.ops.extrude_face_region(bm, geom=[face])
        geom_extruded = res.get("geom", [])
        extruded_verts = [elem for elem in geom_extruded if isinstance(elem, bmesh.types.BMVert)]
        if extruded_verts:
            bmesh.ops.translate(bm, verts=extruded_verts, vec=(0.0, 0.0, self.palm_extrude_height_m))

        mesh = bpy.data.meshes.new(self.viz_name)
        bm.to_mesh(mesh)
        bm.free()

        palm_obj = bpy.data.objects.new(self.viz_name, mesh)

        # Create a plain-axes empty as the top-level PalmBody root
        palm_root = bpy.data.objects.get(self.root_name)
        if palm_root is None:
            palm_root = bpy.data.objects.new(self.root_name, None)
            palm_root.empty_display_type = 'PLAIN_AXES'
            bpy.context.scene.collection.objects.link(palm_root)
        # Place root where the palm mesh is, then parent the palm mesh under it
        palm_root.matrix_world = palm_obj.matrix_world.copy()

        scene_col = bpy.context.scene.collection
        scene_col.objects.link(palm_obj)
        palm_obj.parent = palm_root
        palm_obj.matrix_parent_inverse = palm_root.matrix_world.inverted()

        # Assign "viz" material to palm body
        viz_mat = bpy.data.materials.get("viz")
        if viz_mat is not None and palm_obj.data is not None:
            palm_obj.data.materials.clear()
            palm_obj.data.materials.append(viz_mat)

        self.merge_vertices_by_distance()
        self.remove_inside_mesh()
        self.merge_vertices_by_distance()
        self.remove_thumb_squares()
        self.merge_vertices_by_distance()
        self.decimate_modifier()
        self.add_attachment_geometries()
        # self.palm_z_offset()

        
        # # Collider for main palm viz
        # coll_palm = bpy.data.objects.new(f"collider_{self.name}", palm_obj.data.copy())
        # scene_col.objects.link(coll_palm)
        # coll_palm.parent = palm_obj
        # coll_palm.matrix_parent_inverse = palm_obj.matrix_world.inverted()
        coll_mat = bpy.data.materials.get("colliders")
        # if coll_mat is not None and coll_palm.data is not None:
        #     coll_palm.data.materials.clear()
        #     coll_palm.data.materials.append(coll_mat)

        self.recalculate_face_normals(palm_obj)

        # Create palm pad: outline extruded 3 mm, starting at top surface + 2 mm, along world +Z
        if bpy.data.objects.get(self.pad_name) is None:
            pad_bm = bmesh.new()
            verts = []
            z0 = float(self.palm_extrude_height_m)
            for x_mm, y_mm in self.outline:
                v = pad_bm.verts.new((float(x_mm) * mm_to_m, float(y_mm) * mm_to_m, z0))
                verts.append(v)
            pad_bm.verts.ensure_lookup_table()
            try:
                face = pad_bm.faces.new(verts)
            except ValueError:
                face = pad_bm.faces.get(verts)
            pad_bm.normal_update()
            res_pad = bmesh.ops.extrude_face_region(pad_bm, geom=[face])
            extruded_pad = [e for e in res_pad.get("geom", []) if isinstance(e, bmesh.types.BMVert)]
            if extruded_pad:
                bmesh.ops.translate(pad_bm, verts=extruded_pad, vec=(0.0, 0.0, 0.001))  # 1 mm

            pad_mesh = bpy.data.meshes.new(f"{self.pad_name}_mesh")
            pad_bm.to_mesh(pad_mesh)
            pad_bm.free()

            pad_obj = bpy.data.objects.new(self.pad_name, pad_mesh)
            scene_col.objects.link(pad_obj)
            pad_obj.parent = palm_root
            pad_obj.matrix_parent_inverse = palm_root.matrix_world.inverted()

            # Remesh palm pad
            try:
                pad_obj.select_set(True)
                bpy.context.view_layer.objects.active = pad_obj
                rem = pad_obj.modifiers.new(name="PadRemesh", type="REMESH")
                rem.mode = 'SHARP'
                rem.octree_depth = int(self.palm_pad_resolution_level)
                rem.use_smooth_shade = True
                bpy.ops.object.modifier_apply(modifier=rem.name)
            except Exception as e:
                print(f"[pad][warn] Failed to remesh palm pad: {e}")

            # Displace only the top (extruded) surface along local +Z using palm bump params
            def _bump_height_mm(bump_pack, x_mm, y_mm):
                if bump_pack is None:
                    return 0.0
                # bump_pack can be list (legacy) or dict with bumps/meta
                if isinstance(bump_pack, dict):
                    bumps = bump_pack.get("bumps", [])
                    meta = bump_pack.get("meta", {})
                    raw_hmin = float(meta.get("raw_hmin", 0.0))
                    raw_hmax = float(meta.get("raw_hmax", 0.0))
                    scale = float(meta.get("scale", 0.0))
                else:
                    bumps = bump_pack
                    raw_hmin = 0.0
                    raw_hmax = 0.0
                    scale = 0.0
                raw = 0.0
                for bp in bumps or []:
                    try:
                        c = bp.get("center", (0.0, 0.0))
                        sigma_long = float(bp.get("sigma_long", 1.0))
                        sigma_short = float(bp.get("sigma_short", 1.0))
                        theta = float(bp.get("theta", 0.0))
                        amp = float(bp.get("amp", 0.0))
                        ct = math.cos(theta)
                        st = math.sin(theta)
                        dx = x_mm - float(c[0])
                        dy = y_mm - float(c[1])
                        xr = ct * dx + st * dy
                        yr = -st * dx + ct * dy
                        raw += amp * math.exp(-0.5 * ((xr / sigma_long) ** 2 + (yr / sigma_short) ** 2))
                    except Exception:
                        continue
                if scale > 0.0 and raw_hmax > raw_hmin:
                    return (raw - raw_hmin) / max(raw_hmax - raw_hmin, 1e-9) * scale
                return raw

            def _displace_top_surface_local_z(obj, bump_pack, edge_zeroing_spread: float):
                if obj is None or obj.type != "MESH":
                    return
                bm2 = bmesh.new()
                bm2.from_mesh(obj.data)
                if not bm2.verts:
                    bm2.free()
                    return
                bm2.verts.ensure_lookup_table()
                zmax = max(v.co.z for v in bm2.verts)
                tol = 1e-6
                top_verts = [v for v in bm2.verts if v.co.z >= zmax - tol]
                top_set = set(top_verts)

                # Boundary verts: top-surface verts that connect to any non-top vert (side walls)
                boundary = []
                for v in top_verts:
                    for e in v.link_edges:
                        ov = e.other_vert(v)
                        if ov not in top_set:
                            boundary.append(v)
                            break

                # Compute a smooth weight field from boundary (0) to interior (1)
                weights = {v: 1.0 for v in top_verts}
                spread_ratio = float(edge_zeroing_spread) if edge_zeroing_spread is not None else 0.0
                if boundary and spread_ratio > 0.0:
                    import heapq
                    dist = {v: 0.0 for v in boundary}
                    # heapq requires items to be comparable on ties; BMVert is not comparable,
                    # so we include a stable tiebreaker (vertex index).
                    heap = [(0.0, int(v.index), v) for v in boundary]
                    heapq.heapify(heap)
                    while heap:
                        d, _vi, v = heapq.heappop(heap)
                        if d != dist.get(v, None):
                            continue
                        for e in v.link_edges:
                            ov = e.other_vert(v)
                            if ov not in top_set:
                                continue
                            # edge length on top surface (meters)
                            w = (v.co - ov.co).length
                            nd = d + float(w)
                            if nd < dist.get(ov, 1e30):
                                dist[ov] = nd
                                heapq.heappush(heap, (nd, int(ov.index), ov))

                    max_d = max(dist.get(v, 0.0) for v in top_verts) if top_verts else 0.0
                    spread_d = max(1e-12, float(spread_ratio) * float(max_d))
                    for v in top_verts:
                        d = float(dist.get(v, 0.0))
                        t = max(0.0, min(1.0, d / spread_d))
                        # smoothstep
                        weights[v] = t * t * (3.0 - 2.0 * t)
                elif boundary:
                    # edge_zeroing_spread <= 0 => hard zero at boundary only
                    for v in boundary:
                        weights[v] = 0.0

                for v in top_verts:
                    x_mm = float(v.co.x) / mm_to_m
                    y_mm = float(v.co.y) / mm_to_m
                    dz_mm = _bump_height_mm(bump_pack, x_mm, y_mm) * float(weights.get(v, 1.0))
                    v.co.z += dz_mm * mm_to_m
                bm2.to_mesh(obj.data)
                bm2.free()

            def _generate_bump_pack_from_config(
                bump_type: str,
                num_bumps: int,
                max_height_mm: float,
                height_intensity_list: np.ndarray,
                spread_list: np.ndarray,
                aspect_ratio_list: np.ndarray,
                rotation_deg_list: np.ndarray,
                center_angle_deg_list: np.ndarray,
                center_offset_list: np.ndarray,
                surface_radius_mm: float = 50.0
            ):
                """
                Generate bump_pack dictionary from config parameters.
                Each bump has: center, sigma_long, sigma_short, theta, amp, type
                """
                bumps = []
                for i in range(num_bumps):
                    if i >= len(height_intensity_list):
                        break
                    # Calculate bump center from angle and offset
                    angle_rad = np.deg2rad(center_angle_deg_list[i]) if i < len(center_angle_deg_list) else 0.0
                    offset_scaled = center_offset_list[i] if i < len(center_offset_list) else 0.0
                    center_r = offset_scaled * surface_radius_mm
                    cx = center_r * np.cos(angle_rad)
                    cy = center_r * np.sin(angle_rad)
                    
                    # Calculate sigma from spread (scaled to surface size)
                    spread = spread_list[i] if i < len(spread_list) else 0.25
                    sigma_base = spread * surface_radius_mm
                    
                    # Apply aspect ratio
                    aspect = aspect_ratio_list[i] if i < len(aspect_ratio_list) else 1.0
                    sigma_long = sigma_base * np.sqrt(aspect)
                    sigma_short = sigma_base / np.sqrt(aspect)
                    
                    # Rotation angle
                    theta = np.deg2rad(rotation_deg_list[i]) if i < len(rotation_deg_list) else 0.0
                    
                    # Amplitude (height intensity * max height)
                    intensity = height_intensity_list[i] if i < len(height_intensity_list) else 1.0
                    amp = intensity * max_height_mm
                    
                    bumps.append({
                        "center": (float(cx), float(cy)),
                        "sigma_long": float(sigma_long),
                        "sigma_short": float(sigma_short),
                        "theta": float(theta),
                        "amp": float(amp),
                        "type": str(bump_type),
                    })
                
                # Calculate raw min/max for normalization
                raw_hmax = max_height_mm if bumps else 0.0
                raw_hmin = 0.0
                
                return {
                    "bumps": bumps,
                    "meta": {
                        "raw_hmin": raw_hmin,
                        "raw_hmax": raw_hmax,
                        "scale": max_height_mm,
                    }
                }

            # Calculate approximate palm surface radius
            palm_radius_mm = np.max(np.linalg.norm(self.outline, axis=1)) if len(self.outline) > 0 else 50.0
            
            # Generate palm bump pack from config parameters
            palm_pack = _generate_bump_pack_from_config(
                bump_type=self.palm_bump_type,
                num_bumps=self.palm_bumps_number,
                max_height_mm=self.palm_bump_max_height_mm,
                height_intensity_list=self.palm_bump_height_intensity_list,
                spread_list=self.palm_bumps_spread_list,
                aspect_ratio_list=self.palm_bumps_aspect_ratio_list,
                rotation_deg_list=self.palm_bump_rotation_deg_list,
                center_angle_deg_list=self.palm_bump_center_angle_deg_list,
                center_offset_list=self.palm_bump_center_offset_list,
                surface_radius_mm=palm_radius_mm,
            )
            _displace_top_surface_local_z(pad_obj, palm_pack, self.palm_bump_edge_zeroing_spread)

            # Create cube colliders for palm pad
            if self.collision_mesh:
                create_pad_cube_colliders(
                    pad_obj=pad_obj,
                    collider_base_name=self.pad_collider_name,
                    scene_collection=scene_col,
                    sub_collection=None,
                )

    def remove_inside_mesh(self):
        """
        make a boolean operation to remove the inside of the main extruded mesh by the inward offset outline and specified thickness
        """
        # Validate inward offset outline
        inner = np.asarray(self.inward_offset_outline, dtype=float)
        if inner.ndim != 2 or inner.shape[1] != 2 or inner.shape[0] < 3:
            print("No valid inward offset outline available for boolean cut.")
            return

        # Try to find the main palm body object
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None:
            # Fallback: take the first mesh object in the scene
            mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
            if not mesh_objs:
                print(f"No {self.viz_name} mesh found to apply boolean on.")
                return
            palm_obj = mesh_objs[0]

        # Create cutter mesh from inward offset outline
        bm = bmesh.new()
        verts = []
        for x_mm, y_mm in inner:
            v = bm.verts.new((float(x_mm) * mm_to_m, float(y_mm) * mm_to_m, 0.0))
            verts.append(v)

        bm.verts.ensure_lookup_table()
        try:
            face = bm.faces.new(verts)
        except ValueError:
            face = bm.faces.get(verts)

        bm.normal_update()

        # Extrude cutter to at least the main thickness to ensure intersection
        height_m = self.palm_extrude_height_m - self.palm_thickness_m
        res = bmesh.ops.extrude_face_region(bm, geom=[face])
        geom_extruded = res.get("geom", [])
        extruded_verts = [elem for elem in geom_extruded if isinstance(elem, bmesh.types.BMVert)]
        if extruded_verts:
            bmesh.ops.translate(bm, verts=extruded_verts, vec=(0.0, 0.0, height_m))

        cutter_mesh = bpy.data.meshes.new("PalmInnerCutter")
        bm.to_mesh(cutter_mesh)
        bm.free()

        cutter_obj = bpy.data.objects.new("PalmInnerCutter", cutter_mesh)
        bpy.context.scene.collection.objects.link(cutter_obj)

        self.recalculate_face_normals(cutter_obj)

        # Apply Boolean difference: PalmBody minus cutter
        # Ensure correct active object
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj

        bool_mod = palm_obj.modifiers.new(name="InnerCut", type="BOOLEAN")
        bool_mod.operation = "DIFFERENCE"
        bool_mod.solver = "EXACT"
        bool_mod.object = cutter_obj

        try:
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        except Exception as e:
            print(f"Failed to apply boolean modifier: {e}")
        finally:
            # Remove cutter object from scene and data
            try:
                bpy.data.objects.remove(cutter_obj, do_unlink=True)
            except Exception:
                pass

        self.recalculate_face_normals(palm_obj)

    def remove_thumb_squares(self):
        """
        make a boolean operation to remove the main extruded mesh by the thumb squares meshes and specified extrusion-cutheight
        """
        squares = np.asarray(self.thumb_squares, dtype=float)
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None:
            print(f"No {self.viz_name} mesh found to apply thumb square booleans on.")
            return

        # For each square outline, create a cutter prism and subtract it
        # Make the cutter extend both below and above the base plane to avoid
        # coplanar boolean issues (everything still extrudes mainly in +Z).
        height_m = self.palm_extrude_height_m - self.palm_thickness_m
        for idx, sq in enumerate(squares):
            # Ensure closed loop by ignoring potential duplicate last point
            pts = np.asarray(sq, dtype=float)
            if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if pts.shape[0] < 4:
                continue

            
            bm = bmesh.new()
            verts = []
            for x_mm, y_mm in pts:
                # Start slightly below the base plane so the cutter fully
                # passes through the PalmBody volume.
                v = bm.verts.new((float(x_mm) * mm_to_m, float(y_mm) * mm_to_m, -height_m))
                verts.append(v)

            bm.verts.ensure_lookup_table()
            try:
                face = bm.faces.new(verts)
            except ValueError:
                face = bm.faces.get(verts)

            bm.normal_update()

            res = bmesh.ops.extrude_face_region(bm, geom=[face])
            geom_extruded = res.get("geom", [])
            extruded_verts = [elem for elem in geom_extruded if isinstance(elem, bmesh.types.BMVert)]
            if extruded_verts:
                # Extrude upwards so cutter spans roughly [-height_m, +height_m]
                bmesh.ops.translate(bm, verts=extruded_verts, vec=(0.0, 0.0, 2.0 * height_m))

            cutter_mesh = bpy.data.meshes.new(f"ThumbSquareCutter_{idx}")
            bm.to_mesh(cutter_mesh)
            bm.free()

            cutter_obj = bpy.data.objects.new(f"ThumbSquareCutter_{idx}", cutter_mesh)
            bpy.context.scene.collection.objects.link(cutter_obj)

            self.recalculate_face_normals(cutter_obj)

            # Apply Boolean difference for this square
            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"ThumbCut_{idx}", type="BOOLEAN")
            bool_mod.operation = "DIFFERENCE"
            bool_mod.solver = "EXACT"
            bool_mod.object = cutter_obj

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply thumb square boolean modifier {idx}: {e}")
            finally:
                try:
                    bpy.data.objects.remove(cutter_obj, do_unlink=True)
                except Exception:
                    pass

            self.recalculate_face_normals(palm_obj)

    def recalculate_face_normals(self, obj):
        """
        recalculate the face normals of the given object outward-facing
        """
        if obj is None or obj.type != "MESH":
            print("recalculate_face_normals: provided object is not a mesh or is None.")
            return

        # Make sure object is active and in edit mode
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if bpy.context.object.mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")

        # Select all faces and recalculate normals outside
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            # Blender 2.8+:
            bpy.ops.mesh.normals_make_consistent(inside=False)
        except Exception:
            # Fallback for very old versions if needed
            pass

        # Return to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

    def merge_vertices_by_distance(self):
        """
        merge the main vertices by a given distance default 0.001 mm
        """
        # Merge vertices on the main palm body object to clean up small gaps
        # Default threshold: 0.001 mm -> converted to meters
        merge_dist_mm = 0.001
        merge_dist_m = merge_dist_mm * mm_to_m

        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to merge vertices on.")
            return

        # Make sure PalmBody is active and in object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")

        # Use mesh.select_mode to ensure vertices are selected
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")

        # Merge by distance (or "remove doubles" in older Blender)
        # Some Blender versions don't have mesh.merge_by_distance at all.
        if hasattr(bpy.ops.mesh, "merge_by_distance"):
            try:
                bpy.ops.mesh.merge_by_distance(distance=merge_dist_m)
            except Exception as e:
                bpy.ops.mesh.remove_doubles(threshold=merge_dist_m)
        else:
            bpy.ops.mesh.remove_doubles(threshold=merge_dist_m)

        # Return to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

    def decimate_modifier(self):
        """
        apply a decimate modifier to the main palm body object, planar and angle of 0.1d
        """
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to apply decimate modifier on.")
            return

        # Ensure object mode and correct active object
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Add a planar decimate (DISSOLVE) modifier with small angle limit
        mod = palm_obj.modifiers.new(name="PalmPlanarDecimate", type="DECIMATE")
        mod.decimate_type = "DISSOLVE"  # planar/angle-based decimation
        # 0.1 degrees in radians
        mod.angle_limit = 0.1 * (3.141592653589793 / 180.0)

        # Apply the modifier
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            print(f"Failed to apply decimate modifier on {self.viz_name} mesh: {e}")

    def add_attachment_geometries(self):
        """
        Add attachment geometries to the palm body model.
        The attachment geometries are the cubes that are used to attach the fingers and thumbs to the palm body.
        size of the cubes is 15x15x5 mm (x,y,z) and initial position is (0.0, 0.0, 0.0).
        the cubes are transfrmed to the finger and thumb bases locations and normal orientations then have a z offset of self.palm_extrude_height_m - self.palm_thickness_m - self.finger_palm_base_offset_m
        The attachment geometries are child of the Palm body object
        """
        palm_obj = bpy.data.objects.get(self.viz_name)
        palm_root = bpy.data.objects.get(self.root_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to attach geometries to.")
            return
        palm_parent = palm_root if palm_root is not None else palm_obj

        # Read bases and normals (in mm, 2D) from NPZ if present
        finger_bases_2d = self.data.get("finger_bases_2d")
        finger_normals_2d = self.data.get("finger_normals_2d")
        thumb_bases_2d = self.data.get("thumb_bases_2d")
        thumb_normals_2d = self.data.get("thumb_normals_2d")

        def _safe_array(arr):
            return np.array(arr, dtype=float) if arr is not None else np.zeros((0, 2), dtype=float)

        finger_bases_2d = _safe_array(finger_bases_2d)
        finger_normals_2d = _safe_array(finger_normals_2d)
        thumb_bases_2d = _safe_array(thumb_bases_2d)
        thumb_normals_2d = _safe_array(thumb_normals_2d)

        # Cube dimensions in meters (size 15x15x5 mm)
        size_x = 15.0 * mm_to_m
        size_y = 15.0 * mm_to_m
        size_z = 5.0 * mm_to_m

        # Z offset for cube centers
        z_offset = self.palm_extrude_height_m - self.palm_thickness_m - self.finger_palm_base_offset_m

        def _unit(v):
            v = np.asarray(v, dtype=float)
            n = float(np.linalg.norm(v))
            return v / n if n > 0.0 else v

        # Get attachment material (if it exists)
        attach_mat = bpy.data.materials.get("attach_free")

        def _make_cubes(bases_2d, normals_2d, prefix: str):
            for i in range(min(len(bases_2d), len(normals_2d))):
                base_xy_mm = np.asarray(bases_2d[i], dtype=float)
                n_xy = _unit(normals_2d[i])
                bx, by = float(base_xy_mm[0]) * mm_to_m, float(base_xy_mm[1]) * mm_to_m

                # Rotation around Z so that local +Y aligns with the base normal
                angle_z = math.atan2(float(n_xy[1]), float(n_xy[0]))

                # Create cube at origin, then transform
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
                cube = bpy.context.active_object
                if cube is None:
                    continue
                cube.name = f"{prefix}_{i+1}"

                # Set exact dimensions and apply scale so the cube is truly 15x15x5 mm
                cube.dimensions = (size_x, size_y, size_z)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                # Rotate so that the cube's local +Z axis aligns with the base normal in XY:
                # 1) Rotate 90 deg around Y so Z->X
                # 2) Rotate around Z by angle_z to align X with the normal direction.
                cube.rotation_euler = (0.0, math.pi / 2.0, angle_z)

                cube.location = (bx, by, z_offset)

                # Assign attachment material if present
                if attach_mat is not None and cube.data is not None:
                    # Clear existing materials and assign attach_free
                    cube.data.materials.clear()
                    cube.data.materials.append(attach_mat)

                # Parent to palm root (or palm mesh fallback)
                cube.parent = palm_parent

        _make_cubes(finger_bases_2d, finger_normals_2d, "attach_finger")
        _make_cubes(thumb_bases_2d, thumb_normals_2d, "attach_thumb")

    def palm_z_offset(self):
        """
        move the palm object along the z axis by the -palm_extrude_height_m
        """
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to apply Z offset on.")
            return

        # Ensure object mode and correct active object
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Translate along Z by -palm_extrude_height_m
        current_loc = palm_obj.location
        palm_obj.location = (
            current_loc.x,
            current_loc.y,
            current_loc.z - float(self.palm_extrude_height_m),
        )

    def offset_all_finger_and_thumb_components(self):
        """
        move all the finger and thumb PARENT components along the z axis by the palm_extrude_height_m - palm_thickness_m - finger_palm_base_offset_m
        """
        # Compute Z offset in meters
        offset_z = float(self.palm_extrude_height_m - self.palm_thickness_m - self.finger_palm_base_offset_m)
        # print(f"offset_z: {offset_z}")
        if offset_z == 0.0:
            return

        hand_col = bpy.data.collections.get("hand")
        if hand_col is None:
            print("offset_all_finger_and_thumb_components: 'hand' collection not found; nothing to offset.")
            return

        def _offset_collection(col):
            # Offset only parent mesh objects in this collection; children will follow
            for obj in col.objects:
                # Only move parent components (no Blender parent)
                if obj.parent is not None:
                    continue
                loc = obj.location
                obj.location = (loc.x, loc.y, loc.z + offset_z)

            # Recurse into child collections
            for child in col.children:
                _offset_collection(child)

        _offset_collection(hand_col)

    def add_thumb_base_mount(self):
        """
        add the thumb base mount to the palm body model.
        the thumb base mount is a cube that is used to attach the thumb to the palm body.
        size of the cube is 25x7 mm (x,y) and z is self.palm_extrude_height_m - self.palm_thickness_m and the initial position is (0.0, 0.0, 0.0).
        the cube is transfrmed to the thumb base locations with 2.5 mm offset in the opposite direction of the normal vectr
        and 5 mm axis is aligned with the normal vector of the thumb base.
        The thumb base mount is then merged with the palm body object.
        """
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to add thumb base mount to.")
            return

        # Read thumb bases and normals (in mm, 2D) from NPZ
        thumb_bases_2d = self.data.get("thumb_bases_2d")
        thumb_normals_2d = self.data.get("thumb_normals_2d")

        def _safe_array(arr):
            return np.array(arr, dtype=float) if arr is not None else np.zeros((0, 2), dtype=float)

        thumb_bases_2d = _safe_array(thumb_bases_2d)
        thumb_normals_2d = _safe_array(thumb_normals_2d)

        if len(thumb_bases_2d) == 0 or len(thumb_normals_2d) == 0:
            print("No thumb bases or normals found; skipping thumb base mount.")
            return

        # Cube dimensions in meters: 25x7 mm (x,y), z = palm_extrude_height - palm_thickness
        size_x = self.thumb_mount_size_m[0]
        size_y = self.thumb_mount_size_m[1]
        normal_offset = -self.thumb_mount_normal_offset_m
        size_z = self.thumb_mount_height_m

        def _unit(v):
            v = np.asarray(v, dtype=float)
            n = float(np.linalg.norm(v))
            return v / n if n > 0.0 else v

        # Ensure we are in object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        for i in range(min(len(thumb_bases_2d), len(thumb_normals_2d))):
            base_xy_mm = np.asarray(thumb_bases_2d[i], dtype=float)
            n_xy = _unit(thumb_normals_2d[i])
            # Apply 7.0/2.0 mm offset in opposite direction of normal
            bx = float(base_xy_mm[0]) * mm_to_m + normal_offset * float(n_xy[0])
            by = float(base_xy_mm[1]) * mm_to_m + normal_offset * float(n_xy[1])

            # Rotation around Z so that local +Y (short axis) aligns with the base normal
            angle_z = math.atan2(float(n_xy[1]), float(n_xy[0])) - math.pi / 2.0

            # Create cube at origin, then transform
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
            mount_cube = bpy.context.active_object
            if mount_cube is None:
                continue
            mount_cube.name = f"thumb_base_mount_{i}"

            # Set exact dimensions (20x5x(height-thickness) mm) and apply scale
            mount_cube.dimensions = (size_x, size_y, size_z)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            # Rotate so the short axis (Y) aligns with the normal direction
            mount_cube.rotation_euler = (0.0, 0.0, angle_z)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

            # Position the cube at the thumb base location, centered vertically
            mount_cube.location = (bx, by, size_z / 2.0)

            # Apply the location transform before boolean
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

            self.recalculate_face_normals(mount_cube)

            # Merge the mount cube with the palm body using Boolean union
            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"ThumbMount_{i}", type="BOOLEAN")
            bool_mod.operation = "UNION"
            bool_mod.solver = "EXACT"
            bool_mod.object = mount_cube

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply thumb base mount boolean modifier {i}: {e}")
            finally:
                # Remove the mount cube after merging
                try:
                    bpy.data.objects.remove(mount_cube, do_unlink=True)
                except Exception:
                    pass

            self.recalculate_face_normals(palm_obj)

        self.merge_vertices_by_distance()

    def add_screw_holes(self):
        """
        add the screw holes to the palm body model.
        there is a screw_holes object in the components.blend file 
        for each attach_ objects in the palm body model, the screw_holes object is duplicated and transformed to the attach_ object 3d pose.
        then, the screw_holes object is subtracted from the palm body model using a boolean modifier ( for every attach_ object in the palm body model)
        """
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to add screw holes to.")
            return

        # Find the screw_holes template object in the scene, or append it once from components.blend
        screw_holes_template = bpy.data.objects.get("screw_holes")
        if screw_holes_template is None:
            components_blend_path = os.path.join(os.path.dirname(__file__), "blender", "components.blend")
            if not os.path.exists(components_blend_path):
                print(f"components.blend not found at {components_blend_path}; skipping screw holes.")
                return

            current_blend = bpy.data.filepath
            try:
                samefile = current_blend and os.path.samefile(current_blend, components_blend_path)
            except Exception:
                samefile = False

            if not samefile:
                object_dir = os.path.join(components_blend_path, "Object")
                object_path = os.path.join(object_dir, "screw_holes")
                if not os.path.exists(object_path):
                    print("No 'screw_holes' object found in components.blend; skipping screw holes.")
                    return
                try:
                    bpy.ops.wm.append(
                        filepath=object_path,
                        directory=object_dir,
                        filename="screw_holes",
                        link=False,
                    )
                except Exception as e:
                    print(f"Failed to append 'screw_holes' from components.blend: {e}")
                    return

            screw_holes_template = bpy.data.objects.get("screw_holes")
            if screw_holes_template is None:
                print("No 'screw_holes' object available; skipping screw holes.")
                return

        # Ensure we are in object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Find all attach_ objects that are children of the palm body or palm root
        palm_root = bpy.data.objects.get(self.root_name)
        palm_parents = {palm_obj, palm_root} if palm_root else {palm_obj}
        attach_objects = [
            obj for obj in bpy.data.objects 
            if obj.name.startswith("attach_") and obj.parent in palm_parents
        ]
        if not attach_objects:
            print("No attach_ objects found as children of palm body; skipping screw holes.")
            return

        for idx, attach_obj in enumerate(attach_objects):
            # Duplicate the screw_holes template
            bpy.ops.object.select_all(action="DESELECT")
            screw_holes_template.select_set(True)
            bpy.context.view_layer.objects.active = screw_holes_template
            bpy.ops.object.duplicate(linked=False)
            screw_holes_copy = bpy.context.active_object
            if screw_holes_copy is None:
                continue
            screw_holes_copy.name = f"screw_holes_{idx}"

            # Reset the duplicate to identity transform first (bake any existing transform into mesh)
            bpy.ops.object.select_all(action="DESELECT")
            screw_holes_copy.select_set(True)
            bpy.context.view_layer.objects.active = screw_holes_copy
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            
            # Set origin to geometry center so it aligns properly with attach object
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

            # Now apply the attach_ object's pose
            screw_holes_copy.matrix_world = attach_obj.matrix_world.copy()

            # Apply the new transform
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            self.recalculate_face_normals(screw_holes_copy)

            # Subtract screw_holes from palm body using Boolean difference
            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"ScrewHoles_{idx}", type="BOOLEAN")
            bool_mod.operation = "DIFFERENCE"
            bool_mod.solver = "EXACT"
            bool_mod.object = screw_holes_copy

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply screw holes boolean modifier {idx}: {e}")
            finally:
                # Remove the screw_holes copy after subtracting
                try:
                    bpy.data.objects.remove(screw_holes_copy, do_unlink=True)
                except Exception:
                    pass

        # Remove the screw_holes template object after all holes are added
        try:
            bpy.data.objects.remove(screw_holes_template, do_unlink=True)
        except Exception:
            pass

        self.recalculate_face_normals(palm_obj)
        self.merge_vertices_by_distance()

    def add_hand_mount(self):
        '''
        add another viz_ object whcih should be the child of the palm body
        to make thee object, first we make a surface from the outline and we extrude it for 4 mm in the -z direction.
        note that before extrusion, we remove the modify the outine surface by removing the thuumb corner points.
        '''
        palm_obj = bpy.data.objects.get(self.viz_name)
        palm_root = bpy.data.objects.get(self.root_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found to add hand mount to.")
            return

        # Get thumb corners to identify points to remove
        thumb_corners_2d = self.data.get("thumb_corners_2d")
        if thumb_corners_2d is not None:
            thumb_corners_2d = np.array(thumb_corners_2d, dtype=float)
        else:
            thumb_corners_2d = np.zeros((0, 2), dtype=float)

        # Create modified outline by removing points near thumb corners
        # Use thumb_squares corners to identify the region to remove
        modified_outline = []
        removal_threshold_mm = 5.0  # Points within this distance of thumb corners are removed
        
        for pt in self.outline:
            keep = True
            for corner in thumb_corners_2d:
                dist = np.linalg.norm(pt - corner)
                if dist < removal_threshold_mm:
                    keep = False
                    break
            if keep:
                modified_outline.append(pt)
        
        modified_outline = np.array(modified_outline, dtype=float)
        
        if len(modified_outline) < 3:
            print("Not enough points in modified outline for hand mount.")
            return

        # Build a 2D polygon from the modified outline at Z = 0
        mount_height_m = 4.0 * mm_to_m
        
        bm = bmesh.new()
        verts = []
        for x_mm, y_mm in modified_outline:
            v = bm.verts.new((float(x_mm) * mm_to_m, float(y_mm) * mm_to_m, 0.0))
            verts.append(v)

        bm.verts.ensure_lookup_table()
        try:
            face = bm.faces.new(verts)
        except ValueError:
            face = bm.faces.get(verts)

        bm.normal_update()

        # Extrude the polygon face in -Z direction
        res = bmesh.ops.extrude_face_region(bm, geom=[face])
        geom_extruded = res.get("geom", [])
        extruded_verts = [elem for elem in geom_extruded if isinstance(elem, bmesh.types.BMVert)]
        if extruded_verts:
            bmesh.ops.translate(bm, verts=extruded_verts, vec=(0.0, 0.0, -mount_height_m))

        # Create mesh and object
        mount_mesh = bpy.data.meshes.new("viz_hand_mount")
        bm.to_mesh(mount_mesh)
        bm.free()

        mount_obj = bpy.data.objects.new("viz_hand_mount", mount_mesh)
        bpy.context.scene.collection.objects.link(mount_obj)

        # Assign "viz" material to hand mount
        viz_mat = bpy.data.materials.get("viz")
        if viz_mat is not None and mount_obj.data is not None:
            mount_obj.data.materials.clear()
            mount_obj.data.materials.append(viz_mat)

        # Parent to palm root or palm object
        palm_parent = palm_root if palm_root is not None else palm_obj
        mount_obj.parent = palm_parent
        mount_obj.matrix_parent_inverse = palm_parent.matrix_world.inverted()

        # Recalculate normals
        self.recalculate_face_normals(mount_obj)


        ## add mount holes to the hand mount object
        # the object name is mount_holes and it does not need any transformation
        # we only subtract the mount holes object from the hand mount object using a boolean modifier and then we remove the mount holes object
        mount_holes = bpy.data.objects.get("mount_holes")
        if mount_holes is None:
            # Try to append from components.blend
            components_blend_path = os.path.join(os.path.dirname(__file__), "blender", "components.blend")
            if os.path.exists(components_blend_path):
                current_blend = bpy.data.filepath
                try:
                    samefile = current_blend and os.path.samefile(current_blend, components_blend_path)
                except Exception:
                    samefile = False
                
                if not samefile:
                    object_dir = os.path.join(components_blend_path, "Object")
                    try:
                        bpy.ops.wm.append(
                            filepath=os.path.join(object_dir, "mount_holes"),
                            directory=object_dir,
                            filename="mount_holes",
                            link=False,
                        )
                    except Exception:
                        pass
                
                mount_holes = bpy.data.objects.get("mount_holes")
        
        if mount_holes is not None:
            # Subtract mount_holes from hand mount using Boolean difference
            bpy.ops.object.select_all(action="DESELECT")
            mount_obj.select_set(True)
            bpy.context.view_layer.objects.active = mount_obj

            bool_mod = mount_obj.modifiers.new(name="MountHoles", type="BOOLEAN")
            bool_mod.operation = "DIFFERENCE"
            bool_mod.solver = "EXACT"
            bool_mod.object = mount_holes

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply mount holes boolean modifier: {e}")

            # Remove the mount_holes object
            try:
                bpy.data.objects.remove(mount_holes, do_unlink=True)
            except Exception:
                pass

            self.recalculate_face_normals(mount_obj)

    def add_screw_attachment(self):
        '''
        *we add some geometries to the viz_PalmBody and viz_hand_mount object for the screw attachments.
        *for the viz_PalmBody object, we add multiple cylinders in a few locations. the diameter of the cylinders are 8.5 mm and height is self.palm_extrude_height_m - self.palm_thickness_m.
        *the xy location of the cylinders (z remains the same):
        1- the point on the inward_offset_outline that is closest to the average location of the two consecutive finger bases. it shouldn't be closer than 10 mm to the bases.
        after cmerging these cylinders with viz_PalmBody object, we subtract cylinders with diameter 4.5 mm and height 10 mmin the same location from the viz_PalmBody object.
        then we subtracct cylinders of diameter 3.2 mm and heght 15 mm in the same location from the viz_hand_mount object.
        2- two points on the inward_offset_outline that are closest to the thumb square sides and are not closer than 10 mm to each other
        also selected points should be outside the thumb squares.
        3- two points from inward_offset_outline that have the smallest abs(y) value and opposite x values 
        if points are inside any of the thumb squares or are closer than 10 mm to any of the other cylinders centers or the finger bases, don't add the point and skip.

        **we move the cylinder centers 1 mm closer to the origin.

        '''
        palm_obj = bpy.data.objects.get(self.viz_name)
        mount_obj = bpy.data.objects.get("viz_hand_mount")
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found for screw attachments.")
            return

        # Get finger bases
        finger_bases_2d = self.data.get("finger_bases_2d")
        if finger_bases_2d is None or len(finger_bases_2d) < 2:
            print("Not enough finger bases for screw attachments.")
            return
        finger_bases_2d = np.array(finger_bases_2d, dtype=float)

        # Cylinder dimensions
        outer_diameter_m = 8.5 * mm_to_m
        outer_height_m = self.palm_extrude_height_m - self.palm_thickness_m
        inner_diameter_m = 4.5 * mm_to_m
        inner_height_m = 10.0 * mm_to_m
        mount_hole_diameter_m = 3.2 * mm_to_m
        mount_hole_height_m = 15.0 * mm_to_m
        min_distance_mm = 10.0

        # Find cylinder locations: average of consecutive finger bases, then closest point on inward_offset_outline
        self.cylinder_locations = []
        for i in range(len(finger_bases_2d) - 1):
            avg_base = (finger_bases_2d[i] + finger_bases_2d[i + 1]) / 2.0

            # Find closest point on inward_offset_outline that is >= 10mm from both bases
            best_pt = None
            best_dist = float('inf')
            for pt in self.inward_offset_outline:
                dist_to_avg = np.linalg.norm(pt - avg_base)
                dist_to_base1 = np.linalg.norm(pt - finger_bases_2d[i])
                dist_to_base2 = np.linalg.norm(pt - finger_bases_2d[i + 1])
                if dist_to_base1 >= min_distance_mm and dist_to_base2 >= min_distance_mm:
                    if dist_to_avg < best_dist:
                        best_dist = dist_to_avg
                        best_pt = pt
            
            if best_pt is not None:
                # Move 1mm closer to the origin
                dist_to_origin = np.linalg.norm(best_pt)
                if dist_to_origin > 1.0:
                    direction_to_origin = -best_pt / dist_to_origin
                    best_pt = best_pt + direction_to_origin * 1.0
                self.cylinder_locations.append(best_pt)

        # 2- Two points on inward_offset_outline closest to thumb square sides (not closer than 10mm to each other)
        def point_to_segment_distance(pt, seg_start, seg_end):
            """Calculate distance from point to line segment."""
            seg = seg_end - seg_start
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-12:
                return np.linalg.norm(pt - seg_start)
            t = max(0, min(1, np.dot(pt - seg_start, seg) / seg_len_sq))
            proj = seg_start + t * seg
            return np.linalg.norm(pt - proj)

        def point_in_polygon(pt, polygon):
            """Check if point is inside polygon using ray casting."""
            n = len(polygon)
            inside = False
            x, y = pt[0], pt[1]
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i][0], polygon[i][1]
                xj, yj = polygon[j][0], polygon[j][1]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = i
            return inside

        def is_outside_all_thumb_squares(pt):
            """Check if point is outside all thumb squares."""
            for sq in self.thumb_squares:
                if len(sq) >= 4:
                    # Use first 4 points as the polygon
                    if point_in_polygon(pt, sq[:4]):
                        return False
            return True

        thumb_side_candidates = []
        for sq in self.thumb_squares:
            if len(sq) < 4:
                continue
            # Each square has sides: (0,1), (1,2), (2,3), (3,0)
            for side_idx in range(4):
                seg_start = sq[side_idx]
                seg_end = sq[(side_idx + 1) % 4]
                
                # Find closest point on inward_offset_outline to this side that is outside all thumb squares
                best_pt = None
                best_dist = float('inf')
                for pt in self.inward_offset_outline:
                    if not is_outside_all_thumb_squares(pt):
                        continue
                    d = point_to_segment_distance(pt, seg_start, seg_end)
                    if d < best_dist:
                        best_dist = d
                        best_pt = pt.copy()
                
                if best_pt is not None:
                    thumb_side_candidates.append(best_pt)

        # Select up to 2 points that are at least 10mm apart from each other
        thumb_side_points = []
        for pt in thumb_side_candidates:
            too_close = False
            for existing in thumb_side_points:
                if np.linalg.norm(pt - existing) < min_distance_mm:
                    too_close = True
                    break
            if not too_close:
                # Move 1mm closer to origin
                dist_to_origin = np.linalg.norm(pt)
                if dist_to_origin > 1.0:
                    direction_to_origin = -pt / dist_to_origin
                    pt = pt + direction_to_origin * 1.0
                thumb_side_points.append(pt)
                if len(thumb_side_points) >= 2:
                    break

        self.cylinder_locations.extend(thumb_side_points)

        # 3- Two points with smallest abs(y) and opposite x values from inward_offset_outline
        #    Skip if inside thumb squares or closer than 10mm to existing cylinder_locations or finger bases
        # Split into positive x and negative x, sort each by abs(y) value (ascending)
        pos_x_pts = sorted([pt for pt in self.inward_offset_outline if pt[0] > 0], key=lambda p: abs(p[1]))
        neg_x_pts = sorted([pt for pt in self.inward_offset_outline if pt[0] < 0], key=lambda p: abs(p[1]))
        
        # Get the point with smallest y from each side
        pos_x_candidate = pos_x_pts[0].copy() if len(pos_x_pts) > 0 else None
        neg_x_candidate = neg_x_pts[0].copy() if len(neg_x_pts) > 0 else None
        
        def should_skip(pt):
            """Check if point should be skipped (inside thumb square or too close to existing)."""
            if not is_outside_all_thumb_squares(pt):
                return True
            for existing in self.cylinder_locations:
                if np.linalg.norm(pt - existing) < min_distance_mm:
                    return True
            for base in finger_bases_2d:
                if np.linalg.norm(pt - base) < min_distance_mm:
                    return True
            return False
        
        # Check and add positive x point
        if pos_x_candidate is not None and not should_skip(pos_x_candidate):
            # Move 1mm closer to origin
            dist_to_origin = np.linalg.norm(pos_x_candidate)
            if dist_to_origin > 1.0:
                direction_to_origin = -pos_x_candidate / dist_to_origin
                pos_x_candidate = pos_x_candidate + direction_to_origin * 1.0
            self.cylinder_locations.append(pos_x_candidate)
        
        # Check and add negative x point
        if neg_x_candidate is not None and not should_skip(neg_x_candidate):
            # Move 1mm closer to origin
            dist_to_origin = np.linalg.norm(neg_x_candidate)
            if dist_to_origin > 1.0:
                direction_to_origin = -neg_x_candidate / dist_to_origin
                neg_x_candidate = neg_x_candidate + direction_to_origin * 1.0
            self.cylinder_locations.append(neg_x_candidate)

        if not self.cylinder_locations:
            print("No valid locations found for screw attachments.")
            return

        # Z position for cylinders (centered vertically in the palm body height)
        z_center = outer_height_m / 2.0

        # Ensure object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Step 1: Add outer cylinders and merge with palm body
        for idx, loc_mm in enumerate(self.cylinder_locations):
            loc_m = (float(loc_mm[0]) * mm_to_m, float(loc_mm[1]) * mm_to_m, z_center)

            bpy.ops.mesh.primitive_cylinder_add(
                radius=outer_diameter_m / 2.0,
                depth=outer_height_m,
                location=loc_m
            )
            cyl = bpy.context.active_object
            cyl.name = f"screw_attach_outer_{idx}"
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            self.recalculate_face_normals(cyl)

            # Union with palm body
            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"ScrewAttachOuter_{idx}", type="BOOLEAN")
            bool_mod.operation = "UNION"
            bool_mod.solver = "EXACT"
            bool_mod.object = cyl

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply outer cylinder union {idx}: {e}")
            finally:
                try:
                    bpy.data.objects.remove(cyl, do_unlink=True)
                except Exception:
                    pass

        # Step 2: Subtract inner cylinders from palm body
        for idx, loc_mm in enumerate(self.cylinder_locations):
            loc_m = (float(loc_mm[0]) * mm_to_m, float(loc_mm[1]) * mm_to_m, inner_height_m / 2.0)

            bpy.ops.mesh.primitive_cylinder_add(
                radius=inner_diameter_m / 2.0,
                depth=inner_height_m,
                location=loc_m
            )
            cyl = bpy.context.active_object
            cyl.name = f"screw_attach_inner_{idx}"
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            self.recalculate_face_normals(cyl)

            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"ScrewAttachInner_{idx}", type="BOOLEAN")
            bool_mod.operation = "DIFFERENCE"
            bool_mod.solver = "EXACT"
            bool_mod.object = cyl

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply inner cylinder difference {idx}: {e}")
            finally:
                try:
                    bpy.data.objects.remove(cyl, do_unlink=True)
                except Exception:
                    pass

        # Step 3: Subtract mount holes from hand mount object
        if mount_obj is not None and mount_obj.type == "MESH":
            for idx, loc_mm in enumerate(self.cylinder_locations):
                # Position below Z=0 since hand mount extends in -Z
                loc_m = (float(loc_mm[0]) * mm_to_m, float(loc_mm[1]) * mm_to_m, -mount_hole_height_m / 2.0)

                bpy.ops.mesh.primitive_cylinder_add(
                    radius=mount_hole_diameter_m / 2.0,
                    depth=mount_hole_height_m,
                    location=loc_m
                )
                cyl = bpy.context.active_object
                cyl.name = f"screw_attach_mount_{idx}"
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                self.recalculate_face_normals(cyl)

                bpy.ops.object.select_all(action="DESELECT")
                mount_obj.select_set(True)
                bpy.context.view_layer.objects.active = mount_obj

                bool_mod = mount_obj.modifiers.new(name=f"ScrewAttachMount_{idx}", type="BOOLEAN")
                bool_mod.operation = "DIFFERENCE"
                bool_mod.solver = "EXACT"
                bool_mod.object = cyl

                try:
                    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
                except Exception as e:
                    print(f"Failed to apply mount hole difference {idx}: {e}")
                finally:
                    try:
                        bpy.data.objects.remove(cyl, do_unlink=True)
                    except Exception:
                        pass

        self.recalculate_face_normals(palm_obj)
        if mount_obj is not None:
            self.recalculate_face_normals(mount_obj)
        self.merge_vertices_by_distance()

    def add_motor_cable_cutout(self):
        '''
        add cutouts to the palm body model to allow the motor cable to pass through.
        the cutouts are squares with width 10 mm and height 5 mm.
        for each finger base, we add a cutout at the location and transormation of the finger base in x y coordinate but z location remains zero
        we subtract them from the viz_PalmBody object using a boolean modifier.
        '''
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found for motor cable cutouts.")
            return

        # Get finger bases and normals
        finger_bases_2d = self.data.get("finger_bases_2d")
        finger_normals_2d = self.data.get("finger_normals_2d")
        if finger_bases_2d is None or len(finger_bases_2d) == 0:
            print("No finger bases found for motor cable cutouts.")
            return
        finger_bases_2d = np.array(finger_bases_2d, dtype=float)
        if finger_normals_2d is not None:
            finger_normals_2d = np.array(finger_normals_2d, dtype=float)
        else:
            finger_normals_2d = np.zeros_like(finger_bases_2d)

        # Cutout dimensions in meters
        cutout_width_m = 10.0 * mm_to_m
        cutout_height_m = 5.0 * mm_to_m
        cutout_depth_m = 20.0 * mm_to_m  # Enough depth to cut through

        def _unit(v):
            v = np.asarray(v, dtype=float)
            n = float(np.linalg.norm(v))
            return v / n if n > 0.0 else v

        # Ensure object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        for idx in range(len(finger_bases_2d)):
            base_xy_mm = finger_bases_2d[idx]
            normal_xy = _unit(finger_normals_2d[idx]) if idx < len(finger_normals_2d) else np.array([0.0, 1.0])

            # Position in meters, z = 0
            bx = float(base_xy_mm[0]) * mm_to_m
            by = float(base_xy_mm[1]) * mm_to_m

            # Rotation around Z so cutout aligns with finger normal
            angle_z = math.atan2(float(normal_xy[1]), float(normal_xy[0]))

            # Create cube for cutout
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
            cutout = bpy.context.active_object
            if cutout is None:
                continue
            cutout.name = f"motor_cable_cutout_{idx}"

            # Set dimensions: width (X) x depth (Y) x height (Z)
            cutout.dimensions = (cutout_width_m, cutout_depth_m, cutout_height_m)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            # Rotate to align with finger normal
            cutout.rotation_euler = (0.0, 0.0, angle_z)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

            # Position at finger base, z = 0 (centered on z)
            cutout.location = (bx, by, 0.0)
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

            self.recalculate_face_normals(cutout)

            # Subtract from palm body
            bpy.ops.object.select_all(action="DESELECT")
            palm_obj.select_set(True)
            bpy.context.view_layer.objects.active = palm_obj

            bool_mod = palm_obj.modifiers.new(name=f"MotorCableCutout_{idx}", type="BOOLEAN")
            bool_mod.operation = "DIFFERENCE"
            bool_mod.solver = "EXACT"
            bool_mod.object = cutout

            try:
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            except Exception as e:
                print(f"Failed to apply motor cable cutout boolean modifier {idx}: {e}")
            finally:
                try:
                    bpy.data.objects.remove(cutout, do_unlink=True)
                except Exception:
                    pass

        self.recalculate_face_normals(palm_obj)
        self.merge_vertices_by_distance()

    def add_usb_cable_cutout(self):
        '''
        add cutouts to the palm body model to allow the usb cable to pass through.
        the cutouts are squares with width 8 mm and height 8 mm.
        we add a cutout at the outline point with smallest abs(y) and smallest x value that is at leats 10 mm away from any of the finger bases and cylinders_locations.
        we subtract them from the viz_PalmBody object using a boolean modifier.
        '''
        palm_obj = bpy.data.objects.get(self.viz_name)
        if palm_obj is None or palm_obj.type != "MESH":
            print(f"No {self.viz_name} mesh found for USB cable cutout.")
            return

        # Get finger bases
        finger_bases_2d = self.data.get("finger_bases_2d")
        if finger_bases_2d is not None:
            finger_bases_2d = np.array(finger_bases_2d, dtype=float)
        else:
            finger_bases_2d = np.zeros((0, 2), dtype=float)

        min_distance_mm = 10.0

        # Use cylinder_locations saved from add_screw_attachment
        cylinder_locs = self.cylinder_locations if hasattr(self, 'cylinder_locations') else []

        # Combine all locations to avoid
        avoid_locations = list(finger_bases_2d) + list(cylinder_locs)

        # Cutout dimensions
        cutout_size_m = 10.0 * mm_to_m
        cutout_depth_m = 15.0 * mm_to_m

        # Find outline point with smallest abs(y) and smallest x, at least 10mm from avoid locations
        # Sort by (abs(y), x) to get smallest abs(y) first, then smallest x as tiebreaker
        sorted_pts = sorted(self.outline, key=lambda p: (abs(p[1]), p[0]))

        best_pt = None
        for pt in sorted_pts:
            valid = True
            for avoid in avoid_locations:
                if np.linalg.norm(pt - avoid) < min_distance_mm:
                    valid = False
                    break
            if valid:
                best_pt = pt.copy()
                break

        if best_pt is None:
            print("No valid location found for USB cable cutout.")
            return

        # Position in meters, z = 0
        bx = float(best_pt[0]) * mm_to_m
        by = float(best_pt[1]) * mm_to_m

        # Ensure object mode
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj
        if bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Create cube for cutout
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        cutout = bpy.context.active_object
        if cutout is None:
            print("Failed to create USB cable cutout cube.")
            return
        cutout.name = "usb_cable_cutout"

        # Set dimensions: 8mm x 8mm x depth
        cutout.dimensions = (cutout_size_m, cutout_size_m, cutout_depth_m)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Position at the selected point, z = 0
        cutout.location = (bx, by, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        self.recalculate_face_normals(cutout)

        # Subtract from palm body
        bpy.ops.object.select_all(action="DESELECT")
        palm_obj.select_set(True)
        bpy.context.view_layer.objects.active = palm_obj

        bool_mod = palm_obj.modifiers.new(name="USBCableCutout", type="BOOLEAN")
        bool_mod.operation = "DIFFERENCE"
        bool_mod.solver = "EXACT"
        bool_mod.object = cutout

        try:
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        except Exception as e:
            print(f"Failed to apply USB cable cutout boolean modifier: {e}")
        finally:
            try:
                bpy.data.objects.remove(cutout, do_unlink=True)
            except Exception:
                pass

        self.recalculate_face_normals(palm_obj)
        self.merge_vertices_by_distance()

    def generate(self):
        self.create_main_extruded_mesh()
        self.offset_all_finger_and_thumb_components()
        if self.detailed_viz:
            self.add_thumb_base_mount()
            self.add_screw_holes()
            self.add_hand_mount()
            self.add_screw_attachment()
            self.add_motor_cable_cutout()
            self.add_usb_cable_cutout()
        else:
            # Remove the screw_holes and mount_holes objects when not using detailed viz
            for obj in list(bpy.data.objects):
                if obj.name.lower().startswith("screw_holes") or obj.name.lower().startswith("mount_holes"):
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except Exception:
                        pass

class FingerAssembly:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.elements = []
        self.data = np.load(data_path, allow_pickle=True)
        self.components_blend = (self.data["components_blend"].tolist() if hasattr(self.data["components_blend"], "tolist") else str(self.data["components_blend"]))
        self.prefixes = self.data.get("prefixes")
        matrices = self.data.get("matrices")
        groups = self.data.get("groups")
        subgroups = self.data.get("subgroups")
        self.finger_pad_resolution_level = int(self.data.get("finger_pad_resolution_level", 4))
        self.finger_bump_edge_zeroing_spread = float(self.data.get("finger_bump_edge_zeroing_spread", 0.3))
        elem_types = self.data.get("elem_types")
        joint_surface_outline_points = self.data.get("joint_surface_outline_points")
        joint_surface_local_T = self.data.get("joint_surface_local_T")
        joint_bump_params = self.data.get("joint_bump_params")
        
        # Load finger/thumb bump config parameters (indexed by finger/thumb index, then pad position)
        self.finger_bump_type_lists = self.data.get("finger_bump_type_lists", None)
        self.finger_bump_number_lists = self.data.get("finger_bump_number_lists", None)
        self.finger_bump_max_height_mm_lists = self.data.get("finger_bump_max_height_mm_lists", None)
        self.finger_bump_height_intensity_lists = self.data.get("finger_bump_height_intensity_lists", None)
        self.finger_bumps_spread_lists = self.data.get("finger_bumps_spread_lists", None)
        self.finger_bumps_aspect_ratio_lists = self.data.get("finger_bumps_aspect_ratio_lists", None)
        self.finger_bump_rotation_deg_lists = self.data.get("finger_bump_rotation_deg_lists", None)
        self.finger_bump_center_angle_deg_lists = self.data.get("finger_bump_center_angle_deg_lists", None)
        self.finger_bump_center_offset_lists = self.data.get("finger_bump_center_offset_lists", None)
        
        self.thumb_bump_type_lists = self.data.get("thumb_bump_type_lists", None)
        self.thumb_bump_number_lists = self.data.get("thumb_bump_number_lists", None)
        self.thumb_bump_max_height_mm_lists = self.data.get("thumb_bump_max_height_mm_lists", None)
        self.thumb_bump_height_intensity_lists = self.data.get("thumb_bump_height_intensity_lists", None)
        self.thumb_bumps_spread_lists = self.data.get("thumb_bumps_spread_lists", None)
        self.thumb_bumps_aspect_ratio_lists = self.data.get("thumb_bumps_aspect_ratio_lists", None)
        self.thumb_bump_rotation_deg_lists = self.data.get("thumb_bump_rotation_deg_lists", None)
        self.thumb_bump_center_angle_deg_lists = self.data.get("thumb_bump_center_angle_deg_lists", None)
        self.thumb_bump_center_offset_lists = self.data.get("thumb_bump_center_offset_lists", None)

        self.finger_fingertip_scale_factors = self.data.get("finger_fingertip_scale_factors", None)
        self.thumb_fingertip_scale_factors = self.data.get("thumb_fingertip_scale_factors", None)
        self.link_added_length_mm = self.data.get("link_added_length_mm", None)
        cm = self.data.get("collision_mesh", None)
        self.collision_mesh = bool(cm) if cm is not None else False

        # Track joint index within each finger/thumb for pad parameter assignment
        subgroup_joint_counter = {}  # subgroup -> number of joints seen so far
        
        n = int(len(self.prefixes)) if self.prefixes is not None else 0
        for i in range(n):
            group = str(groups[i]) if groups is not None else "fingers"
            subgroup = str(subgroups[i]) if subgroups is not None else ""
            elem_type = str(elem_types[i]) if elem_types is not None and i < len(elem_types) else ""
            
            # Determine pad_index for this joint within its finger/thumb
            pad_index = -1
            if elem_type == "joint":
                if subgroup not in subgroup_joint_counter:
                    subgroup_joint_counter[subgroup] = 0
                pad_index = subgroup_joint_counter[subgroup]
                subgroup_joint_counter[subgroup] += 1
            
            fingertip_scale = (1.0, 1.0, 1.0)
            if str(self.prefixes[i]) == "link_4":
                try:
                    if group == "thumbs":
                        _ft_idx = int(str(subgroup).split('_')[-1]) - 1
                        if (self.thumb_fingertip_scale_factors is not None
                                and _ft_idx < len(self.thumb_fingertip_scale_factors)):
                            _raw = self.thumb_fingertip_scale_factors[_ft_idx]
                            fingertip_scale = tuple(float(x) for x in _raw)
                    else:
                        _ft_idx = int(str(subgroup).split('_')[-1]) - 1
                        if (self.finger_fingertip_scale_factors is not None
                                and _ft_idx < len(self.finger_fingertip_scale_factors)):
                            _raw = self.finger_fingertip_scale_factors[_ft_idx]
                            fingertip_scale = tuple(float(x) for x in _raw)
                except Exception:
                    fingertip_scale = (1.0, 1.0, 1.0)

            self.elements.append({
                "prefix": str(self.prefixes[i]),
                "group": group,
                "subgroup": subgroup,
                "matrix_world": matrices[i] if matrices is not None else None,
                "elem_type": elem_type,
                "joint_surface_outline_points": (joint_surface_outline_points[i] if joint_surface_outline_points is not None and i < len(joint_surface_outline_points) else None),
                "joint_surface_local_T": (joint_surface_local_T[i] if joint_surface_local_T is not None and i < len(joint_surface_local_T) else None),
                "joint_bump_params": (joint_bump_params[i] if joint_bump_params is not None and i < len(joint_bump_params) else None),
                "pad_index": pad_index,  # Index of this joint's pad within its finger/thumb (0-based)
                "fingertip_scale": fingertip_scale,
                "link_added_length_mm": float(self.link_added_length_mm[i]) if (self.link_added_length_mm is not None and i < len(self.link_added_length_mm)) else 0.0,
            })

        # Determine prefixes needed
        self.prefixes = sorted(set(e.get("prefix") for e in self.elements if e.get("prefix")))
        if not self.prefixes:
            print("No elements to assemble.")
            return

        # Open the components.blend as main file to inherit its settings (units, render, world, etc.)
        try:
            bpy.ops.wm.open_mainfile(filepath=self.components_blend)
        except Exception as e:
            print(f"Failed to open main file {self.components_blend}: {e}")
            return

        # Use objects present in components.blend as templates. We'll duplicate them and keep originals intact.
        self.all_objs = [o for o in bpy.data.objects]
        self.name_to_obj = {o.name: o for o in self.all_objs}

    def _numpy_like_to_blender_matrix(self, T, translate_scale=0.001):
        import mathutils  # type: ignore
        return mathutils.Matrix((
            (float(T[0][0]), float(T[0][1]), float(T[0][2]), float(T[0][3]) * translate_scale),
            (float(T[1][0]), float(T[1][1]), float(T[1][2]), float(T[1][3]) * translate_scale),
            (float(T[2][0]), float(T[2][1]), float(T[2][2]), float(T[2][3]) * translate_scale),
            (0.0, 0.0, 0.0, 1.0),
        ))

    def _get_or_create_collection(self, name, parent):
        col = bpy.data.collections.get(name)
        if col is None:
            col = bpy.data.collections.new(name)
            parent.children.link(col)
        elif name not in parent.children.keys():
            parent.children.link(col)
        return col

    def _reset_collection(self, name, parent):
        existing = bpy.data.collections.get(name)
        if existing is not None:
            # unlink all children and objects
            for obj in list(existing.objects):
                try:
                    existing.objects.unlink(obj)
                except Exception:
                    pass
            try:
                if name in parent.children.keys():
                    parent.children.unlink(existing)
                bpy.data.collections.remove(existing)
            except Exception:
                pass
        col = bpy.data.collections.new(name)
        parent.children.link(col)
        return col

    def _duplicate_group(self, objs, suffix, target_col):
        dup_map = {}
        remaining = set(objs)
        ordered = []
        while remaining:
            progressed = False
            for o in list(remaining):
                if o.parent is None or o.parent not in remaining:
                    ordered.append(o)
                    remaining.remove(o)
                    progressed = True
            if not progressed:
                ordered.extend(list(remaining))
                remaining.clear()
        for src in ordered:
            dup = src.copy()
            if getattr(src, "data", None) is not None:
                dup.data = src.data.copy()
            dup.name = f"{src.name}_{suffix}"
            target_col.objects.link(dup)
            dup_map[src] = dup
        for src, dup in dup_map.items():
            dup.parent = dup_map.get(src.parent, None)
            dup.matrix_basis = src.matrix_basis.copy()
            # Also copy matrix_parent_inverse to ensure child objects (like colliders) 
            # maintain correct relative position to their parent
            if src.parent is not None:
                dup.matrix_parent_inverse = src.matrix_parent_inverse.copy()
        return dup_map

    def run_assembly(self):
        
        def gather_subtree(root_obj):
            stack = [root_obj]
            out = []
            seen = set()
            while stack:
                cur = stack.pop()
                if cur is None or cur.name in seen:
                    continue
                seen.add(cur.name)
                out.append(cur)
                for ch in getattr(cur, "children", []):
                    stack.append(ch)
            return out

        # For each prefix, find root objects that start with the prefix, whose parent doesn't start with the same prefix,
        # then collect the full subtree (including children regardless of names).
        prefix_to_templates = {}
        for p in self.prefixes:
            starts = [o for o in self.all_objs if o.name.startswith(p)]
            roots = [o for o in starts if (o.parent is None or not o.parent.name.startswith(p))]
            subtree = []
            seen_names = set()
            for r in roots:
                for node in gather_subtree(r):
                    if node.name not in seen_names:
                        subtree.append(node)
                        seen_names.add(node.name)
            # Fallback: if no explicit roots were found, use all starting-with objects
            if not subtree and starts:
                for o in starts:
                    for node in gather_subtree(o):
                        if node.name not in seen_names:
                            subtree.append(node)
                            seen_names.add(node.name)
            prefix_to_templates[p] = subtree

        # Prepare collections
        scene_root = bpy.context.scene.collection
        hand_col = self._reset_collection("hand", scene_root)

        # Also prepare per-finger/thumb subcollections under 'hand'
        subname_to_collection = {}
        subnames = self.data.get("subgroups") if isinstance(self.elements, list) else None
        # Build subname list matching elements length
        if subnames is None:
            # Fallback: generate sequential subnames
            f_i, t_i = 1, 1
            subnames = []
            for e in self.elements:
                if e.get("group", "fingers") == "fingers":
                    subnames.append(f"finger_{f_i}")
                    f_i += 1
                else:
                    subnames.append(f"thumb_{t_i}")
                    t_i += 1
        else:
            subnames = [str(n) for n in subnames.tolist()] if hasattr(subnames, "tolist") else [str(n) for n in subnames]

        # Build instances
        joint_idx_by_subname = {}
        counter = 0
        for idx, elem in enumerate(self.elements):
            prefix = elem.get("prefix")
            group = elem.get("group", "fingers")
            T = elem.get("matrix_world")
            subname = subnames[idx] if idx < len(subnames) else (f"finger_{idx+1}" if group == "fingers" else f"thumb_{idx+1}")
            if not prefix or T is None:
                continue
            templates = prefix_to_templates.get(prefix, [])
            if not templates:
                print(f"Warning: No template objects found for prefix '{prefix}'")
                continue
            # per-finger/thumb collection
            sub_col = subname_to_collection.get(subname)
            if sub_col is None:
                sub_col = self._get_or_create_collection(subname, hand_col)
                subname_to_collection[subname] = sub_col
            dup_map = self._duplicate_group(templates, f"{group[0]}{counter}", sub_col)

            ft_scale = elem.get("fingertip_scale", (1.0, 1.0, 1.0))
            # Normalize: accept scalar (legacy) or 3-tuple
            if isinstance(ft_scale, (int, float)):
                ft_scale = (float(ft_scale), float(ft_scale), float(ft_scale))
            else:
                ft_scale = tuple(float(x) for x in ft_scale)
            _is_default = all(abs(s - 1.0) < 1e-9 for s in ft_scale)
            if prefix == "link_4" and not _is_default:
                import mathutils  # type: ignore
                S = mathutils.Matrix.Diagonal((ft_scale[0], ft_scale[1], ft_scale[2], 1.0))
                for obj in dup_map.values():
                    if obj.name.lower().startswith("viz_"):
                        # Collect viz_ and all its descendants (colliders, etc.)
                        to_scale = [obj]
                        stack = list(obj.children)
                        while stack:
                            c = stack.pop()
                            to_scale.append(c)
                            stack.extend(c.children)
                        # Save desired scaled world matrices before changing anything
                        desired_mw = {t: (S @ t.matrix_world).copy() for t in to_scale}
                        # Process parent-first: bake scale into mesh, set clean world matrix
                        for target in to_scale:
                            loc, rot, scl = desired_mw[target].decompose()
                            if target.data is not None and hasattr(target.data, 'transform'):
                                scl_mat = mathutils.Matrix.Diagonal((*scl, 1.0))
                                target.data.transform(scl_mat)
                                target.data.update()
                            target.matrix_world = (
                                mathutils.Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
                            )
                # Store per-axis scale on root objects so URDF exporter can generate unique mesh names
                for r in [d for s, d in dup_map.items() if d.parent is None] or list(dup_map.values()):
                    r["mesh_scale_factor_x"] = ft_scale[0]
                    r["mesh_scale_factor_y"] = ft_scale[1]
                    r["mesh_scale_factor_z"] = ft_scale[2]

            # --- link_0 z-scaling based on link_added_length_mm ---
            link_length = elem.get("link_added_length_mm", 0.0)
            if prefix == "link_0" and link_length > 0:
                import mathutils  # type: ignore
                z_scale = link_length / 5.0
                S = mathutils.Matrix.Diagonal((1.0, 1.0, z_scale, 1.0))
                for obj in dup_map.values():
                    if obj.name.lower().startswith("viz_"):
                        # Collect viz_ and all its descendants (colliders, etc.)
                        to_scale = [obj]
                        stack = list(obj.children)
                        while stack:
                            c = stack.pop()
                            to_scale.append(c)
                            stack.extend(c.children)
                        # Save desired scaled world matrices before changing anything
                        desired_mw = {t: (S @ t.matrix_world).copy() for t in to_scale}
                        # Process parent-first: bake scale into mesh, set clean world matrix
                        for target in to_scale:
                            loc, rot, scl = desired_mw[target].decompose()
                            if target.data is not None and hasattr(target.data, 'transform'):
                                scl_mat = mathutils.Matrix.Diagonal((*scl, 1.0))
                                target.data.transform(scl_mat)
                                target.data.update()
                            target.matrix_world = (
                                mathutils.Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
                            )
                # Move attach_2 child object to z = link_length (in mm -> meters)
                for obj in dup_map.values():
                    if "attach_2" in obj.name:
                        obj.location.z = link_length * 0.001
                # Store per-axis scale on root objects for unique URDF mesh naming
                for r in [d for s, d in dup_map.items() if d.parent is None] or list(dup_map.values()):
                    r["mesh_scale_factor_x"] = 1.0
                    r["mesh_scale_factor_y"] = 1.0
                    r["mesh_scale_factor_z"] = z_scale

            # --- insert_holes boolean subtraction for link_4 viz_ ---
            # Performed BEFORE the root world-transform (T) is applied so that
            # the duplicated viz_ and the insert_holes template are both still in
            # their original components.blend coordinate space where they overlap
            # by design.  After the boolean, the root transform moves everything
            # into its final position.
            if prefix == "link_4":
                insert_holes_template = bpy.data.objects.get("insert_holes")
                if insert_holes_template is None:
                    components_blend_path = os.path.join(os.path.dirname(__file__), "blender", "components.blend")
                    if os.path.exists(components_blend_path):
                        current_blend = bpy.data.filepath
                        try:
                            samefile = current_blend and os.path.samefile(current_blend, components_blend_path)
                        except Exception:
                            samefile = False
                        if not samefile:
                            object_dir = os.path.join(components_blend_path, "Object")
                            object_path = os.path.join(object_dir, "insert_holes")
                            if os.path.exists(object_path):
                                try:
                                    bpy.ops.wm.append(
                                        filepath=object_path,
                                        directory=object_dir,
                                        filename="insert_holes",
                                        link=False,
                                    )
                                except Exception as e:
                                    print(f"Failed to append 'insert_holes' from components.blend: {e}")
                            else:
                                print("No 'insert_holes' object found in components.blend; skipping insert holes.")
                    insert_holes_template = bpy.data.objects.get("insert_holes")
                if insert_holes_template is None:
                    print("No 'insert_holes' object available; skipping insert holes for link_4.")
                else:
                    viz_obj = None
                    for obj in dup_map.values():
                        if obj.name.lower().startswith("viz_"):
                            viz_obj = obj
                            break
                    if viz_obj is not None:
                        # Duplicate insert_holes — keep its original transform so it
                        # stays aligned with the link_4 template geometry.
                        insert_holes_copy = insert_holes_template.copy()
                        if insert_holes_template.data:
                            insert_holes_copy.data = insert_holes_template.data.copy()
                        insert_holes_copy.name = f"insert_holes_{counter}"
                        bpy.context.scene.collection.objects.link(insert_holes_copy)
                        bpy.context.view_layer.update()

                        # Bake the cutter's world transform into its mesh so the
                        # boolean modifier sees it at the correct world position.
                        bpy.ops.object.select_all(action="DESELECT")
                        insert_holes_copy.select_set(True)
                        bpy.context.view_layer.objects.active = insert_holes_copy
                        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

                        # Apply boolean DIFFERENCE on the viz_ object only
                        bpy.ops.object.select_all(action="DESELECT")
                        viz_obj.select_set(True)
                        bpy.context.view_layer.objects.active = viz_obj
                        bool_mod = viz_obj.modifiers.new(name=f"InsertHoles_{counter}", type="BOOLEAN")
                        bool_mod.operation = "DIFFERENCE"
                        bool_mod.solver = "EXACT"
                        bool_mod.object = insert_holes_copy
                        try:
                            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
                        except Exception as e:
                            print(f"Failed to apply insert holes boolean modifier for '{viz_obj.name}': {e}")
                        finally:
                            try:
                                bpy.data.objects.remove(insert_holes_copy, do_unlink=True)
                            except Exception:
                                pass

            for r in [d for s, d in dup_map.items() if d.parent is None] or list(dup_map.values()):
                r.matrix_world = self._numpy_like_to_blender_matrix(T)

            # --- pad displacement helpers (local to each pad) ---
            def _bump_height_mm(bump_pack, x_mm, y_mm):
                if bump_pack is None:
                    return 0.0
                if isinstance(bump_pack, dict):
                    bumps = bump_pack.get("bumps", [])
                    meta = bump_pack.get("meta", {}) or {}
                    raw_hmin = float(meta.get("raw_hmin", 0.0))
                    raw_hmax = float(meta.get("raw_hmax", 0.0))
                    scale = float(meta.get("scale", 0.0))
                else:
                    bumps = bump_pack
                    raw_hmin = 0.0
                    raw_hmax = 0.0
                    scale = 0.0
                raw = 0.0
                for bp in bumps or []:
                    try:
                        c = bp.get("center", (0.0, 0.0))
                        sigma_long = float(bp.get("sigma_long", 1.0))
                        sigma_short = float(bp.get("sigma_short", 1.0))
                        theta = float(bp.get("theta", 0.0))
                        amp = float(bp.get("amp", 0.0))
                        ct = math.cos(theta)
                        st = math.sin(theta)
                        dx = x_mm - float(c[0])
                        dy = y_mm - float(c[1])
                        xr = ct * dx + st * dy
                        yr = -st * dx + ct * dy
                        raw += amp * math.exp(-0.5 * ((xr / sigma_long) ** 2 + (yr / sigma_short) ** 2))
                    except Exception:
                        continue
                if scale > 0.0 and raw_hmax > raw_hmin:
                    return (raw - raw_hmin) / max(raw_hmax - raw_hmin, 1e-9) * scale
                return raw

            def _displace_top_surface_local_z(obj, bump_pack, edge_zeroing_spread: float):
                if obj is None or obj.type != "MESH":
                    return
                bm2 = bmesh.new()
                bm2.from_mesh(obj.data)
                if not bm2.verts:
                    bm2.free()
                    return
                bm2.verts.ensure_lookup_table()
                zmax = max(v.co.z for v in bm2.verts)
                tol = 1e-6
                top_verts = [v for v in bm2.verts if v.co.z >= zmax - tol]
                top_set = set(top_verts)

                boundary = []
                for v in top_verts:
                    for e in v.link_edges:
                        ov = e.other_vert(v)
                        if ov not in top_set:
                            boundary.append(v)
                            break

                weights = {v: 1.0 for v in top_verts}
                spread_ratio = float(edge_zeroing_spread) if edge_zeroing_spread is not None else 0.0
                if boundary and spread_ratio > 0.0:
                    import heapq
                    dist = {v: 0.0 for v in boundary}
                    heap = [(0.0, int(v.index), v) for v in boundary]
                    heapq.heapify(heap)
                    while heap:
                        d, _vi, v = heapq.heappop(heap)
                        if d != dist.get(v, None):
                            continue
                        for e in v.link_edges:
                            ov = e.other_vert(v)
                            if ov not in top_set:
                                continue
                            w = (v.co - ov.co).length
                            nd = d + float(w)
                            if nd < dist.get(ov, 1e30):
                                dist[ov] = nd
                                heapq.heappush(heap, (nd, int(ov.index), ov))
                    max_d = max(dist.get(v, 0.0) for v in top_verts) if top_verts else 0.0
                    spread_d = max(1e-12, float(spread_ratio) * float(max_d))
                    for v in top_verts:
                        d = float(dist.get(v, 0.0))
                        t = max(0.0, min(1.0, d / spread_d))
                        weights[v] = t * t * (3.0 - 2.0 * t)
                elif boundary:
                    for v in boundary:
                        weights[v] = 0.0

                for v in top_verts:
                    x_mm = float(v.co.x) / mm_to_m
                    y_mm = float(v.co.y) / mm_to_m
                    dz_mm = _bump_height_mm(bump_pack, x_mm, y_mm) * float(weights.get(v, 1.0))
                    v.co.z += dz_mm * mm_to_m
                bm2.to_mesh(obj.data)
                bm2.free()

            # ---------- helper to generate bump_pack from config ----------
            def _generate_bump_pack_from_config(
                bump_type: str,
                num_bumps: int,
                max_height_mm: float,
                height_intensity_list,
                spread_list,
                aspect_ratio_list,
                rotation_deg_list,
                center_angle_deg_list,
                center_offset_list,
                surface_radius_mm: float = 15.0
            ):
                """Generate bump_pack dictionary from config parameters for finger/thumb pads."""
                bumps = []
                for i in range(num_bumps):
                    hi_list = list(height_intensity_list) if hasattr(height_intensity_list, '__iter__') else []
                    sp_list = list(spread_list) if hasattr(spread_list, '__iter__') else []
                    ar_list = list(aspect_ratio_list) if hasattr(aspect_ratio_list, '__iter__') else []
                    rd_list = list(rotation_deg_list) if hasattr(rotation_deg_list, '__iter__') else []
                    ca_list = list(center_angle_deg_list) if hasattr(center_angle_deg_list, '__iter__') else []
                    co_list = list(center_offset_list) if hasattr(center_offset_list, '__iter__') else []
                    
                    if i >= len(hi_list):
                        break
                    angle_rad = np.deg2rad(ca_list[i]) if i < len(ca_list) else 0.0
                    offset_scaled = co_list[i] if i < len(co_list) else 0.0
                    center_r = offset_scaled * surface_radius_mm
                    cx = center_r * np.cos(angle_rad)
                    cy = center_r * np.sin(angle_rad)
                    spread = sp_list[i] if i < len(sp_list) else 0.25
                    sigma_base = spread * surface_radius_mm
                    aspect = ar_list[i] if i < len(ar_list) else 1.0
                    sigma_long = sigma_base * np.sqrt(aspect)
                    sigma_short = sigma_base / np.sqrt(aspect)
                    theta = np.deg2rad(rd_list[i]) if i < len(rd_list) else 0.0
                    intensity = hi_list[i] if i < len(hi_list) else 1.0
                    amp = intensity * max_height_mm
                    bumps.append({
                        "center": (float(cx), float(cy)),
                        "sigma_long": float(sigma_long),
                        "sigma_short": float(sigma_short),
                        "theta": float(theta),
                        "amp": float(amp),
                        "type": str(bump_type),
                    })
                return {
                    "bumps": bumps,
                    "meta": {"raw_hmin": 0.0, "raw_hmax": max_height_mm, "scale": max_height_mm}
                }

            # ---------- add viz pad for joint elements ----------
            if str(elem.get("elem_type", "")).lower() == "joint":
                # Joint index from palm (per finger/thumb chain), even if this joint has no surface pad.
                joint_idx_by_subname[subname] = int(joint_idx_by_subname.get(subname, 0)) + 1
                joint_idx_from_palm = int(joint_idx_by_subname[subname])

                # Finger/thumb number from subname
                chain_id = None
                finger_idx = 0  # 0-based index into the config lists
                is_thumb = False
                try:
                    if str(subname).lower().startswith("finger_"):
                        finger_idx = int(str(subname).split('_')[-1]) - 1  # Convert to 0-based
                        chain_id = f"f{finger_idx + 1}"
                    elif str(subname).lower().startswith("thumb_"):
                        finger_idx = int(str(subname).split('_')[-1]) - 1  # Convert to 0-based
                        chain_id = f"t{finger_idx + 1}"
                        is_thumb = True
                except Exception:
                    chain_id = None
                if chain_id is None:
                    chain_id = ("f1" if str(group) == "fingers" else "t1")
                    is_thumb = (str(group) == "thumbs")

                try:
                    outline_mm = elem.get("joint_surface_outline_points")
                    Tloc_mm = elem.get("joint_surface_local_T")
                    pad_index = elem.get("pad_index", -1)  # Index of this pad within its finger/thumb
                    
                    # Generate bump_pack from config parameters
                    bump_pack = None
                    if pad_index >= 0:
                        try:
                            if is_thumb and self.thumb_bump_type_lists is not None:
                                if finger_idx < len(self.thumb_bump_type_lists):
                                    type_list = self.thumb_bump_type_lists[finger_idx]
                                    num_list = self.thumb_bump_number_lists[finger_idx] if self.thumb_bump_number_lists is not None else []
                                    max_h_list = self.thumb_bump_max_height_mm_lists[finger_idx] if self.thumb_bump_max_height_mm_lists is not None else []
                                    hi_lists = self.thumb_bump_height_intensity_lists[finger_idx] if self.thumb_bump_height_intensity_lists is not None else []
                                    sp_lists = self.thumb_bumps_spread_lists[finger_idx] if self.thumb_bumps_spread_lists is not None else []
                                    ar_lists = self.thumb_bumps_aspect_ratio_lists[finger_idx] if self.thumb_bumps_aspect_ratio_lists is not None else []
                                    rd_lists = self.thumb_bump_rotation_deg_lists[finger_idx] if self.thumb_bump_rotation_deg_lists is not None else []
                                    ca_lists = self.thumb_bump_center_angle_deg_lists[finger_idx] if self.thumb_bump_center_angle_deg_lists is not None else []
                                    co_lists = self.thumb_bump_center_offset_lists[finger_idx] if self.thumb_bump_center_offset_lists is not None else []
                                    
                                    if pad_index < len(type_list):
                                        surface_radius = np.max(np.linalg.norm(outline_mm[:, :2], axis=1)) if len(outline_mm) > 0 else 15.0
                                        bump_pack = _generate_bump_pack_from_config(
                                            bump_type=str(type_list[pad_index]),
                                            num_bumps=int(num_list[pad_index]) if pad_index < len(num_list) else 0,
                                            max_height_mm=float(max_h_list[pad_index]) if pad_index < len(max_h_list) else 0.0,
                                            height_intensity_list=hi_lists[pad_index] if pad_index < len(hi_lists) else [],
                                            spread_list=sp_lists[pad_index] if pad_index < len(sp_lists) else [],
                                            aspect_ratio_list=ar_lists[pad_index] if pad_index < len(ar_lists) else [],
                                            rotation_deg_list=rd_lists[pad_index] if pad_index < len(rd_lists) else [],
                                            center_angle_deg_list=ca_lists[pad_index] if pad_index < len(ca_lists) else [],
                                            center_offset_list=co_lists[pad_index] if pad_index < len(co_lists) else [],
                                            surface_radius_mm=surface_radius,
                                        )
                            elif not is_thumb and self.finger_bump_type_lists is not None:
                                if finger_idx < len(self.finger_bump_type_lists):
                                    type_list = self.finger_bump_type_lists[finger_idx]
                                    num_list = self.finger_bump_number_lists[finger_idx] if self.finger_bump_number_lists is not None else []
                                    max_h_list = self.finger_bump_max_height_mm_lists[finger_idx] if self.finger_bump_max_height_mm_lists is not None else []
                                    hi_lists = self.finger_bump_height_intensity_lists[finger_idx] if self.finger_bump_height_intensity_lists is not None else []
                                    sp_lists = self.finger_bumps_spread_lists[finger_idx] if self.finger_bumps_spread_lists is not None else []
                                    ar_lists = self.finger_bumps_aspect_ratio_lists[finger_idx] if self.finger_bumps_aspect_ratio_lists is not None else []
                                    rd_lists = self.finger_bump_rotation_deg_lists[finger_idx] if self.finger_bump_rotation_deg_lists is not None else []
                                    ca_lists = self.finger_bump_center_angle_deg_lists[finger_idx] if self.finger_bump_center_angle_deg_lists is not None else []
                                    co_lists = self.finger_bump_center_offset_lists[finger_idx] if self.finger_bump_center_offset_lists is not None else []
                                    
                                    if pad_index < len(type_list):
                                        surface_radius = np.max(np.linalg.norm(outline_mm[:, :2], axis=1)) if len(outline_mm) > 0 else 15.0
                                        bump_pack = _generate_bump_pack_from_config(
                                            bump_type=str(type_list[pad_index]),
                                            num_bumps=int(num_list[pad_index]) if pad_index < len(num_list) else 0,
                                            max_height_mm=float(max_h_list[pad_index]) if pad_index < len(max_h_list) else 0.0,
                                            height_intensity_list=hi_lists[pad_index] if pad_index < len(hi_lists) else [],
                                            spread_list=sp_lists[pad_index] if pad_index < len(sp_lists) else [],
                                            aspect_ratio_list=ar_lists[pad_index] if pad_index < len(ar_lists) else [],
                                            rotation_deg_list=rd_lists[pad_index] if pad_index < len(rd_lists) else [],
                                            center_angle_deg_list=ca_lists[pad_index] if pad_index < len(ca_lists) else [],
                                            center_offset_list=co_lists[pad_index] if pad_index < len(co_lists) else [],
                                            surface_radius_mm=surface_radius,
                                        )
                        except Exception as e:
                            print(f"[pad][warn] Failed to generate bump_pack from config for '{subname}' pad {pad_index}: {e}")
                    
                    # Fallback to joint_bump_params if no config-based bump_pack generated
                    if bump_pack is None:
                        bump_pack = elem.get("joint_bump_params")
                    outline_mm = np.asarray(outline_mm, dtype=float) if outline_mm is not None else np.zeros((0, 3), dtype=float)
                    Tloc_mm = np.asarray(Tloc_mm, dtype=float) if Tloc_mm is not None else np.eye(4, dtype=float)
                    if outline_mm.ndim != 2 or outline_mm.shape[1] != 3 or outline_mm.shape[0] < 3 or Tloc_mm.shape != (4, 4):
                        outline_mm = np.zeros((0, 3), dtype=float)

                    if outline_mm.size > 0:
                        # Pick the duplicated object that corresponds exactly to `prefix`.
                        # Important: in some templates the `prefix` object is NOT a root; the previous
                        # "pick any root" fallback caused wrong pad transforms for grandchild motor layouts.
                        joint_root = None
                        try:
                            src_main = self.name_to_obj.get(str(prefix))
                            if src_main is not None:
                                joint_root = dup_map.get(src_main)
                        except Exception:
                            joint_root = None
                        if joint_root is None:
                            # Fallback to previous behavior
                            roots = [d for s, d in dup_map.items() if d.parent is None]
                            joint_root = roots[0] if roots else None

                        if joint_root is not None:
                            # Avoid duplicates if the script is run multiple times in the same .blend session
                            pad_name = f"viz_pad_{chain_id}_j{joint_idx_from_palm}"
                            collider_name = f"collider_pad_{chain_id}_j{joint_idx_from_palm}"
                            if bpy.data.objects.get(pad_name) is not None:
                                raise RuntimeError(f"{pad_name} already exists")
                            # Check if any collider with this base name exists (cube colliders have _{n} suffix)
                            existing_colliders = [o for o in bpy.data.objects if o.name.startswith(collider_name)]
                            if existing_colliders:
                                raise RuntimeError(f"{collider_name} colliders already exist")

                            # Build mesh in SURFACE-LOCAL coordinates (outline already in surface frame),
                            # and place the object using the PRECOMPUTED WORLD transform of the joint surface.
                            pts_m = np.array(outline_mm, dtype=float) * mm_to_m
                            if pts_m.shape[0] >= 4 and np.allclose(pts_m[0], pts_m[-1], atol=1e-9):
                                pts_m = pts_m[:-1]

                            if pts_m.shape[0] >= 3:
                                extrude_vec = (0.0, 0.0, 0.5 * mm_to_m)  # 0.5 mm along surface +Z

                                bm = bmesh.new()
                                verts = [bm.verts.new((float(p[0]), float(p[1]), float(p[2]))) for p in pts_m]
                                bm.verts.ensure_lookup_table()
                                try:
                                    face = bm.faces.new(verts)
                                except ValueError:
                                    face = bm.faces.get(verts)
                                bm.normal_update()

                                res = bmesh.ops.extrude_face_region(bm, geom=[face])
                                geom_extruded = res.get("geom", [])
                                extruded_verts = [e for e in geom_extruded if isinstance(e, bmesh.types.BMVert)]
                                if extruded_verts:
                                    bmesh.ops.translate(bm, verts=extruded_verts, vec=extrude_vec)

                                mesh = bpy.data.meshes.new(f"{pad_name}_mesh")
                                bm.to_mesh(mesh)
                                bm.free()

                                pad_obj = bpy.data.objects.new(pad_name, mesh)
                                from mathutils import Matrix  # type: ignore
                                Tloc_m = np.array(Tloc_mm, dtype=float)
                                Tloc_m[:3, 3] *= mm_to_m
                                Tloc_m = Matrix(Tloc_m.tolist())

                                # Link pad to the per-finger/thumb subcollection (consistent with other duplicated objs)
                                if sub_col is not None:
                                    try:
                                        sub_col.objects.link(pad_obj)
                                    except Exception:
                                        pass

                                # Find motor* (child/grandchild) so pad is sibling of viz_motor_ after parenting.
                                def _descendants_depth2(root):
                                    out = []
                                    for c1 in list(getattr(root, "children", [])):
                                        out.append(c1)
                                        for c2 in list(getattr(c1, "children", [])):
                                            out.append(c2)
                                    return out

                                motor_obj = None
                                motor_candidates = [o for o in _descendants_depth2(joint_root) if o.name.lower().startswith("motor")]
                                for mo2 in motor_candidates:
                                    if any("viz_motor_" in ch.name.lower() for ch in list(getattr(mo2, "children", []))):
                                        motor_obj = mo2
                                        break
                                if motor_obj is None and motor_candidates:
                                    motor_obj = motor_candidates[0]
                                pad_parent = motor_obj if motor_obj is not None else joint_root

                                # Parent to motor (or joint_root fallback) while preserving: world = pad_parent.world @ Tloc_m
                                pad_obj.parent = pad_parent
                                pad_obj.matrix_world = pad_parent.matrix_world @ Tloc_m

                                # Remesh joint pad
                                try:
                                    pad_obj.select_set(True)
                                    bpy.context.view_layer.objects.active = pad_obj
                                    rem = pad_obj.modifiers.new(name="PadRemesh", type="REMESH")
                                    rem.mode = 'SHARP'
                                    rem.octree_depth = int(self.finger_pad_resolution_level)
                                    rem.use_smooth_shade = True
                                    bpy.ops.object.modifier_apply(modifier=rem.name)
                                except Exception as e:
                                    print(f"[pad][warn] Failed to remesh joint pad '{pad_obj.name}': {e}")

                                # Displace only the top (extruded) surface along local +Z using joint bump params
                                try:
                                    _displace_top_surface_local_z(pad_obj, bump_pack, self.finger_bump_edge_zeroing_spread)
                                except Exception as e:
                                    print(f"[pad][warn] Failed to displace joint pad '{pad_obj.name}': {e}")

                                # Create cube colliders for this pad, child of the viz_pad it comes from.
                                # Name must start with collider_pad_ (no viz_ prefix).
                                if self.collision_mesh:
                                    create_pad_cube_colliders(
                                        pad_obj=pad_obj,
                                        collider_base_name=collider_name,
                                        scene_collection=bpy.context.scene.collection,
                                        sub_collection=sub_col,
                                    )

                except Exception as e:
                    print(f"[pad][warn] Failed to create viz_pad for element '{prefix}': {e}")

            counter += 1

        # Remove all original components collections and their objects; keep only 'hand' hierarchy
        def _collect_keep_sets(root):
            keep_cols = set()
            keep_objs = set()
            def walk(col):
                if col.name in keep_cols:
                    return
                keep_cols.add(col.name)
                for o in col.objects:
                    keep_objs.add(o.name)
                for c in col.children:
                    walk(c)
            walk(root)
            return keep_cols, keep_objs

        keep_cols, keep_objs = _collect_keep_sets(hand_col)
        # Delete collections not in keep
        for col in list(bpy.data.collections):
            if col.name in keep_cols:
                continue
            # unlink from all parents
            try:
                for pcol in list(bpy.data.collections):
                    if col.name in pcol.children.keys():
                        try:
                            pcol.children.unlink(col)
                        except Exception:
                            pass
                bpy.context.scene.collection.children.unlink(col) if col.name in bpy.context.scene.collection.children.keys() else None
            except Exception:
                pass
            try:
                bpy.data.collections.remove(col)
            except Exception:
                pass
        # Delete orphan/template objects not under keep
        for obj in list(bpy.data.objects):
            # Preserve screw_holes objects even if they are not in keep sets
            if obj.name.lower().startswith("screw_holes"):
                continue
            if obj.name.lower().startswith("mount_holes"):
                continue
            if obj.name in keep_objs:
                continue
            # if object not linked to any remaining collection, or not in keep set, drop it
            linked = any(True for _ in obj.users_collection)
            if not linked or obj.name not in keep_objs:
                try:
                    bpy.data.objects.remove(obj)
                except Exception:
                    pass

def assign_viz_materials():
    """Assign per-finger/thumb/palm materials to all viz_ objects."""
    hand_col = bpy.data.collections.get("hand")
    if hand_col is None:
        print("[materials] 'hand' collection not found; skipping material assignment.")
        return

    def _is_link4(obj):
        """Check if obj or any ancestor has a name starting with 'link_4'."""
        cur = obj
        while cur is not None:
            if cur.name.startswith("link_4"):
                return True
            cur = cur.parent
        return False

    def _set_material(obj, mat_name):
        mat = bpy.data.materials.get(mat_name)
        if mat is not None and obj.data is not None:
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    def _should_color(obj):
        """Return True if this viz_ object should receive the finger/thumb color.

        Eligible objects are named viz_pad or viz_cover, or have an ancestor
        whose name starts with link_ or link_t_ (e.g. link_1, link_t_2).
        """
        name_lower = obj.name.lower()
        if name_lower.startswith("viz_pad") or name_lower.startswith("viz_cover"):
            return True
        cur = obj.parent
        while cur is not None:
            if cur.name.startswith("link_") or cur.name.startswith("link_t_"):
                return True
            cur = cur.parent
        return False

    # --- Finger/thumb subcollections ---
    for sub_col in hand_col.children:
        col_name = sub_col.name  # e.g. "finger_1", "thumb_1"

        # Determine base material for this chain
        if col_name.startswith("finger_"):
            n = col_name.split("_")[-1]  # "1", "2", "3"
            base_mat = f"f{n}"
        elif col_name.startswith("thumb_"):
            base_mat = "t"
        else:
            base_mat = "black"

        for obj in sub_col.objects:
            if not obj.name.lower().startswith("viz_"):
                continue
            if _is_link4(obj):
                _set_material(obj, "white_tip")
            elif _should_color(obj):
                _set_material(obj, base_mat)
            else:
                _set_material(obj, "black")

    # --- Palm viz_ objects (children of PalmBody_root) ---
    palm_root = bpy.data.objects.get("PalmBody_root")
    if palm_root:
        stack = [palm_root]
        while stack:
            cur = stack.pop()
            if cur.name.lower().startswith("viz_") and cur.type == "MESH":
                _set_material(cur, "palm")
            stack.extend(list(cur.children))

def save_blend_file(output_dir: str):
    # Save resulting Blender file into the requested output directory
    out_dir = str(output_dir) if output_dir else bpy.path.abspath("//")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "assembled_hand_model.blend")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=False, copy=False, relative_remap=False)
        print(f"Saved generated model: {out_path}")
    except Exception as e:
        print(f"Failed to save generated model .blend file: {e}")


def apply_origin_offset(palm_npz_path: str):
    """
    Offset the PalmBody_root origin while keeping all meshes in their global positions.
    This should be called after all assembly is complete and before URDF export.
    
    The origin_offset_mm is loaded from palm_outlines_data.npz and applied to move
    PalmBody_root. Children are updated so their world positions remain unchanged.
    """
    from mathutils import Matrix, Vector  # type: ignore
    
    # Load origin offset from palm data
    if not os.path.isfile(palm_npz_path):
        print(f"[origin_offset] Palm NPZ not found: {palm_npz_path}")
        return
    
    palm_data = np.load(palm_npz_path, allow_pickle=True)
    origin_offset = palm_data.get("origin_offset_mm")
    if origin_offset is None:
        print("[origin_offset] No origin_offset_mm in palm data, skipping.")
        return
    
    offset_m = np.array(origin_offset, dtype=float) * mm_to_m
    if np.allclose(offset_m, 0.0):
        print("[origin_offset] Origin offset is zero, skipping.")
        return
    
    palm_root = bpy.data.objects.get("PalmBody_root")
    if palm_root is None:
        print("[origin_offset] PalmBody_root not found, skipping.")
        return
    
    # Store original world matrices of all children (direct and indirect)
    def get_all_descendants(obj):
        descendants = []
        stack = list(obj.children)
        while stack:
            child = stack.pop()
            descendants.append(child)
            stack.extend(list(child.children))
        return descendants
    
    all_children = get_all_descendants(palm_root)
    child_world_matrices = {child: child.matrix_world.copy() for child in all_children}
    
    # Move PalmBody_root to the new origin
    old_location = palm_root.location.copy()
    palm_root.location = Vector((
        old_location.x + float(offset_m[0]),
        old_location.y + float(offset_m[1]),
        old_location.z + float(offset_m[2]),
    ))
    
    # Update view layer to apply the change
    bpy.context.view_layer.update()
    
    # Restore children's world positions by updating their matrix_parent_inverse
    for child in all_children:
        if child.parent is not None:
            # Restore the original world matrix
            child.matrix_world = child_world_matrices[child]
    
    print(f"[origin_offset] Applied origin offset: {origin_offset} mm")

class ExportURDF:
    def __init__(self, data_path: str, assembly_npz_path: str | None = None, export_dir: str | None = None):
        """
        export the assembled hand model as a URDF file
        """
        from mathutils import Matrix  # type: ignore

        self.data_path = data_path
        self.collection_name = "hand"
        # allow caller to choose export directory (so GenerateHand can control output location)
        if export_dir:
            self.export_dir = export_dir
        else:
            self.export_dir = bpy.path.abspath("//urdf_hand_export")
        self.robot_name = "hand_robot"
        self.robot_package = "hand_pkg"
        self.attach_thresh = 0.0005  # 0.5 mm

        # internal state
        self.Matrix = Matrix
        # mesh export cache: (subfolder, normalized_name) -> relpath (so we export shared meshes once)
        self._mesh_rel_cache = {}

        # Joint limits loaded from GenerateHand.save_assembly_data() -> blender/assembly_data.npz
        # Default fallback (previous hardcoded values)
        self.default_joint_limit = {"effort": 0.95, "velocity": 8.48, "lower": -0.47, "upper": 2.443}
        self.joint_limit_by_instance = {}  # (group_initial, elem_counter:int) -> limit dict
        self.joint_id_by_instance = {}     # (group_initial, elem_counter:int) -> joint self.id (int)
        self.expected_joint_count_by_subgroup = {}  # subgroup name (e.g. finger_1, thumb_1) -> expected # of joint elements

        try:
            npz_path = assembly_npz_path if assembly_npz_path else os.path.join(self.data_path, "blender", "assembly_data.npz")
            if os.path.isfile(npz_path):
                ad = np.load(npz_path, allow_pickle=True)
                elem_counter = np.array(ad.get("elem_counter", []), dtype=int)
                elem_types = np.array(ad.get("elem_types", []), dtype=object)
                groups = np.array(ad.get("groups", []), dtype=object)
                subgroups = np.array(ad.get("subgroups", []), dtype=object)
                jlo = np.array(ad.get("joint_lower", []), dtype=float)
                jhi = np.array(ad.get("joint_upper", []), dtype=float)
                jvel = np.array(ad.get("joint_velocity", []), dtype=float)
                jeff = np.array(ad.get("joint_effort", []), dtype=float)
                jid = np.array(ad.get("joint_id", []), dtype=float)

                n = int(min(len(elem_counter), len(elem_types), len(groups), len(subgroups), len(jlo), len(jhi), len(jvel), len(jeff), len(jid)))
                for i in range(n):
                    # Expected joint-count per chain (finger_# / thumb_#)
                    if str(elem_types[i]) == "joint":
                        sg = str(subgroups[i])
                        if sg:
                            self.expected_joint_count_by_subgroup[sg] = int(self.expected_joint_count_by_subgroup.get(sg, 0)) + 1

                    if str(elem_types[i]) != "joint":
                        continue
                    gi = str(groups[i])[0].lower() if str(groups[i]) else ""
                    if gi not in ("f", "t"):
                        # Only fingers/thumbs are expected here
                        continue
                    # joint id (k in naming)
                    try:
                        if math.isfinite(float(jid[i])):
                            self.joint_id_by_instance[(gi, int(elem_counter[i]))] = int(float(jid[i]))
                    except Exception:
                        pass
                    lo = float(jlo[i])
                    hi = float(jhi[i])
                    vel = float(jvel[i])
                    eff = float(jeff[i])
                    # keep only finite values
                    if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(vel) and math.isfinite(eff)):
                        continue
                    self.joint_limit_by_instance[(gi, int(elem_counter[i]))] = {
                        "effort": eff,
                        "velocity": vel,
                        "lower": lo,
                        "upper": hi,
                    }
                # collision mesh flag from HandConfig
                cm = ad.get("collision_mesh", None)
                self.collision_mesh = bool(cm) if cm is not None else False
            else:
                self.collision_mesh = False
        except Exception as e:
            print(f"[URDF][warn] Failed to load joint limits from assembly_data.npz: {e}")
            self.collision_mesh = False


    # ---------------- Utilities (ported from urdf_exp.py) ----------------
    def depsgraph(self):
        return bpy.context.evaluated_depsgraph_get()

    def name_has(self, o, key):
        return key in o.name.lower()

    def world_mat(self, o):
        return o.matrix_world.copy()

    def euler_rpy_XYZ(self, R):
        e = R.to_euler('XYZ')
        return (e.x, e.y, e.z)

    def xyzrpy_from_world(self, A_world, B_world):
        T = A_world.inverted() @ B_world
        t = T.to_translation()
        r = self.euler_rpy_XYZ(T.to_3x3())
        return (t, r)

    def all_descendants(self, o):
        out, st = [], [o]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(list(x.children))
        return out

    def first_order_objects(self):
        coll = bpy.data.collections.get(self.collection_name)
        if not coll:
            raise RuntimeError(f'Collection "{self.collection_name}" not found.')

        # In the assembled hand, objects are typically inside subcollections
        # (e.g., hand/finger_1, hand/thumb_1). Collect recursively.
        def _collect_objects(c):
            out = list(c.objects)
            for ch in c.children:
                out.extend(_collect_objects(ch))
            return out

        objs = _collect_objects(coll)

        # Keep only those whose parent is NOT also in this collection set
        immediate = [o for o in objs if (o.parent is None) or (o.parent not in objs)]
        immediate.sort(key=lambda o: o.name.lower())
        return immediate

    def type_of_main(self, o):
        n = o.name.lower()
        if n.startswith("link_"):
            return "link"
        # New naming convention per README:
        # - n* objects (also accept legacy nj*)
        # - w* objects (also accept legacy wj*)
        if n.startswith("nj") or n.startswith("n"):
            return "n"
        if n.startswith("wj") or n.startswith("w"):
            return "w"
        return None

    def children_with_key(self, o, key, mesh_only=False):
        xs = []
        for d in self.all_descendants(o):
            if key in d.name.lower():
                if not mesh_only or d.type == 'MESH':
                    xs.append(d)
        return xs

    def world_loc(self, obj):
        return self.world_mat(obj).translation

    # ---------------- Inertia Calculation Methods ----------------
    def compute_mesh_volume_and_com(self, obj):
        """
        Compute volume and center of mass (in world coordinates) for a mesh object.
        Uses signed tetrahedron volume method for watertight meshes.
        Returns (volume_m3, com_world_vec) or (0.0, origin) if failed.
        """
        from mathutils import Vector  # type: ignore
        
        if obj is None or obj.type != 'MESH':
            return 0.0, Vector((0, 0, 0))
        
        dg = self.depsgraph()
        try:
            tmp_mesh = bpy.data.meshes.new_from_object(object=obj, preserve_all_data_layers=True, depsgraph=dg)
        except TypeError:
            tmp_mesh = bpy.data.meshes.new_from_object(obj, True, dg)
        
        bm = bmesh.new()
        bm.from_mesh(tmp_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.verts.ensure_lookup_table()
        
        world_mat = self.world_mat(obj)
        
        total_volume = 0.0
        weighted_com = Vector((0, 0, 0))
        
        for face in bm.faces:
            if len(face.verts) != 3:
                continue
            # Get vertices in world coordinates
            v0 = world_mat @ face.verts[0].co
            v1 = world_mat @ face.verts[1].co
            v2 = world_mat @ face.verts[2].co
            
            # Signed volume of tetrahedron with origin
            # V = (1/6) * (v0 · (v1 × v2))
            cross = v1.cross(v2)
            signed_vol = v0.dot(cross) / 6.0
            total_volume += signed_vol
            
            # Centroid of tetrahedron
            tet_com = (v0 + v1 + v2) / 4.0  # Origin is at (0,0,0)
            weighted_com += tet_com * signed_vol
        
        bm.free()
        bpy.data.meshes.remove(tmp_mesh)
        
        volume = abs(total_volume)
        if volume > 1e-12:
            com = weighted_com / total_volume
        else:
            com = self.world_loc(obj)
        
        return volume, com

    def compute_mesh_inertia_at_com(self, obj, mass):
        """
        Compute inertia tensor (3x3 matrix) at the mesh's center of mass.
        Uses the covariance method for triangle meshes.
        Returns inertia matrix as ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))
        in world coordinates, at the mesh's center of mass.
        """
        from mathutils import Vector, Matrix  # type: ignore
        
        if obj is None or obj.type != 'MESH' or mass <= 0:
            return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        
        dg = self.depsgraph()
        try:
            tmp_mesh = bpy.data.meshes.new_from_object(object=obj, preserve_all_data_layers=True, depsgraph=dg)
        except TypeError:
            tmp_mesh = bpy.data.meshes.new_from_object(obj, True, dg)
        
        bm = bmesh.new()
        bm.from_mesh(tmp_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.verts.ensure_lookup_table()
        
        world_mat = self.world_mat(obj)
        
        # First compute volume and CoM
        total_volume = 0.0
        weighted_com = Vector((0, 0, 0))
        
        for face in bm.faces:
            if len(face.verts) != 3:
                continue
            v0 = world_mat @ face.verts[0].co
            v1 = world_mat @ face.verts[1].co
            v2 = world_mat @ face.verts[2].co
            cross = v1.cross(v2)
            signed_vol = v0.dot(cross) / 6.0
            total_volume += signed_vol
            tet_com = (v0 + v1 + v2) / 4.0
            weighted_com += tet_com * signed_vol
        
        if abs(total_volume) < 1e-12:
            bm.free()
            bpy.data.meshes.remove(tmp_mesh)
            return ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        
        com = weighted_com / total_volume
        density = mass / abs(total_volume)
        
        # Compute inertia tensor at origin, then shift to CoM
        # Using canonical tetrahedron inertia formulas
        Ixx, Iyy, Izz = 0.0, 0.0, 0.0
        Ixy, Ixz, Iyz = 0.0, 0.0, 0.0
        
        for face in bm.faces:
            if len(face.verts) != 3:
                continue
            v0 = world_mat @ face.verts[0].co
            v1 = world_mat @ face.verts[1].co
            v2 = world_mat @ face.verts[2].co
            
            cross = v1.cross(v2)
            signed_vol = v0.dot(cross) / 6.0
            tet_mass = density * signed_vol
            
            # Covariance matrix contribution from this tetrahedron
            # For a tetrahedron with vertices at origin, v0, v1, v2:
            # The inertia contribution uses the formula for tetrahedra
            x = [0.0, v0.x, v1.x, v2.x]
            y = [0.0, v0.y, v1.y, v2.y]
            z = [0.0, v0.z, v1.z, v2.z]
            
            # Second moments
            xx = sum(x[i]*x[j] for i in range(4) for j in range(i, 4)) * 2.0 / 20.0
            yy = sum(y[i]*y[j] for i in range(4) for j in range(i, 4)) * 2.0 / 20.0
            zz = sum(z[i]*z[j] for i in range(4) for j in range(i, 4)) * 2.0 / 20.0
            xy = sum(x[i]*y[j] + x[j]*y[i] for i in range(4) for j in range(i, 4)) / 20.0
            xz = sum(x[i]*z[j] + x[j]*z[i] for i in range(4) for j in range(i, 4)) / 20.0
            yz = sum(y[i]*z[j] + y[j]*z[i] for i in range(4) for j in range(i, 4)) / 20.0
            
            Ixx += tet_mass * (yy + zz)
            Iyy += tet_mass * (xx + zz)
            Izz += tet_mass * (xx + yy)
            Ixy -= tet_mass * xy
            Ixz -= tet_mass * xz
            Iyz -= tet_mass * yz
        
        bm.free()
        bpy.data.meshes.remove(tmp_mesh)
        
        # Shift inertia from origin to CoM using parallel axis theorem (in reverse)
        # I_com = I_origin - m * d^2 (where d is distance matrix)
        cx, cy, cz = com.x, com.y, com.z
        Ixx -= mass * (cy*cy + cz*cz)
        Iyy -= mass * (cx*cx + cz*cz)
        Izz -= mass * (cx*cx + cy*cy)
        Ixy += mass * cx * cy
        Ixz += mass * cx * cz
        Iyz += mass * cy * cz
        
        return ((Ixx, Ixy, Ixz), (Ixy, Iyy, Iyz), (Ixz, Iyz, Izz))

    def compute_link_inertial(self, link_dict, default_density_kg_m3=1100.0, motor_density_kg_m3=1500.0):
        """
        Compute inertial properties for a URDF link based on its viz_ objects.
        
        Args:
            link_dict: Link dictionary with 'frame_world', 'visuals', 'collisions'
            default_density_kg_m3: Default density (1.1 g/cm³ = 1100 kg/m³)
            motor_density_kg_m3: Density for motor colliders (1.5 g/cm³ = 1500 kg/m³)
        
        Returns:
            dict with 'origin_xyz', 'origin_rpy', 'mass', 'ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'
            or None if no valid objects found
        """
        from mathutils import Vector  # type: ignore
        
        def get_density_for_collider(coll):
            """Return appropriate density based on collider name."""
            name = (coll.name or "").lower()
            if "motor" in name:
                return motor_density_kg_m3
            return default_density_kg_m3
        
        # We need to find the viz_ objects associated with this link
        # The link_dict contains visuals list, but we need the actual Blender objects
        # We'll use a different approach: store viz objects when building link
        
        viz_objects = link_dict.get("viz_objects", [])
        if not viz_objects:
            return None
        
        total_mass = 0.0
        weighted_com = Vector((0, 0, 0))
        viz_data = []  # [(viz_obj, mass, com, colliders, collider_masses)]
        
        for viz_obj in viz_objects:
            # Find all collider_ children
            colliders = self.children_with_key(viz_obj, "collider_", mesh_only=True)
            if not colliders:
                # If no colliders, use the viz object itself
                colliders = [viz_obj] if viz_obj.type == 'MESH' else []
            
            # Calculate mass and CoM from colliders with appropriate densities
            viz_mass = 0.0
            viz_com_weighted = Vector((0, 0, 0))
            collider_masses = {}
            for coll in colliders:
                vol, com = self.compute_mesh_volume_and_com(coll)
                density = get_density_for_collider(coll)
                coll_mass = vol * density
                collider_masses[coll] = coll_mass
                viz_mass += coll_mass
                viz_com_weighted += com * coll_mass
            
            if viz_mass > 1e-12:
                viz_com = viz_com_weighted / viz_mass
            else:
                # Fallback: use viz object origin as CoM with minimal mass
                viz_com = self.world_loc(viz_obj)
                viz_mass = 0.001  # 1 gram minimum
                collider_masses = {c: 0.001 / max(len(colliders), 1) for c in colliders}
            
            viz_data.append((viz_obj, viz_mass, viz_com, colliders, collider_masses))
            total_mass += viz_mass
            weighted_com += viz_com * viz_mass
        
        if total_mass < 1e-12:
            return None
        
        # Overall CoM in world coordinates
        overall_com = weighted_com / total_mass
        
        # Compute combined inertia at overall CoM using parallel axis theorem
        Ixx, Iyy, Izz = 0.0, 0.0, 0.0
        Ixy, Ixz, Iyz = 0.0, 0.0, 0.0
        
        for viz_obj, viz_mass, viz_com, colliders, collider_masses in viz_data:
            for coll in colliders:
                # Get this collider's pre-computed mass (already accounts for density)
                coll_mass = collider_masses.get(coll, 0.0)
                
                if coll_mass < 1e-12:
                    continue
                
                # Inertia at collider's own CoM
                I = self.compute_mesh_inertia_at_com(coll, coll_mass)
                
                # Shift to overall CoM using parallel axis theorem
                _, coll_com = self.compute_mesh_volume_and_com(coll)
                dx = coll_com.x - overall_com.x
                dy = coll_com.y - overall_com.y
                dz = coll_com.z - overall_com.z
                
                Ixx += I[0][0] + coll_mass * (dy*dy + dz*dz)
                Iyy += I[1][1] + coll_mass * (dx*dx + dz*dz)
                Izz += I[2][2] + coll_mass * (dx*dx + dy*dy)
                Ixy += I[0][1] - coll_mass * dx * dy
                Ixz += I[0][2] - coll_mass * dx * dz
                Iyz += I[1][2] - coll_mass * dy * dz
        
        # Convert CoM to link-local coordinates
        link_frame = link_dict["frame_world"]
        com_local_t, com_local_r = self.xyzrpy_from_world(link_frame, self.Matrix.Translation(overall_com))
        
        return {
            "origin_xyz": (com_local_t.x, com_local_t.y, com_local_t.z),
            "origin_rpy": com_local_r,
            "mass": total_mass,
            "ixx": abs(Ixx),
            "ixy": Ixy,
            "ixz": Ixz,
            "iyy": abs(Iyy),
            "iyz": Iyz,
            "izz": abs(Izz),
        }

    def material_name(self, obj):
        if not obj.material_slots:
            print("+++++++++++++++++++++++WARNING: material not specifiec for attachments+++++++++++++++++++++++++++")
            return ""
        m = obj.material_slots[0].material
        return m.name if m else ""

    def material_key(self, obj):
        """
        Normalized material name for robust comparisons.
        Blender frequently creates duplicated materials like 'Mat', 'Mat.001', 'Mat.002', ...
        For URDF linking rules we treat those numeric suffixes as the same base material.
        """
        import re

        name = (self.material_name(obj) or "").strip()
        if not name:
            return ""
        name = name.lower()
        # Strip Blender numeric copy suffixes: ".001", ".002", etc.
        name = re.sub(r"\.\d+$", "", name)
        return name

    def object_name_key(self, name: str, *, strip_instance_suffix: bool, keep_blender_copy_index: bool):
        """
        Normalize Blender object names for shared-mesh export:
        - Optionally strip per-instance suffix added by assembly: ..._<f|t><elem_counter>
        - Handle Blender numeric copy suffixes: ".001", ".002", etc.
          - If keep_blender_copy_index=True: convert ".001" -> "_1" (underscore + int)
          - Else: strip it entirely (treat as identical copies)

        Examples:
        - keep_blender_copy_index=False: 'viz_motor_w.002_f0' -> 'viz_motor_w'
        - keep_blender_copy_index=True:  'viz_motor_w.002_f0' -> 'viz_motor_w_2'
        - 'viz_pad_ws_l_p_t12' -> 'viz_pad_ws_l_p'
        - keep_blender_copy_index=False: 'viz_motor_x.001' -> 'viz_motor_x'
        - keep_blender_copy_index=True:  'viz_motor_x.001' -> 'viz_motor_x_1'
        """
        import re
        n = (name or "").strip()
        if not n:
            return ""
        # Handle Blender numeric copy suffix possibly before the instance suffix:
        #   foo.001_f12 -> foo_1_f12
        #   foo.001     -> foo_1
        #   foo_f12     -> foo_f12
        def _copy_suffix_to_underscore(s: str) -> str:
            m2 = re.search(r"\.(\d+)(?=(_[ft]\d+)?$)", s, flags=re.IGNORECASE)
            if not m2:
                return s
            idx = int(m2.group(1))
            return re.sub(r"\.\d+(?=(_[ft]\d+)?$)", f"_{idx}", s, flags=re.IGNORECASE)

        def _strip_copy_suffix(s: str) -> str:
            return re.sub(r"\.\d+(?=(_[ft]\d+)?$)", "", s, flags=re.IGNORECASE)

        if keep_blender_copy_index:
            n = _copy_suffix_to_underscore(n)
        else:
            n = _strip_copy_suffix(n)

        if strip_instance_suffix:
            n = re.sub(r"_[ft]\d+$", "", n, flags=re.IGNORECASE)
        return n

    def mesh_export_name(self, obj, main_kind):
        """
        Decide export filename stem for a mesh object.
        - Pads must be exported separately (keep _f/_t instance suffix).
        - Link mains: keep Blender copy index but convert .001 -> _1.
        - Non-link mains: strip Blender copy index (treat copies as same), and strip _f/_t for sharing.
        """
        nm = str(getattr(obj, "name", ""))
        is_pad = nm.lower().startswith("viz_pad_") or nm.lower().startswith("collider_pad_")
        if is_pad:
            # Separate per pad instance; keep _f/_t, but make Blender duplicates underscore-indexed.
            return self.object_name_key(nm, strip_instance_suffix=False, keep_blender_copy_index=True)

        if str(main_kind) == "link":
            return self.object_name_key(nm, strip_instance_suffix=True, keep_blender_copy_index=True)

        return self.object_name_key(nm, strip_instance_suffix=True, keep_blender_copy_index=False)

    # ---------- robust, operator-free STL export (modifiers applied) ----------
    def export_mesh_STL(self, obj, out_dir, fname=None):
        """
        Export obj's mesh (modifiers applied) in LOCAL coordinates to out_dir/fname.stl
        """
        import struct
        os.makedirs(out_dir, exist_ok=True)
        if fname is None:
            fname = obj.name
        filepath = os.path.join(out_dir, f"{fname}.stl")

        dg = self.depsgraph()
        try:
            tmp_mesh = bpy.data.meshes.new_from_object(object=obj, preserve_all_data_layers=True, depsgraph=dg)
        except TypeError:
            tmp_mesh = bpy.data.meshes.new_from_object(obj, True, dg)

        bm = bmesh.new()
        bm.from_mesh(tmp_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.normal_update()

        with open(filepath, "wb") as f:
            header = b"Created by Blender Python (binary STL)"
            f.write(header + b"\0" * (80 - len(header)))
            tri_count = sum(1 for face in bm.faces if len(face.verts) == 3)
            f.write(struct.pack("<I", tri_count))
            for face in bm.faces:
                if len(face.verts) != 3:
                    continue
                n = face.normal
                f.write(struct.pack("<3f", n.x, n.y, n.z))
                vs = [v.co for v in face.verts]
                for v in vs:
                    f.write(struct.pack("<3f", v.x, v.y, v.z))
                f.write(struct.pack("<H", 0))
        bm.free()
        if tmp_mesh.users == 0:
            bpy.data.meshes.remove(tmp_mesh, do_unlink=True)
        return filepath

    # -------------- URDF writer --------------
    def write_urdf(self, path, robot_name, links, joints, package_name):
        def fmt3(v):
            return " ".join(f"{x:.9g}" for x in v)

        lines = ['<?xml version="1.0"?>', f'<robot name="{robot_name}">']
        for L in links:
            lines.append(f'  <link name="{L["name"]}">')
            
            # Add inertial properties if available
            inertial = self.compute_link_inertial(L)
            if inertial is not None:
                lines.append('    <inertial>')
                lines.append(f'      <origin xyz="{fmt3(inertial["origin_xyz"])}" rpy="{fmt3(inertial["origin_rpy"])}"/>')
                if L["name"] == "base_link":
                    lines.append(f'      <mass value="10000"/>')
                    lines.append(f'      <inertia ixx="10000" ixy="0" ixz="0" iyy="10000" iyz="0" izz="10000"/>')

                else:
                    lines.append(f'      <mass value="{inertial["mass"]:.9g}"/>')
                    lines.append(f'      <inertia ixx="{inertial["ixx"]:.9g}" ixy="{inertial["ixy"]:.9g}" ixz="{inertial["ixz"]:.9g}" iyy="{inertial["iyy"]:.9g}" iyz="{inertial["iyz"]:.9g}" izz="{inertial["izz"]:.9g}"/>')
                lines.append('    </inertial>')
            
            for (mesh_rel, xyz, rpy) in L.get("visuals", []):
                lines.append('    <visual>')
                lines.append(f'      <origin xyz="{fmt3(xyz)}" rpy="{fmt3(rpy)}"/>')
                # Write mesh path as-is (no package:// prefix)
                lines.append(f'      <geometry><mesh filename="{mesh_rel}"/></geometry>')
                lines.append('    </visual>')
            if self.collision_mesh:
                for (mesh_rel, xyz, rpy) in L.get("collisions", []):
                    lines.append('    <collision>')
                    lines.append(f'      <origin xyz="{fmt3(xyz)}" rpy="{fmt3(rpy)}"/>')
                    # Write mesh path as-is (no package:// prefix)
                    lines.append(f'      <geometry><mesh filename="{mesh_rel}"/></geometry>')
                    lines.append('    </collision>')
            lines.append('  </link>')
        for J in joints:
            jtype = J.get("type", "revolute")
            lines.append(f'  <joint name="{J["name"]}" type="{jtype}">')
            lines.append(f'    <parent link="{J["parent"]}"/>')
            lines.append(f'    <child link="{J["child"]}"/>')
            lim = J.get("limit") or {}
            eff = float(lim.get("effort", self.default_joint_limit["effort"]))
            vel = float(lim.get("velocity", self.default_joint_limit["velocity"]))
            low = float(lim.get("lower", self.default_joint_limit["lower"]))
            up = float(lim.get("upper", self.default_joint_limit["upper"]))
            lines.append(f'    <limit effort="{eff:.9g}" velocity="{vel:.9g}" lower="{low:.9g}" upper="{up:.9g}"/>')
            # lines.append(f'    <dynamics damping="1.5e-1" friction="2" stiffness="2.0"/>')

            lines.append(f'    <origin xyz="{fmt3(J["origin_xyz"])}" rpy="{fmt3(J["origin_rpy"])}"/>')
            if "axis" in J and J["axis"] is not None:
                lines.append(f'    <axis xyz="{fmt3(J["axis"])}"/>')
            lines.append('  </joint>')
        lines.append('</robot>')
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # -------------- Rule helpers (ported) --------------
    def nearest_attach_pair(self, prev_attaches, candidate_attaches):
        best = (None, None, float("inf"))
        for a in prev_attaches:
            pa = self.world_loc(a)
            for b in candidate_attaches:
                d = (pa - self.world_loc(b)).length
                if d < best[2]:
                    best = (a, b, d)
        return best

    def joint_axis_in_joint_frame_local(self, rev_obj):
        from mathutils import Vector  # type: ignore
        dims = getattr(rev_obj, "dimensions", Vector((0, 0, 0)))
        sizes = (dims.x, dims.y, dims.z)
        idx = max(range(3), key=lambda i: sizes[i])
        if idx == 0:
            return Vector((1, 0, 0))
        if idx == 1:
            return Vector((0, 1, 0))
        return Vector((0, 0, 1))

    def export_viz_and_colliders_into_link(self, link_dict, main_obj, only_these_viz=None, scale_suffix=""):
        # Relative to link frame
        link_frame = link_dict["frame_world"]
        main_kind = self.type_of_main(main_obj)
        if only_these_viz is None:
            viz_parents = [v for v in self.children_with_key(main_obj, "viz_", mesh_only=False)]
        else:
            viz_parents = list(only_these_viz)

        # Initialize viz_objects list if not present
        if "viz_objects" not in link_dict:
            link_dict["viz_objects"] = []

        seen_vis = set()
        seen_col = set()
        for vp in viz_parents:
            for v in self.children_with_key(vp, "viz_", mesh_only=True):
                if v in seen_vis:
                    continue
                export_name = self.mesh_export_name(v, main_kind) + scale_suffix
                rel = f"meshes/{export_name}.stl"  # Use forward slashes for URDF compatibility
                cache_key = ("meshes", export_name)
                if cache_key not in self._mesh_rel_cache:
                    self.export_mesh_STL(v, os.path.join(self.export_dir, "meshes"), fname=export_name)
                    self._mesh_rel_cache[cache_key] = rel
                t, r = self.xyzrpy_from_world(link_frame, self.world_mat(v))
                link_dict["visuals"].append((rel, (t.x, t.y, t.z), r))
                seen_vis.add(v)
                # Store viz object for inertia calculation
                link_dict["viz_objects"].append(v)
            if self.collision_mesh:
                for c in self.children_with_key(vp, "collider_", mesh_only=True):
                    if c in seen_col:
                        continue
                    export_name = self.mesh_export_name(c, main_kind) + scale_suffix
                    rel = f"collision/{export_name}.stl"  # Use forward slashes for URDF compatibility
                    cache_key = ("collision", export_name)
                    if cache_key not in self._mesh_rel_cache:
                        self.export_mesh_STL(c, os.path.join(self.export_dir, "collision"), fname=export_name)
                        self._mesh_rel_cache[cache_key] = rel
                    t, r = self.xyzrpy_from_world(link_frame, self.world_mat(c))
                    link_dict["collisions"].append((rel, (t.x, t.y, t.z), r))
                    seen_col.add(c)

    # ---------------- Main export ----------------
    def build_urdf(self):
        """
        Palm-rooted export:
        - Base link is PalmBody
        - Each finger/thumb chain is connected to PalmBody via aligned attach_* objects on PalmBody
        - Finger/thumb chains are exported per subcollection under `hand`
        """
        from mathutils import Matrix  # type: ignore

        def new_link(name, frame_world):
            return {"name": name, "frame_world": frame_world.copy(), "visuals": [], "collisions": [], "viz_objects": []}

        palm_root = bpy.data.objects.get("PalmBody_root")
        if palm_root is None:
            raise RuntimeError('PalmBody_root not found. Expected an empty named "PalmBody_root".')
        # Use the PalmBody_root empty as the base frame
        palm_frame = self.world_mat(palm_root)
        base_link = new_link("base_link", palm_frame)

        # Export all viz_/collider_ children under PalmBody_root into the base link
        self.export_viz_and_colliders_into_link(base_link, palm_root)

        links = [base_link]
        joints = []

        # Palm attachments (child attach_* objects): attach_finger_* / attach_thumb_*
        palm_attaches_all = [a for a in self.children_with_key(palm_root, "attach_", mesh_only=True)]
        palm_attaches = [a for a in palm_attaches_all
                         if a.name.lower().startswith("attach_finger_") or a.name.lower().startswith("attach_thumb_")]
        if not palm_attaches:
            print('[URDF][warn] PalmBody has no attach_finger_/attach_thumb_ children; no fingers/thumbs will be connected.')

        # Helper: collect first-order "main objects" inside a specific collection (recursively)
        def first_order_objects_in_collection(coll):
            def _collect_objects(c):
                out = list(c.objects)
                for ch in c.children:
                    out.extend(_collect_objects(ch))
                return out
            objs = _collect_objects(coll)
            immediate = [o for o in objs if (o.parent is None) or (o.parent not in objs)]
            immediate.sort(key=lambda o: o.name.lower())
            return immediate

        def _all_attach_objects_in_collection(chain_coll):
            """Collect all attach_* mesh objects under a chain collection."""
            objs = first_order_objects_in_collection(chain_coll)
            attaches = []
            for o in objs:
                attaches.extend(self.children_with_key(o, "attach_", mesh_only=True))
            return attaches

        def _min_attach_distance(palm_attach_obj, chain_coll):
            """Minimum world distance between palm_attach_obj and any attach_ under chain_coll."""
            attaches = _all_attach_objects_in_collection(chain_coll)
            if not attaches:
                return float("inf")
            pa = self.world_loc(palm_attach_obj)
            return min((pa - self.world_loc(a)).length for a in attaches)

        # Helper: export one finger/thumb chain seeded from a PalmBody attach_* object
        def export_chain_for_collection(chain_coll, chain_name: str, palm_attach_obj):
            mains = [o for o in first_order_objects_in_collection(chain_coll) if self.type_of_main(o)]
            if not mains:
                print(f'[URDF][warn] No main objects found in collection "{chain_coll.name}", skipping.')
                return

            # Naming:
            # - base_link stays "base_link"
            # - links:  L_f{n}_{m}  (thumbs: L_t{n}_{m}), where m starts at 1 from base_link
            # - joints: J_f{n}_{m}_{k} (thumbs: J_t{n}_{m}_{k}), where m starts at 1 from base_link, k is joint self.id
            def _chain_token(name: str):
                s = str(name or "").lower()
                try:
                    if s.startswith("finger_"):
                        return f"f{int(s.split('_')[-1])}"
                    if s.startswith("thumb_"):
                        return f"t{int(s.split('_')[-1])}"
                except Exception:
                    pass
                # fallback from collection name
                try:
                    cs = str(chain_coll.name or "").lower()
                    if cs.startswith("finger_"):
                        return f"f{int(cs.split('_')[-1])}"
                    if cs.startswith("thumb_"):
                        return f"t{int(cs.split('_')[-1])}"
                except Exception:
                    pass
                return "f0"

            chain_tok = _chain_token(chain_name)
            link_ord = 0   # created URDF links after base_link in this chain
            joint_ord = 0  # created URDF joints in this chain

            def new_chain_link(frame_world):
                nonlocal link_ord
                link_ord += 1
                return new_link(f"L_{chain_tok}_{link_ord}", frame_world)

            # Per README: the first matched main object can be `link_*` and must stay in the SAME
            # URDF link as the base (PalmBody). So we do NOT create an extra per-chain base link.
            # We only seed the matching using the palm attachment pose.
            visited = set()
            last_main = palm_attach_obj
            last_main_mainobj = None  # last matched "main object" (link_/n*/w*). Palm attach is not a main object.
            last_attaches = [palm_attach_obj]
            used_attaches = set([palm_attach_obj])
            current_link = base_link
            joints_start = len(joints)

            def remaining_attaches_excluding(mo, used):
                xs = [a for a in self.children_with_key(mo, "attach_", mesh_only=True)]
                return [a for a in xs if a not in used]

            pending_rev = None
            pending_parent_link = None

            def first_revolute(main_obj):
                if main_obj is None:
                    return None
                revs = self.children_with_key(main_obj, "revolute", mesh_only=True)
                return revs[0] if revs else None

            def _joint_limit_for_revolute_obj(rev_obj):
                """
                Map a Blender revolute object's name to a saved joint element (from assembly_data.npz)
                by parsing the per-element suffix: ..._<f|t><elem_counter>
                """
                if rev_obj is None:
                    return None
                try:
                    import re

                    name = str(getattr(rev_obj, "name", ""))
                    # Strip Blender duplicate suffix: ".001", ".002", ...
                    name = re.sub(r"\.\d+$", "", name)
                    m = re.search(r"_(?P<gi>[ft])(?P<idx>\d+)$", name.lower())
                    if not m:
                        return None
                    gi = m.group("gi")
                    idx = int(m.group("idx"))
                    return self.joint_limit_by_instance.get((gi, idx))
                except Exception:
                    return None

            def _joint_id_for_revolute_obj(rev_obj):
                """Return self.id (k) for this revolute, if available from assembly_data.npz."""
                if rev_obj is None:
                    return None
                try:
                    import re
                    name = str(getattr(rev_obj, "name", ""))
                    # Strip Blender duplicate suffix: ".001", ".002", ...
                    name = re.sub(r"\.\d+$", "", name)
                    m = re.search(r"_(?P<gi>[ft])(?P<idx>\d+)$", name.lower())
                    if not m:
                        return None
                    gi = m.group("gi")
                    idx = int(m.group("idx"))
                    return self.joint_id_by_instance.get((gi, idx))
                except Exception:
                    return None

            def add_revolute_joint(parent_link, child_link, rev_obj):
                if rev_obj is None:
                    return
                nonlocal joint_ord
                joint_ord += 1
                j_t, j_r = self.xyzrpy_from_world(parent_link["frame_world"], self.world_mat(rev_obj))
                axis = self.joint_axis_in_joint_frame_local(rev_obj)
                lim = _joint_limit_for_revolute_obj(rev_obj) or self.default_joint_limit
                k = _joint_id_for_revolute_obj(rev_obj)
                k = int(k) if k is not None else 0
                joints.append({
                    "name": f"J_{chain_tok}_{joint_ord}_{k}",
                    "type": "revolute",
                    "parent": parent_link["name"],
                    "child": child_link["name"],
                    "origin_xyz": (j_t.x, j_t.y, j_t.z),
                    "origin_rpy": j_r,
                    "axis": (axis.x, axis.y, axis.z),
                    "limit": lim,
                })

            while True:
                candidates = [mo for mo in mains if mo not in visited]
                if not candidates:
                    break

                best = (None, None, None, float("inf"))
                for mo in candidates:
                    attaches = self.children_with_key(mo, "attach_", mesh_only=True)
                    if not attaches:
                        continue
                    a_prev, b_cand, d = self.nearest_attach_pair(last_attaches, attaches)
                    if a_prev and b_cand and d < best[3]:
                        best = (mo, a_prev, b_cand, d)

                # Fallback: if we can't find a next match using the "remaining attaches" set, try again
                # with *all* attaches on the last main object. This helps with thumb cases where the
                # greedy choice of which attach is "remaining" can cause the chain to dead-end.
                if best[0] is None or best[3] > self.attach_thresh:
                    last_obj_for_fallback = last_main_mainobj if last_main_mainobj is not None else last_main
                    prev_all = self.children_with_key(last_obj_for_fallback, "attach_", mesh_only=True) if last_obj_for_fallback is not None else []
                    if prev_all:
                        best2 = (None, None, None, float("inf"))
                        for mo in candidates:
                            attaches = self.children_with_key(mo, "attach_", mesh_only=True)
                            if not attaches:
                                continue
                            a_prev2, b_cand2, d2 = self.nearest_attach_pair(prev_all, attaches)
                            if a_prev2 and b_cand2 and d2 < best2[3]:
                                best2 = (mo, a_prev2, b_cand2, d2)
                        if best2[0] is not None and best2[3] <= self.attach_thresh:
                            best = best2

                if best[0] is None or best[3] > self.attach_thresh:
                    break

                mo, a_prev, b_this, dist = best
                visited.add(mo)
                # Mark both ends of the matched attach pair as used to avoid accidental re-use loops
                if a_prev is not None:
                    used_attaches.add(a_prev)
                used_attaches.add(b_this)
                kind = self.type_of_main(mo)
                # print(f"[URDF] [{chain_name}] match: {last_main.name} -> {mo.name} via {a_prev.name} ~ {b_this.name} (d={dist*1000:.2f} mm) [{kind}]")

                # Material comparison for urdf_description.md rules.
                # The description defines two cases: "same material" vs "materials differ".
                # If one/both attachments have no material assigned, we treat that as "materials differ"
                # (conservative; avoids accidentally merging links when metadata is missing).
                mat_prev = self.material_key(a_prev)
                mat_this = self.material_key(b_this)
                mats_same = (mat_prev != "" and mat_this != "" and mat_prev == mat_this)
                mats_differ = not mats_same

                # Deferred nj case (same behavior as urdf_exp.py)
                if pending_rev is not None:
                    if pending_parent_link is None:
                        pending_parent_link = current_link

                    # Create the deferred joint (from the previous n*'s revolute) and advance to the new link,
                    # BUT do not consume the current `mo` here. The current `mo` (which may itself be an n*/w*)
                    # still needs to run through the normal logic below so it can generate its own joint-to-next.
                    child_frame2 = self.world_mat(pending_rev)
                    child_link2 = new_chain_link(child_frame2)
                    links.append(child_link2)
                    add_revolute_joint(pending_parent_link, child_link2, pending_rev)

                    current_link = child_link2
                    pending_rev = None
                    pending_parent_link = None

                if kind == "link":
                    # README.md: link_* itself does not define revolute; it is generally part of the
                    # current URDF link (material-based splitting for link_* is not fully specified there).
                    _link_scale_suffix = ""
                    _sx = float(mo.get("mesh_scale_factor_x", 1.0))
                    _sy = float(mo.get("mesh_scale_factor_y", 1.0))
                    _sz = float(mo.get("mesh_scale_factor_z", 1.0))
                    _is_default = all(abs(s - 1.0) < 1e-9 for s in (_sx, _sy, _sz))
                    if not _is_default:
                        _link_scale_suffix = f"_s{int(round(_sx*1000))}_{int(round(_sy*1000))}_{int(round(_sz*1000))}"
                    self.export_viz_and_colliders_into_link(current_link, mo, scale_suffix=_link_scale_suffix)
                    last_main = mo
                    last_main_mainobj = mo
                    last_attaches = remaining_attaches_excluding(last_main, used_attaches)

                elif kind == "n":
                    vp_this = b_this.parent
                    same_material = mats_same

                    revs = self.children_with_key(mo, "revolute", mesh_only=True)
                    rev = revs[0] if revs else None
                    if rev is None:
                        print(f"[URDF][warn] {mo.name}: nj has no revolute mesh; behavior may be undefined.")

                    if same_material and vp_this:
                        # urdf_description.md:
                        # - parent viz_ containing aligned attach_ and its siblings remain in the last URDF link
                        parent_of_viz = vp_this.parent
                        sibling_viz = [v for v in self.children_with_key(mo, "viz_", mesh_only=False) if v.parent == parent_of_viz]
                        self.export_viz_and_colliders_into_link(current_link, mo, only_these_viz=set(sibling_viz))
                        # Use this n*'s revolute to connect CURRENT -> NEXT (deferred until the next main object is matched).
                        if rev is not None:
                            pending_rev = rev
                            pending_parent_link = current_link
                        last_main = mo
                        last_main_mainobj = mo
                        last_attaches = remaining_attaches_excluding(last_main, used_attaches)
                    else:
                        # Unmatched attach-types/materials case:
                        # Put ALL viz_ of this n* into the CURRENT link, and use THIS n*'s revolute
                        # to connect CURRENT -> NEXT (deferred until next main is matched).
                        if mats_differ:
                            print(f"[URDF] [{chain_name}] {mo.name}: n* attach/material mismatch ({mat_prev!r} != {mat_this!r}) -> joint parent=current, child=next")
                        self.export_viz_and_colliders_into_link(current_link, mo)
                        if rev is not None:
                            pending_rev = rev
                            pending_parent_link = current_link
                        last_main = mo
                        last_main_mainobj = mo
                        last_attaches = remaining_attaches_excluding(last_main, used_attaches)

                elif kind == "w":
                    vp_this = b_this.parent
                    revs = self.children_with_key(mo, "revolute", mesh_only=True)
                    if vp_this is None:
                        child_frame4 = self.world_mat(revs[0]) if revs else Matrix.Identity(4)
                        child_link4 = new_chain_link(child_frame4)
                        self.export_viz_and_colliders_into_link(child_link4, mo)
                        links.append(child_link4)
                        if revs:
                            rev = revs[0]
                            add_revolute_joint(current_link, child_link4, rev)
                        last_main = mo
                        last_main_mainobj = mo
                        last_attaches = remaining_attaches_excluding(last_main, used_attaches)
                        current_link = child_link4
                    else:
                        parent_of_viz = vp_this.parent
                        sibling_viz = [v for v in self.children_with_key(mo, "viz_", mesh_only=False) if v.parent == parent_of_viz]
                        other_viz = [v for v in self.children_with_key(mo, "viz_", mesh_only=False) if v.parent != parent_of_viz]
                        # README.md does not define a material-based split for w*.
                        # Always apply the sibling-level-viz stays / other-level-viz splits via w*'s revolute.
                        if sibling_viz:
                            self.export_viz_and_colliders_into_link(current_link, mo, only_these_viz=set(sibling_viz))
                        if other_viz:
                            child_frame5 = self.world_mat(revs[0]) if revs else self.world_mat(other_viz[0])
                            child_link5 = new_chain_link(child_frame5)
                            self.export_viz_and_colliders_into_link(child_link5, mo, only_these_viz=set(other_viz))
                            links.append(child_link5)
                            if revs:
                                rev = revs[0]
                                add_revolute_joint(current_link, child_link5, rev)
                            current_link = child_link5
                        last_main = mo
                        last_main_mainobj = mo
                        last_attaches = remaining_attaches_excluding(last_main, used_attaches)
                else:
                    print(f"[URDF][warn] Unknown main type for {mo.name}, skipping.")
                    continue

            # Diagnostics: if we still have a pending n* revolute, it means the chain ended before
            # we found the "next main object" to attach to.
            if pending_rev is not None:
                print(f"[URDF][warn] [{chain_name}] Unresolved pending n* revolute '{pending_rev.name}'. Chain likely ended early due to attach matching.")

            # Diagnostics: expected vs actual joint count per chain (uses GenerateHand's elem_types in assembly_data.npz).
            expected = self.expected_joint_count_by_subgroup.get(chain_name)
            actual = int(len(joints) - joints_start)
            if expected is not None and expected != actual:
                print(f"[URDF][warn] [{chain_name}] Joint count mismatch: expected {expected} joint elements, but URDF exporter created {actual} joints.")

        # Iterate hand subcollections (finger_*, thumb_*)
        hand_coll = bpy.data.collections.get(self.collection_name)
        if hand_coll is None:
            raise RuntimeError(f'Collection "{self.collection_name}" not found.')

        # Assign each palm attachment to the closest subcollection (finger_* / thumb_*),
        # but only if it's within the threshold (0.5 mm by default).
        subcolls = list(hand_coll.children)
        used_subcolls = set()
        for pa in sorted(palm_attaches, key=lambda o: o.name.lower()):
            best = (None, float("inf"))
            for sub in subcolls:
                if sub.name in used_subcolls:
                    continue
                d = _min_attach_distance(pa, sub)
                if d < best[1]:
                    best = (sub, d)
            if best[0] is None:
                print(f'[URDF][warn] No matching finger/thumb collection found for palm attachment "{pa.name}".')
                continue
            sub, d = best
            if d > self.attach_thresh:
                print(f'[URDF][warn] Palm attach "{pa.name}" has no finger/thumb attach within thresh ({self.attach_thresh*1000:.2f} mm). Best was {d*1000:.2f} mm; skipping.')
                continue
            used_subcolls.add(sub.name)
            # print(f'[URDF] Palm attach "{pa.name}" -> collection "{sub.name}" (min_d={d*1000:.2f} mm)')
            export_chain_for_collection(sub, sub.name, pa)

        os.makedirs(self.export_dir, exist_ok=True)
        urdf_path = os.path.join(self.export_dir, f"{self.robot_name}.urdf")
        self.write_urdf(urdf_path, self.robot_name, links, joints, self.robot_package)
        print(f"[URDF] Wrote {urdf_path}")
        print("[URDF] Links:", [L["name"] for L in links])
        print("[URDF] Joints:", [J["name"] for J in joints])

    def run(self):
        # Ensure object mode
        if bpy.ops.object.mode_set.poll():
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        self.build_urdf()

if __name__ == "__main__":
    # Blender passes args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    # argv[0] = assembly_data.npz path
    # argv[1] = output_dir (optional)
    assembly_npz = argv[0] if len(argv) >= 1 and argv[0] else os.path.join(os.path.dirname(__file__), "blender", "assembly_data.npz")
    assembly_npz = os.path.abspath(assembly_npz)

    out_dir = argv[1] if len(argv) >= 2 and argv[1] else None
    if out_dir is None:
        # default: parent of the "blender" folder containing the npz
        out_dir = os.path.abspath(os.path.join(os.path.dirname(assembly_npz), os.pardir))
    else:
        out_dir = os.path.abspath(out_dir)

    finger_assembly = FingerAssembly(assembly_npz)
    finger_assembly.run_assembly()
    palm_npz = os.path.join(os.path.dirname(assembly_npz), "palm_outlines_data.npz")
    PalmMesh(palm_npz).generate()

    assign_viz_materials()

    # Apply origin offset before URDF export (moves PalmBody_root, keeps meshes in place)
    apply_origin_offset(palm_npz)
    save_blend_file(out_dir)
    
    ExportURDF(os.path.dirname(__file__), assembly_npz_path=assembly_npz, export_dir=os.path.join(out_dir, "urdf_hand_export")).run()



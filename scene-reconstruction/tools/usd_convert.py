#!/usr/bin/env python3
"""Convert a 3DGRUT Gaussian PLY to a NuRec USDZ bundle for Isaac Sim / Omniverse.

Requires pxr (OpenUSD) and msgpack. If host Python is missing them,
the helper auto-reruns with Isaac Sim's python.sh when available.

Usage:
    python tools/usd_convert.py scene.ply
    python tools/usd_convert.py scene.ply scene.usdz
    python tools/usd_convert.py scene.ply scene.usdz --extract-sidecars
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def _find_isaac_python() -> Path | None:
    candidates = []
    if os.environ.get("USD_CONVERT_PYTHON"):
        candidates.append(Path(os.environ["USD_CONVERT_PYTHON"]).expanduser())
    if os.environ.get("ISAACSIM_PATH"):
        candidates.append(Path(os.environ["ISAACSIM_PATH"]).expanduser() / "python.sh")
    candidates += [
        Path.home() / "IsaacSim/_build/linux-aarch64/release/python.sh",
        Path("/isaac-sim/python.sh"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _check_deps():
    missing = []
    try:
        import msgpack  # noqa: F401
    except ImportError:
        missing.append("msgpack")
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        missing.append("pxr")
    if not missing:
        return

    isaac_python = _find_isaac_python()
    if isaac_python and not os.environ.get("USD_CONVERT_NO_REEXEC"):
        env = os.environ.copy()
        env["USD_CONVERT_NO_REEXEC"] = "1"
        print(f"Host Python is missing {', '.join(missing)}; rerunning with {isaac_python}")
        os.execve(str(isaac_python), [str(isaac_python), *sys.argv], env)

    install_hint = "Use Isaac Sim's python.sh, or set USD_CONVERT_PYTHON=/path/to/python.sh."
    print(
        "Error: missing dependencies:\n  "
        + "\n  ".join(missing)
        + f"\n{install_hint}",
        file=sys.stderr,
    )
    sys.exit(1)


def _read_supersplat_compressed_ply(path: Path):
    """Decode SuperSplat chunk-compressed PLY (element chunk + vertex + sh)."""
    import numpy as np

    chunk_dtype = np.dtype([
        ("min_x", "<f4"), ("min_y", "<f4"), ("min_z", "<f4"),
        ("max_x", "<f4"), ("max_y", "<f4"), ("max_z", "<f4"),
        ("min_scale_x", "<f4"), ("min_scale_y", "<f4"), ("min_scale_z", "<f4"),
        ("max_scale_x", "<f4"), ("max_scale_y", "<f4"), ("max_scale_z", "<f4"),
        ("min_r", "<f4"), ("min_g", "<f4"), ("min_b", "<f4"),
        ("max_r", "<f4"), ("max_g", "<f4"), ("max_b", "<f4"),
    ])
    vert_dtype = np.dtype([
        ("packed_position", "<u4"), ("packed_rotation", "<u4"),
        ("packed_scale", "<u4"),    ("packed_color", "<u4"),
    ])

    with path.open("rb") as f:
        n_chunks = n_verts = sh_count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element chunk"):
                n_chunks = int(line.split()[-1])
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property uchar f_rest_"):
                sh_count += 1
            elif line == "end_header":
                break
        chunks = np.fromfile(f, dtype=chunk_dtype, count=n_chunks)
        verts  = np.fromfile(f, dtype=vert_dtype,  count=n_verts)
        sh_dt  = np.dtype([(f"f_rest_{i}", "u1") for i in range(sh_count)])
        sh     = np.fromfile(f, dtype=sh_dt, count=n_verts)

    ci  = np.arange(n_verts, dtype=np.int64) // 256          # chunk index per vertex
    cx  = chunks[ci]                                          # broadcast chunk bounds

    # Positions: 10-10-10 bit per-chunk quantization
    pp  = verts["packed_position"]
    x   = cx["min_x"] + ((pp       ) & 0x3FF).astype(np.float32) / 1023.0 * (cx["max_x"] - cx["min_x"])
    y   = cx["min_y"] + ((pp >> 10 ) & 0x3FF).astype(np.float32) / 1023.0 * (cx["max_y"] - cx["min_y"])
    z   = cx["min_z"] + ((pp >> 20 ) & 0x3FF).astype(np.float32) / 1023.0 * (cx["max_z"] - cx["min_z"])
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # Scales: 8-8-8 bit per-chunk, stored as log-scale (same convention as standard PLY)
    ps  = verts["packed_scale"]
    sx  = cx["min_scale_x"] + ((ps      ) & 0xFF).astype(np.float32) / 255.0 * (cx["max_scale_x"] - cx["min_scale_x"])
    sy  = cx["min_scale_y"] + ((ps >>  8) & 0xFF).astype(np.float32) / 255.0 * (cx["max_scale_y"] - cx["min_scale_y"])
    sz  = cx["min_scale_z"] + ((ps >> 16) & 0xFF).astype(np.float32) / 255.0 * (cx["max_scale_z"] - cx["min_scale_z"])
    scales = np.stack([sx, sy, sz], axis=1).astype(np.float32)

    # Albedo (f_dc SH DC coefficients): 8-8-8 bit per-chunk
    pc  = verts["packed_color"]
    r   = cx["min_r"] + ((pc      ) & 0xFF).astype(np.float32) / 255.0 * (cx["max_r"] - cx["min_r"])
    g   = cx["min_g"] + ((pc >>  8) & 0xFF).astype(np.float32) / 255.0 * (cx["max_g"] - cx["min_g"])
    b   = cx["min_b"] + ((pc >> 16) & 0xFF).astype(np.float32) / 255.0 * (cx["max_b"] - cx["min_b"])
    albedo = np.stack([r, g, b], axis=1).astype(np.float32)

    # Opacity: 8-bit post-sigmoid (0-255 → 0-1) → convert to logit for NuRec density_activation=sigmoid
    opac = np.clip(((pc >> 24) & 0xFF).astype(np.float32) / 255.0, 1e-6, 1.0 - 1e-6)
    densities = np.log(opac / (1.0 - opac)).reshape(-1, 1).astype(np.float32)

    # Rotations: smallest-3 quaternion, 2-bit largest-component index + 3×10 bits
    # Output order: (w, x, y, z) matching standard 3DGS PLY convention
    pr  = verts["packed_rotation"]
    idx = (pr      ) & 0x3
    s   = np.float32(1.0 / np.sqrt(2))
    a   = ((pr >>  2) & 0x3FF).astype(np.float32) / 1023.0 * (2.0 * s) - s
    b   = ((pr >> 12) & 0x3FF).astype(np.float32) / 1023.0 * (2.0 * s) - s
    c   = ((pr >> 22) & 0x3FF).astype(np.float32) / 1023.0 * (2.0 * s) - s
    d   = np.sqrt(np.maximum(np.float32(0.0), 1.0 - a*a - b*b - c*c)).astype(np.float32)
    rots = np.empty((n_verts, 4), dtype=np.float32)
    for i, (w_col, x_col, y_col, z_col) in enumerate([(d,a,b,c),(a,d,b,c),(a,b,d,c),(a,b,c,d)]):
        m = (idx == i)
        rots[m] = np.stack([w_col[m], x_col[m], y_col[m], z_col[m]], axis=1)

    # SH rest: 24 × uint8 quantized to [-2, 2]
    spec_ch_maj = np.stack([sh[f"f_rest_{i}"] for i in range(sh_count)], axis=1).astype(np.float32)
    spec_ch_maj = spec_ch_maj / 255.0 * 4.0 - 2.0
    num_spec  = sh_count // 3
    specular  = spec_ch_maj.reshape((n_verts, 3, num_spec)).transpose(0, 2, 1).reshape((n_verts, sh_count))
    sh_degree = int(np.sqrt(num_spec + 1) - 1) if num_spec else 0

    return positions, rots, scales, densities, albedo, specular, sh_degree


def read_3dgrut_ply(path: Path):
    import numpy as np

    with path.open("rb") as f:
        props = []
        count = None
        is_supersplat = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element chunk"):
                is_supersplat = True
            elif line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                data_start = f.tell()
                break
        if count is None:
            raise RuntimeError("PLY vertex count missing")

    if is_supersplat:
        return _read_supersplat_compressed_ply(path)

    dtype = [(name, "<f4") for name in props]
    with path.open("rb") as f:
        f.seek(data_start)
        arr = np.fromfile(f, dtype=np.dtype(dtype), count=count)

    positions  = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    densities  = arr["opacity"].reshape(-1, 1).astype(np.float32)
    albedo     = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
    scale_names = sorted([p for p in props if p.startswith("scale_")], key=lambda s: int(s.split("_")[-1]))
    rot_names   = sorted([p for p in props if p.startswith("rot_")],   key=lambda s: int(s.split("_")[-1]))
    rest_names  = sorted([p for p in props if p.startswith("f_rest_")],key=lambda s: int(s.split("_")[-1]))
    scales      = np.stack([arr[n] for n in scale_names], axis=1).astype(np.float32)
    rotations   = np.stack([arr[n] for n in rot_names],   axis=1).astype(np.float32)
    spec_ch_maj = np.stack([arr[n] for n in rest_names],  axis=1).astype(np.float32)
    num_spec    = len(rest_names) // 3
    specular    = spec_ch_maj.reshape((count, 3, num_spec)).transpose(0, 2, 1).reshape((count, num_spec * 3))
    sh_degree   = int(np.sqrt(num_spec + 1) - 1) if num_spec else 0
    return positions, rotations, scales, densities, albedo, specular, sh_degree


def make_nurec_payload(positions, rotations, scales, densities, albedo, specular, sh_degree):
    import msgpack
    import numpy as np

    count = positions.shape[0]
    state_dict = {
        "._extra_state": {"obj_track_ids": {"gaussians": []}},
        ".gaussians_nodes.gaussians.positions": None,
        ".gaussians_nodes.gaussians.rotations": None,
        ".gaussians_nodes.gaussians.scales": None,
        ".gaussians_nodes.gaussians.densities": None,
        ".gaussians_nodes.gaussians.extra_signal": None,
        ".gaussians_nodes.gaussians.features_albedo": None,
        ".gaussians_nodes.gaussians.features_specular": None,
        ".gaussians_nodes.gaussians.n_active_features": None,
    }
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians", "device": "cuda",
                        "density_activation": "sigmoid", "scale_activation": "exp",
                        "rotation_activation": "normalize", "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_density_clamping": False,
                            "density_kernel_min_response": 0.0113,
                            "radiance_sph_degree": sh_degree,
                        },
                        "transmittance_threshold": 0.0001,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend", "log_level": 3, "force_update": False,
                    "update_step_train_batch_end": False, "per_ray_features": False,
                    "global_z_order": True,
                    "projection": {
                        "n_rolling_shutter_iterations": 5, "ut_dim": 3,
                        "ut_alpha": 1.0, "ut_beta": 2.0, "ut_kappa": 0.0,
                        "ut_require_all_sigma_points": False,
                        "image_margin_factor": 0.1,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": True, "tight_opacity_bounding": True,
                        "tile_based": True, "near_clip_distance": 1e-8,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": 0},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": state_dict,
        }
    }
    tensors = {
        ".gaussians_nodes.gaussians.positions":        positions,
        ".gaussians_nodes.gaussians.rotations":        rotations,
        ".gaussians_nodes.gaussians.scales":           scales,
        ".gaussians_nodes.gaussians.densities":        densities,
        ".gaussians_nodes.gaussians.features_albedo":  albedo,
        ".gaussians_nodes.gaussians.features_specular": specular,
        ".gaussians_nodes.gaussians.extra_signal":     np.zeros((count, 0), dtype=np.float32),
    }
    for key, val in tensors.items():
        state_dict[key] = val.astype(np.float16).tobytes()
        state_dict[key + ".shape"] = list(val.shape)
    state_dict[".gaussians_nodes.gaussians.n_active_features"] = np.array([sh_degree], dtype=np.int64).tobytes()
    state_dict[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=0) as gz:
        gz.write(msgpack.packb(template))
    return buf.getvalue()


def init_stage(up="Z"):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", up)
    stage.SetTimeCodesPerSecond(24.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetMetadata("defaultPrim", "World")
    return stage


def stage_bytes(stage):
    fd, path = tempfile.mkstemp(suffix=".usda")
    os.close(fd)
    stage.GetRootLayer().Export(path)
    data = Path(path).read_bytes()
    os.unlink(path)
    return data


def make_layers(model_filename, positions):
    from pxr import Gf, Sdf, UsdVol
    import numpy as np

    min_coord = positions.min(axis=0).astype(float)
    max_coord = positions.max(axis=0).astype(float)

    gauss_stage = init_stage("Z")
    gauss_stage.SetMetadataByDictKey("customLayerData", "renderSettings", {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    })
    volume = UsdVol.Volume.Define(gauss_stage, "/World/gauss")
    prim = volume.GetPrim()
    volume.AddTransformOp().Set(Gf.Matrix4d(1.0))
    prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    for rel_name, field_name, role, dtype_name in [
        ("density", "density", "density", "float"),
        ("emissiveColor", "emissiveColor", "emissiveColor", "float3"),
    ]:
        field_path = f"/World/gauss/{field_name}_field"
        field = gauss_stage.DefinePrim(field_path, "OmniNuRecFieldAsset")
        volume.CreateFieldRelationship(rel_name, field_path)
        field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set("./" + model_filename)
        field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set(field_name)
        field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set(dtype_name)
        field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set(role)
        if field_name == "emissiveColor":
            field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([1, 0, 0, 0]))
            field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 1, 0, 0]))
            field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 0, 1, 0]))

    prim.GetAttribute("extent").Set([min_coord.tolist(), max_coord.tolist()])
    prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(0, 0, 0))
    prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(*min_coord.tolist()))
    prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(*max_coord.tolist()))
    prim.GetRelationship("proxy") if prim.HasRelationship("proxy") else prim.CreateRelationship("proxy")

    default_stage = init_stage("Z")
    default_stage.OverridePrim("/World/gauss").GetReferences().AddReference("gauss.usda")
    settings = gauss_stage.GetRootLayer().customLayerData.get("renderSettings")
    if settings:
        default_stage.SetMetadataByDictKey("customLayerData", "renderSettings", settings)
    return default_stage, gauss_stage


def main() -> int:
    _check_deps()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ply", help="Input Gaussian PLY file")
    parser.add_argument(
        "output", nargs="?",
        help="Output USDZ path (default: same directory as PLY with .usdz extension)",
    )
    parser.add_argument(
        "--extract-sidecars", action="store_true",
        help="Also write default.usda / gauss.usda alongside the USDZ",
    )
    args = parser.parse_args()

    ply = Path(args.ply).expanduser().resolve()
    if not ply.exists():
        print(f"Error: PLY not found: {ply}", file=sys.stderr)
        return 1

    output_usdz = Path(args.output).expanduser().resolve() if args.output else ply.with_suffix(".usdz")
    output_usdz.parent.mkdir(parents=True, exist_ok=True)
    model_filename = output_usdz.with_suffix(".nurec").name

    print(f"Reading  : {ply}")
    positions, rotations, scales, densities, albedo, specular, sh_degree = read_3dgrut_ply(ply)
    print(f"Gaussians: {positions.shape[0]:,}  SH degree: {sh_degree}")

    print("Building NuRec payload...")
    model_bytes = make_nurec_payload(positions, rotations, scales, densities, albedo, specular, sh_degree)
    default_stage, gauss_stage = make_layers(model_filename, positions)

    with zipfile.ZipFile(output_usdz, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("default.usda", stage_bytes(default_stage))
        z.writestr(model_filename, model_bytes)
        z.writestr("gauss.usda", stage_bytes(gauss_stage))

    if args.extract_sidecars:
        default_bytes = stage_bytes(default_stage)
        gauss_bytes = stage_bytes(gauss_stage)
        (output_usdz.parent / "default.usda").write_bytes(default_bytes)
        (output_usdz.parent / "gauss.usda").write_bytes(gauss_bytes)
        (output_usdz.parent / output_usdz.with_suffix(".usd").name).write_bytes(gauss_bytes)
        (output_usdz.parent / model_filename).write_bytes(model_bytes)
        print(f"Sidecars : {output_usdz.parent}/")

    print(f"Written  : {output_usdz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

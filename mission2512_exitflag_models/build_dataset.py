#!/usr/bin/env python3
"""Build pre-run-only feature sets for mission-2512 TORBEAM exit flags."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "torbeam_training_data_mission2512"
OUT = Path(__file__).resolve().parent
FLAGS = np.array([0, 1, 2, 3, 4, 8, 10, 100], dtype=np.int64)
RHO5 = np.linspace(0.0, 1.0, 5)
RHO21 = np.linspace(0.0, 1.0, 21)
GRID_Z = np.round(np.linspace(0, 64, 9)).astype(int)
GRID_R = np.round(np.linspace(0, 27, 7)).astype(int)


def discover_shots() -> tuple[int, ...]:
    pattern = re.compile(r"tbm_vector_shot_(\d+)\.mat")
    inputs = {int(pattern.fullmatch(path.name).group(1)) for path in DATA.glob("tbm_vector_shot_*.mat")}
    outputs = {
        int(re.fullmatch(r"training_data_shot_(\d+)\.mat", path.name).group(1))
        for path in DATA.glob("training_data_shot_*.mat")
    }
    if inputs != outputs:
        raise ValueError(f"Incomplete pairs: inputs-only={inputs-outputs}, outputs-only={outputs-inputs}")
    return tuple(sorted(inputs))


def scalar(obj: object, name: str, default: float = np.nan) -> float:
    value = getattr(obj, name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def angle_features(pol_deg: float, tor_deg: float) -> tuple[list[float], list[str]]:
    """Encode launcher theta/phi.

    The MAT-file metadata calls columns 0/1 toroidal/poloidal, respectively,
    but the data generator passes them to ``set_angles_robust(newtheta,
    newphi, 'lau')``.  The executable semantics therefore are column 0 =
    launcher theta (poloidal) and column 1 = launcher phi (toroidal).
    """
    values = [pol_deg, tor_deg]
    names = ["pol_angle_theta_deg", "tor_angle_phi_deg"]
    radians = np.deg2rad([pol_deg, tor_deg])
    for harmonic in range(1, 5):
        for angle, angle_name in zip(radians, ("pol_theta", "tor_phi")):
            values.extend([np.sin(harmonic*angle), np.cos(harmonic*angle)])
            names.extend([f"sin_{angle_name}_k{harmonic}", f"cos_{angle_name}_k{harmonic}"])
    return values, names


def frame_features(frame_input: object, density_factor: float, current_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    eq, prof, tbm, launcher = (
        frame_input.eq_data, frame_input.prof_data, frame_input.tbm_data, frame_input.lau_data
    )
    rho = np.asarray(prof.rhopol, dtype=np.float64)
    ne = np.asarray(prof.ne, dtype=np.float64)*density_factor
    te = np.asarray(prof.te, dtype=np.float64)/density_factor
    rhotor = np.asarray(prof.rhotor, dtype=np.float64)

    shared_values = [
        density_factor, current_factor, scalar(frame_input, "time"),
        scalar(eq, "B0"), scalar(eq, "Ip")*current_factor,
        scalar(eq, "Ip"), scalar(eq, "Rmaj")/100.0, scalar(eq, "zA")/100.0,
        scalar(eq, "Rmin"), scalar(prof, "betan"), scalar(eq, "kappa"), scalar(eq, "li"),
        scalar(eq, "volume")*1.0e6, scalar(eq, "psisep"), scalar(eq, "psiaxis"),
        scalar(eq, "psisep")-scalar(eq, "psiaxis"), scalar(eq, "sign"), scalar(prof, "zeff"),
        scalar(tbm, "freq")/1.0e9, scalar(tbm, "nharm"), scalar(tbm, "nmod"), scalar(tbm, "pgyro"),
    ]
    shared_names = [
        "density_factor", "current_factor", "frame_time_s", "B0", "Ip_scaled", "Ip_base", "R0_m", "Z0_m",
        "minor_radius_cm", "betan", "elongation", "li", "volume_cm3", "psisep", "psiaxis",
        "psi_span", "eq_sign", "zeff", "frequency_GHz", "harmonic", "mode", "pgyro",
    ]
    geometry = [
        scalar(tbm, "xxb"), scalar(tbm, "xyb"), scalar(tbm, "xzb"),
        scalar(tbm, "waist_vertical"), scalar(tbm, "waist_horizontal"),
        scalar(tbm, "curvature_radius_vertical"), scalar(tbm, "curvature_radius_horizontal"),
    ]
    geometry_names = [
        "launch_x", "launch_y", "launch_z", "waist_vertical", "waist_horizontal",
        "curvature_vertical", "curvature_horizontal",
    ]
    geometry_missing = float(not np.all(np.isfinite(geometry)))
    shared_values.extend(geometry+[geometry_missing])
    shared_names.extend(geometry_names+["launcher_geometry_missing"])

    def profile_values(radial_grid: np.ndarray) -> tuple[list[float], list[str]]:
        values, names = [], []
        for profile, profile_name in ((ne, "ne"), (te, "Te"), (rhotor, "rhotor")):
            values.extend(np.interp(radial_grid, rho, profile).tolist())
            names.extend([f"{profile_name}_rho_{value:.2f}" for value in radial_grid])
        return values, names

    compact_profile, compact_profile_names = profile_values(RHO5)
    dense_profile, dense_profile_names = profile_values(RHO21)
    compact = np.asarray(shared_values+compact_profile, dtype=np.float64)
    profile_set = np.asarray(shared_values+dense_profile, dtype=np.float64)
    compact_names = shared_names+compact_profile_names
    profile_names = shared_names+dense_profile_names

    psi = np.asarray(eq.psi, dtype=np.float64)
    psi_span = scalar(eq, "psisep")-scalar(eq, "psiaxis")
    psi_normalized = (psi-scalar(eq, "psiaxis"))/psi_span if abs(psi_span) > 1e-15 else psi
    grid_values, grid_names = [], []
    maps = (
        (psi_normalized, "psi_norm"),
        (np.asarray(eq.Br, dtype=np.float64), "Br"),
        (np.asarray(eq.Bz, dtype=np.float64), "Bz"),
        (np.asarray(eq.Bphi, dtype=np.float64), "Bphi"),
    )
    for values, name in maps:
        sampled = values[np.ix_(GRID_Z, GRID_R)].ravel()
        grid_values.extend(sampled.tolist())
        grid_names.extend([f"{name}_z{z}_r{r}" for z in GRID_Z for r in GRID_R])
    rich = np.asarray(profile_set.tolist()+grid_values, dtype=np.float64)
    rich_names = profile_names+grid_names
    return compact, profile_set, rich, compact_names, profile_names, rich_names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shots = discover_shots()
    per_shot: dict[int, dict[str, np.ndarray]] = {}
    names_saved: tuple[list[str], list[str], list[str]] | None = None
    for shot in shots:
        tbm = loadmat(DATA/f"tbm_vector_shot_{shot}.mat", struct_as_record=False, squeeze_me=True)["tbm_vector_struct"]
        outputs = loadmat(DATA/f"training_data_shot_{shot}.mat", struct_as_record=False, squeeze_me=True)["outputs"]
        results = np.asarray(outputs.results, dtype=object)
        sequence = np.asarray(outputs.low_disc_seq, dtype=np.float64)
        variable_names = [str(value) for value in np.asarray(outputs.modified_variables).ravel()]
        expected = ["tor_angle", "pol_angle", "density multiplication factor", "plasma current multiplication factor"]
        if variable_names != expected:
            raise ValueError(f"Shot {shot}: unexpected sequence columns {variable_names}")
        compact_rows, profile_rows, rich_rows = [], [], []
        labels, frames, points = [], [], []
        for frame in range(results.shape[0]):
            for point in range(results.shape[1]):
                pol_angle, tor_angle, density_factor, current_factor = sequence[point]
                angle_values, angle_names = angle_features(pol_angle, tor_angle)
                compact, profile_set, rich, compact_names, profile_names, rich_names = frame_features(
                    tbm[frame].inputs, density_factor, current_factor
                )
                compact_rows.append(angle_values+compact.tolist())
                profile_rows.append(angle_values+profile_set.tolist())
                rich_rows.append(angle_values+rich.tolist())
                labels.append(int(results[frame, point].exitflag))
                frames.append(frame); points.append(point)
                if names_saved is None:
                    names_saved = (
                        angle_names+compact_names, angle_names+profile_names, angle_names+rich_names
                    )
        per_shot[shot] = {
            "compact":np.asarray(compact_rows), "profile":np.asarray(profile_rows),
            "rich":np.asarray(rich_rows), "labels":np.asarray(labels,dtype=np.int64),
            "frames":np.asarray(frames,dtype=np.int64), "points":np.asarray(points,dtype=np.int64),
        }
        print(f"{shot}: {dict(sorted(Counter(labels).items()))}", flush=True)

    if names_saved is None:
        raise RuntimeError("No samples")
    rng=np.random.default_rng(42)
    assignments={}; accum={
        split:{key:[] for key in ("compact","profile","rich","labels","shots","frames","points")}
        for split in ("train","validation","test")
    }
    for shot,data in per_shot.items():
        shuffled=rng.permutation(np.unique(data["frames"]))
        assignment={"train":shuffled[:-2].tolist(),"validation":[int(shuffled[-2])],"test":[int(shuffled[-1])]}
        assignments[str(shot)]=assignment
        for split,selected in assignment.items():
            mask=np.isin(data["frames"],selected)
            for key in ("compact","profile","rich","labels","frames","points"):
                accum[split][key].append(data[key][mask])
            accum[split]["shots"].append(np.full(mask.sum(),shot,dtype=np.int64))
    arrays:dict[str,np.ndarray]={
        "compact_names":np.asarray(names_saved[0]), "profile_names":np.asarray(names_saved[1]),
        "rich_names":np.asarray(names_saved[2]), "class_values":FLAGS,
    }
    for split,values in accum.items():
        for key,chunks in values.items(): arrays[f"{split}_{key}"]=np.concatenate(chunks)
    np.savez_compressed(OUT/"dataset_cache.npz",**arrays)
    report={
        "mission":2512,"shots":list(shots),"raw_samples":33000,
        "split_unit":"complete equilibrium frame within each shot","split_seed":42,
        "sample_counts":{split:int(len(arrays[f"{split}_labels"])) for split in accum},
        "flag_counts":{
            split:{str(flag):int(np.sum(arrays[f"{split}_labels"]==flag)) for flag in FLAGS}
            for split in accum
        },
        "feature_counts":{
            "compact":len(names_saved[0]),"profile":len(names_saved[1]),"rich":len(names_saved[2])
        },
        "feature_names":{"compact":names_saved[0],"profile":names_saved[1],"rich":names_saved[2]},
        "flag_meanings":{
            "0":"normal exit with non-zero absorption",
            "1":"no intersection with the plasma",
            "2":"plasma crossed without absorption",
            "3":"plasma intersected, cutoff",
            "4":"integrator failure",
            "8":"cutoff at the vacuum-plasma boundary",
            "10":"invalid absorption point: wrapper changed a base flag 0 when rho_max == -1",
            "100":"no TORBEAM output file / execution timeout in the MATLAB wrapper",
        },
        "flag_semantics_note":(
            "The supplied TORBEAM guide defines base flags 0--8 and reserves +100/+200 "
            "as allocation/deallocation modifiers. The mission data were produced through "
            "torbeam_output_ech.m, which additionally defines flag 10 and uses 100 when no "
            "rhoresult output file exists. The wrapper meanings apply to this dataset."
        ),
        "frame_assignment":assignments,
        "angle_column_interpretation":(
            "The MAT metadata names columns 0/1 tor_angle/pol_angle, but "
            "generate_training_data.m passes them as newtheta/newphi and its "
            "bounds comments identify them as launcher poloidal theta / "
            "toroidal phi. Features follow the executable generator semantics."
        ),
        "leakage_policy":"Only pre-run inputs are included; all result fields, trajectories, absorption outputs, runtime, and exit-derived quantities are excluded.",
    }
    (OUT/"dataset_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("sample_counts","flag_counts","feature_counts")},indent=2))


if __name__=="__main__":
    main()

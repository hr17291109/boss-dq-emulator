import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import re
from pathlib import Path

COSMO_KEYS = ("Omega_m", "w0", "ln(10^10As)", "ns")

RENAME = {
    "Omega_m": "omega_m",
    "ln(10^10As)": "ln1e10As",
}

LABELS = {
    "omega_m": r"$\Omega_{\rm m}$",
    "w0": r"$w_0$",
    "ln1e10As": r"$\ln(10^{10} A_{\rm s})$",
    "ns": r"$n_{\rm s}$",
    "fsigma8": r"$f\sigma_8$",
}

_COMMENT = re.compile(r"[#;].*$")

def read_param_file(path) -> dict:
    """key = value / key value, comments, non-float values kept as strings."""
    params = {}
    with open(path) as f:
        for raw in f:
            line = _COMMENT.sub("", raw).strip()
            if not line or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
            else:
                fields = line.split()
                if len(fields) < 2:
                    continue
                key, value = fields[0], fields[-1]
            key, value = key.strip(), value.strip()
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


def load_design_table(marge_csv, param_dir="params",
                      cosmo_keys=COSMO_KEYS) -> pd.DataFrame:
    """set_id -> params/Q{i:04d}_input_params.ini を突き合わせ、ln_z 列を作る"""
    table = pd.read_csv(marge_csv)
    if "chi_gauss" not in table.columns:
        raise KeyError(f"'chi_gauss' not in {marge_csv} (found: {list(table.columns)})")

    rows = []
    for set_id in table["set_id"]:
        index = int(str(set_id).lstrip("Qq"))
        path = Path(param_dir) / f"Q{index:04d}_input_params.ini"
        params = read_param_file(path)
        missing = [k for k in cosmo_keys if k not in params]
        if missing:
            raise KeyError(f"{missing} not in {path} (found: {sorted(params)})")
        rows.append({RENAME.get(k, k): float(params[k]) for k in cosmo_keys})

    table = pd.concat([table, pd.DataFrame(rows, index=table.index)], axis=1)
    table["ln_z"] = -0.5 * table["chi_gauss"]
    table["delta_ln_z_rel"] = table["ln_z"] - table["ln_z"].max()
    if "ln1e10As" in table:
        table["As"] = 1e-10 * np.exp(table["ln1e10As"])
    return table


def plot_evidence_1d(table, feature, ax=None, err_column="delta_ln_z"):
    """ln Z - max(ln Z) vs 宇宙論パラメータ"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    yerr = table[err_column] if err_column in table else None
    ax.errorbar(table[feature], table["delta_ln_z_rel"], yerr=yerr,
                fmt="o", ms=5, elinewidth=1, capsize=0)
    ax.set_xlabel(LABELS.get(feature, feature))
    ax.set_ylabel(r"$\Delta \ln \mathcal{Z}(\mathbf{D}|\mathcal{C})$")
    return ax

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logsumexp
import argparse

N_COSMO = 30
CHAIN_TAGS = ["1_", "2_", "3_", "4_"]
COL_CHI2 = 2


def chain_path(i, region, zbin, tag, outdir="output"):
    return Path(outdir) / f"Q{i}" / f"{zbin}Z_{region}_Q{i}_1000_{tag}cov_chi2.dat"

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append([float(p) for p in line.split()[:3]])
            except ValueError:
                continue
    return np.array(rows)

def load_chains(i, region, zbin, outdir="output"):
    chains = {}
    for tag in CHAIN_TAGS:
        path = chain_path(i, region, zbin, tag, outdir)
        if path.exists():
            chains[tag] = load_rows(path)
    return chains

def best_chain_path(i, region, zbin, outdir="output"):
    return Path(outdir) / f"Q{i}" / f"{zbin}Z_{region}_Q{i}_1000_best_cov_chi2.dat"

def load_best_chain(i, region, zbin, outdir="output"):
    path = best_chain_path(i, region, zbin, outdir)
    if not path.exists():
        raise FileNotFoundError(path)
    return load_rows(path)

def chi2_simple_average(chi2):
    chi2 = np.asarray(chi2, dtype=float)
    return -2.0 * (logsumexp(-0.5 * chi2) - np.log(len(chi2)))

# def summarize(i, region, zbin, outdir="output"):
#     chi2 = load_best_chain(i, region, zbin, outdir)[:, COL_CHI2]
#     return {
#         "set_id": f"Q{i}",
#         "chi_min": chi2.min(),
#         "chi_avg": chi2_simple_average(chi2),
#     }

def log_det_2pi_cov(chain):
    cov = np.cov(chain[:, :2].T)
    sign, logdet = np.linalg.slogdet(2 * np.pi * cov)
    if sign <= 0:
        raise ValueError("chain covariance is not positive definite")
    return logdet

def summarize(i, region, zbin, outdir="output"):
    chain = load_best_chain(i, region, zbin, outdir)
    chi2 = chain[:, COL_CHI2]
    chi_avg = chi2_simple_average(chi2)
    logdet = log_det_2pi_cov(chain)
    return {
        "set_id": f"Q{i}",
        "chi_min": chi2.min(),
        "chi_avg": chi_avg,
        "logdet": logdet,
        "chi_gauss": chi_avg - 2.0 * np.log(2.0) - logdet,
    }

def build_table(region, zbin, outdir="output", n_cosmo=N_COSMO):
    return pd.DataFrame([summarize(i, region, zbin, outdir) for i in range(n_cosmo)])



def main():
    ap = argparse.ArgumentParser(description="内側チェーンの χ² を宇宙論ごとにまとめる")
    ap.add_argument("--region", nargs="+", default=["NGC", "SGC"])
    ap.add_argument("--zbin", nargs="+", default=["HIGH", "LOW"])
    ap.add_argument("--outdir", default="output", help="チェーンの置き場所")
    ap.add_argument("--savedir", default="design", help="表の保存先")
    args = ap.parse_args()

    Path(args.savedir).mkdir(exist_ok=True)
    for region in args.region:
        for zbin in args.zbin:
            df = build_table(region, zbin, args.outdir)
            path = Path(args.savedir) / f"ghc_{zbin}Z_{region}_marge.csv"
            df.to_csv(path, index=False)
            diff = df["chi_avg"] - df["chi_min"]
            print(f"{zbin}Z {region} -> {path}   "
                  f"diff: 標準偏差 {diff.std():.2f}   "
                  f"logdet: 平均 {df['logdet'].mean():.2f}, 標準偏差 {df['logdet'].std():.2f}, "
                  f"範囲 {df['logdet'].min():.2f}〜{df['logdet'].max():.2f}")

if __name__ == "__main__":
    main()
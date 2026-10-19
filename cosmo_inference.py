import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from getdist import MCSamples
from tqdm import tqdm

def read_paramlist(filename):
    f=open(filename,'r')
    paramdic = pd.Series()
    for line0 in f:
        columns = line0.split(' ')
        for i in columns:
            paramdic[columns[0]] = float(columns[-1].split('\n')[0])
    f.close()
    return paramdic

def combine_data(ns, highlow):
    combined_data = []
    for i in range(30):
        data = np.loadtxt('output/Q'+ str(i) + '/'+highlow+'Z_'+ns+'_Q' + str(i) + '_1000_cov_chi2.dat', skiprows=1)
        p = read_paramlist(f"params/Q{str(i).zfill(4)}_input_params.ini")
        best_row = data[np.argmin(data[:,2])]

        combined_data.append({
            'set_id': f'Q{i}',
            'omega_b':   p['omega_b'],
            'omega_cdm': p['omega_cdm'],
            'omega_m': p['Omega_m'],
            'w0': p['w0'],
            'As': p['ln(10^10As)'],
            'ns': p['ns'],
            'delv': best_row[0],
            'vmax': best_row[1],
            'chi':  best_row[2],
            'logz': -best_row[2] / 2
        })

    return combined_data

def make_design_region(X_raw, r_max=None):
    mu = X_raw.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(X_raw.T))
    if r_max is None:
        d = X_raw - mu
        r_max = np.sqrt(np.sum((d @ cov_inv) * d, axis=1)).max()
    return mu, cov_inv, r_max

def log_prior(theta, mu, cov_inv, r_max):
    d = np.asarray(theta) - mu
    return 0.0 if d @ cov_inv @ d < r_max**2 else -np.inf

def predict_chi(theta_array, scaler_X, scaler_y, gpr):
    X_s = scaler_X.transform(np.atleast_2d(theta_array))
    y_s, y_std_s = gpr.predict(X_s, return_std=True)
    chi_pred = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    chi_std  = y_std_s * scaler_y.scale_[0]
    return chi_pred, chi_std

def cosomo_mcmc(rng, covmat, current_theta, current_chi2, other_arr, theta_arr, scaler_X, scaler_y, gpr, region, alpha=1.0, prior_only=False):
    proposed_theta = rng.multivariate_normal(current_theta, covmat)

    if not np.isfinite(log_prior(proposed_theta, *region)):
        theta_arr.append(list(current_theta) + [current_chi2])
        other_arr.append(list(proposed_theta) + [np.nan])
        return current_theta, current_chi2
    
    if prior_only:
        proposed_chi2 = 0.0
    else:
        chi_pred_arr, chi2_std = predict_chi(proposed_theta, scaler_X, scaler_y, gpr)
        proposed_chi2 = chi_pred_arr[0] + alpha * chi2_std[0]

    log_r = -(proposed_chi2 - current_chi2) / 2.0
    if np.log(rng.random()) < log_r:
        theta_arr.append(list(proposed_theta) + [proposed_chi2])
        return proposed_theta, proposed_chi2
    else:
        theta_arr.append(list(current_theta) + [current_chi2])
        other_arr.append(list(proposed_theta) + [proposed_chi2])
        return current_theta, current_chi2

def make_kernel(n_dim, kind="rbf", white=True):
    amp = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
    ls = dict(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))

    if kind == "rbf":
        kernel = amp * RBF(**ls)
    elif kind == "matern52":
        kernel = amp * Matern(nu=2.5, **ls)
    else:
        raise ValueError(f"unknown kernel kind: {kind}")

    if white:
        kernel = kernel + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    return kernel

def GPmcmc(df, Nsteps, nburnin, kernel="rbf", step=0.05, seed=12345, r_max=None, prior_only=False, alpha=0.5, white=True, y_col="chi"):
    rng = np.random.default_rng(seed)
    param_names = ["omega_m", "w0", "As", "ns"]
    X_raw = df[param_names].values
    y_raw = df[y_col].values
    #y_raw = df["chi"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    kern = make_kernel(X_scaled.shape[1], kind=kernel, white=white)

    gpr = GaussianProcessRegressor(
        kernel=kern,
        n_restarts_optimizer=20,
        normalize_y=False,
        random_state=42
    )
    gpr.fit(X_scaled, y_scaled)
    print("学習後のカーネル:", gpr.kernel_)
    if white:
        noise = gpr.kernel_.k2.noise_level
        print(f"ノイズ σ (χ² 単位) = {np.sqrt(noise) * scaler_y.scale_[0]:.1f}")

    region = make_design_region(X_raw, r_max=r_max)

    covmat = np.cov(X_raw.T) * step

    best_row = df.loc[df[y_col].idxmin()]
    current_theta = [best_row['omega_m'], best_row['w0'], best_row['As'], best_row['ns']]

    if prior_only:
        current_chi2 = 0.0
    else:
        mu0, sd0 = predict_chi(current_theta, scaler_X, scaler_y, gpr)
        current_chi2 = mu0[0] + alpha * sd0[0]
        print(f"初期 χ²: 実測 {best_row[y_col]:.3f} / GP {current_chi2:.3f}")
    
    theta_arr = []
    th_copy = current_theta.copy()
    th_copy.append(current_chi2)
    theta_arr.append(th_copy)
    other_arr = []

    for i in tqdm(range(Nsteps)):
        current_theta, current_chi2 = cosomo_mcmc(rng, covmat, current_theta, current_chi2, other_arr, theta_arr, scaler_X, scaler_y, gpr, region, alpha=alpha, prior_only=prior_only)

    theta_arr = np.array(theta_arr)

    labels  = [r"\Omega_m", r"w_0", r"A_s", r"n_s"]
    names   = ["omega_m",   "w0",  "As",  "ns"]

    mc_samples = MCSamples(
        samples=theta_arr[nburnin:,:4],
        names=names,
        labels=labels,
        # settings={'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3},
        # label="Posterior"
    )

    total_steps = len(theta_arr) - 1 
    rejected_steps = len(other_arr)
    accepted_steps = total_steps - rejected_steps
    acceptance_rate = accepted_steps / total_steps

    print(f"総ステップ数: {total_steps}")
    print(f"採択回数: {accepted_steps}")
    print(f"棄却回数: {rejected_steps}")
    print(f"平均採択率: {acceptance_rate:.4f} ({acceptance_rate * 100:.1f}%)")

    d = theta_arr[nburnin:, :4] - region[0]
    dM = np.sqrt(np.sum((d @ region[1]) * d, axis=1))
    print(f"r_max の 95% より外側にあるサンプルの割合: {(dM > 0.95 * region[2]).mean():.3f}")
    return mc_samples    
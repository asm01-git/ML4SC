import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct, Matern
from sklearn.preprocessing import StandardScaler
import time
import os

# Paths are relative to this script's directory, regardless of where it is run from
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(_HERE, 'gym-unbalanced-disk', 'disc-benchmark-files') + os.sep

# ── 1. Load & Visualise Data ───────────────────────────────────────────────────
out     = np.load(DATA + 'training-val-test-data.npz')
th_data = out['th']
u_data  = out['u']
N       = len(th_data)
split_idx = int(0.7 * N)

# fig, ax = plt.subplots(figsize=(7, 5))
# ax.scatter(u_data, th_data, s=1, alpha=0.3, color='steelblue')
# ax.set_xlabel('Input u (V)')
# ax.set_ylabel('Angle θ (rad)')
# ax.set_title(f'Input–Output Scatter — {N} samples')
# ax.grid(alpha=0.3)
# plt.tight_layout()

u_train, th_train = u_data[:split_idx], th_data[:split_idx]
u_val,   th_val   = u_data[split_idx:], th_data[split_idx:]

# ── 2. NARX Feature Construction ──────────────────────────────────────────────
na, nb = 5, 4

def create_IO_data(u, y, na, nb):
    X, Y = [], []
    for k in range(max(na, nb), len(y)):
        X.append(np.concatenate([u[k-nb:k], y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

X_train, Y_train = create_IO_data(u_train, th_train, na, nb)
X_val,   Y_val   = create_IO_data(u_val,   th_val,   na, nb)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples | "
      f"Feature dim: {X_train.shape[1]} (nb={nb} past u + na={na} past θ)")

# ── 3. Metric & Plot Helpers (match submission-file-checker.py) ────────────────
def checker_metrics(th_actual, th_pred, label):
    """Compute RMS (rad), RMS (deg), NRMS — same formula as submission-file-checker.py."""
    residual = th_pred - th_actual
    rms      = float((residual ** 2).mean() ** 0.5)
    rms_deg  = rms / (2 * np.pi) * 360
    nrms     = rms / float(th_actual.std())
    print(f'  [{label}]  RMS = {rms:.4f} rad  |  {rms_deg:.3f} deg  |  NRMS = {nrms:.2%}')
    return rms, rms_deg, nrms


def plot_pred(actual, predicted, title, n_plot=600):
    """Two-row figure: predicted vs measured on top, residual below."""
    n = min(n_plot, len(actual), len(predicted))
    residual = predicted[:n] - actual[:n]
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(actual[:n],    'b',   lw=1.0, label='Measured θ')
    axes[0].plot(predicted[:n], 'r--', lw=1.0, label='Predicted θ')
    axes[0].set_ylabel('θ (rad)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(title)
    axes[1].plot(residual, color='gray', lw=0.8, label='Residual (pred − meas)')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('Residual (rad)'); axes[1].set_xlabel('Sample k')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()


# ── 4. Free-Run Simulation Helper ─────────────────────────────────────────────
def simulation_IO_model(model, ulist, ylist, skip=50):
    upast = ulist[skip - nb:skip].tolist()
    ypast = ylist[skip - na:skip].tolist()
    Y     = ylist[:skip].tolist()
    for u in ulist[skip:]:
        x      = np.concatenate([upast, ypast], axis=0)
        x_sc   = scaler.transform(x[None, :])
        ypred  = model.predict(x_sc)[0]
        Y.append(ypred)
        upast.append(u);     upast.pop(0)
        ypast.append(ypred); ypast.pop(0)
    return np.array(Y)


# ── 5. Full Validation Evaluator ───────────────────────────────────────────────
def evaluate_on_val(model, name):
    print(f'\n=== {name} — Validation Metrics ===')

    # One-step-ahead (NARX uses actual past measurements as features)
    preds_osa = model.predict(X_val)
    rms_o, rms_o_d, nrms_o = checker_metrics(Y_val, preds_osa, 'One-Step-Ahead')

    # Free-run simulation (recursive — predictions fed back)
    print(f'  Running {name} simulation...')
    th_sim = simulation_IO_model(model, u_val, th_val, skip=50)
    rms_s, rms_s_d, nrms_s = checker_metrics(th_val[50:], th_sim[50:], 'Simulation    ')

    plot_pred(
        Y_val, preds_osa,
        f'{name} — One-Step-Ahead (Validation)\n'
        f'RMS = {rms_o:.4f} rad  |  {rms_o_d:.2f}°  |  NRMS = {nrms_o:.2%}')

    plot_pred(
        th_val, th_sim,
        f'{name} — Free-Run Simulation (Validation)\n'
        f'RMS = {rms_s:.4f} rad  |  {rms_s_d:.2f}°  |  NRMS = {nrms_s:.2%}')

    return nrms_o, nrms_s


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Gaussian Process (GP) — NARX
# ─────────────────────────────────────────────────────────────────────────────
print("--- Training Gaussian Process ---")
# GP is O(n³) in training — subsample 2000 points
np.random.seed(42)
idx = np.random.choice(len(X_train), 2000, replace=False)
X_train_gp, Y_train_gp = X_train[idx], Y_train[idx]

# ── Kernel hyperparameters (initial values → optimizer tunes these) ────────────
# DotProduct(sigma_0)     : bias in linear term; large value → linear/ARX-like fit dominates
# ConstantKernel          : amplitude of RBF; small value → nonlinear part suppressed
# RBF(length_scale)       : locality of nonlinear fit; small=wiggly/local, large=smooth/global
# WhiteKernel(noise_level): assumed obs. noise; small=interpolate tightly, large=smooth over noise
#
# After optimisation the kernel printed was:
#   DotProduct(sigma_0=3.5e+04) + 0.00966**2 * RBF(length_scale=2.52) + WhiteKernel(noise_level=3.85e-05)
# → sigma_0 exploded (linear term dominates), RBF amplitude collapsed to ~9e-5 (nonlinear part off),
#   noise near zero — GP converged to an ARX-like linear model.
# kernel = (DotProduct()
#           + ConstantKernel(1.0) * RBF(length_scale=1.0)
#           + WhiteKernel(noise_level=0.1))
d = X_train_gp.shape[1]
kernel = (ConstantKernel(1.0, (1e-2, 1e2))
          * Matern(length_scale=np.sqrt(d) * np.ones(d),   # ARD, sensible init
                   length_scale_bounds=(1e-1, 1e2), nu=2.5)
          + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e0)))
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, normalize_y=True)
 
t0 = time.time()
gp_model.fit(X_train_gp, Y_train_gp)
print(f"GP training took {time.time() - t0:.2f} s.  Optimised kernel: {gp_model.kernel_}")

gp_nrms_osa, gp_nrms_sim = evaluate_on_val(gp_model, 'GP')

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE TEST SUBMISSIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generating Test Submission Files ---")

def create_submissions(model, name_prefix):
    # Prediction task
    pred_data   = np.load(DATA + 'hidden-test-prediction-submission-file.npz')
    upast_test  = pred_data['upast']
    thpast_test = pred_data['thpast']
    X_test_pred = np.concatenate([upast_test[:, 15 - nb:],
                                  thpast_test[:, 15 - na:]], axis=1)
    X_test_pred = scaler.transform(X_test_pred)
    Y_predict   = model.predict(X_test_pred)
    np.savez(DATA + f'hidden-test-prediction-{name_prefix}-submission.npz',
             upast=upast_test, thpast=thpast_test, thnow=Y_predict)
    print(f"  Saved prediction submission for '{name_prefix}'")

    # Simulation task
    sim_data    = np.load(DATA + 'hidden-test-simulation-submission-file.npz')
    u_test_sim  = sim_data['u']
    th_test_sim = sim_data['th']
    Y_sim       = simulation_IO_model(model, u_test_sim, th_test_sim, skip=50)
    np.savez(DATA + f'hidden-test-simulation-{name_prefix}-submission.npz',
             th=Y_sim, u=u_test_sim)
    print(f"  Saved simulation submission for '{name_prefix}'")


create_submissions(gp_model,  'gp')

print("\nDone. Verify format with:")
print(f"  python {DATA}submission-file-checker.py "
      f"{DATA}hidden-test-simulation-gp-submission.npz "
      f"{DATA}hidden-test-simulation-submission-file.npz")

plt.show()

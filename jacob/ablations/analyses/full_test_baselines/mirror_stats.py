"""Check the paper's claim that mirrored head movements are as likely as the originals (13 Nov 2026).

For every training sample it computes the model's own input features (body rate w = 2 f_s a v from the delta
quaternion, and gravity g = R(q)^T e_y) and accumulates their means and second moments. Under the mirror M the
components w_y, w_z and g_x change sign, so for mirror-symmetric data their means are zero, they are positive half the
time, and every correlation between a sign-changing and a sign-keeping component is zero.
"""
import multiprocessing
import sys

import h5py
import numpy as np

H5 = sys.argv[1]
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "train"
NAMES = ["w_x", "w_y", "w_z", "g_x", "g_y", "g_z"]
ODD = [False, True, True, True, False, False]  # changes sign under the mirror


def features(q):
    w0, x0, y0, z0 = q[:-1].T
    w1, x1, y1, z1 = q[1:].T
    # d = conj(q[t-1]) * q[t] = (a, v)
    a = w0*w1 + x0*x1 + y0*y1 + z0*z1
    v = np.stack((w0*x1 - x0*w1 - y0*z1 + z0*y1, w0*y1 + x0*z1 - y0*w1 - z0*x1, w0*z1 - x0*y1 + y0*x1 - z0*w1), 1)
    rate = np.rad2deg(2 * 100.0 * a[:, None] * v)  # degrees per second
    w, x, y, z = q[1:].T
    gravity = np.stack((2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)), 1)
    return np.concatenate((rate, gravity), 1)


def one(path):
    with h5py.File(H5, "r") as h5:
        q = h5[path]["quaternion_wxyz"][...].astype(np.float64)
    f = features(q / np.linalg.norm(q, axis=1, keepdims=True))
    return len(f), f.sum(0), f.T @ f, (f > 0).sum(0), (f < 0).sum(0), np.abs(f).sum(0), \
        np.where(f > 0, f, 0).sum(0), np.where(f < 0, -f, 0).sum(0)


if __name__ == "__main__":
    with h5py.File(H5, "r") as h5:
        paths = [f"{SPLIT}/{u}/{r}/{s}" for u in h5[SPLIT] for r in h5[SPLIT][u] for s in h5[SPLIT][u][r]]
    n, s, ss, pos, neg, ab, psum, nsum = 0, 0, 0, 0, 0, 0, 0, 0
    with multiprocessing.Pool(4) as pool:
        for r in pool.imap_unordered(one, paths, chunksize=16):
            n += r[0]; s = s + r[1]; ss = ss + r[2]; pos = pos + r[3]; neg = neg + r[4]; ab = ab + r[5]
            psum = psum + r[6]; nsum = nsum + r[7]
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    print(f"{SPLIT}: {len(paths):,} segments, {n:,} samples")
    for i, name in enumerate(NAMES):
        print(f"{name}: mean {mean[i]:+.5f}  sd {sd[i]:.4f}  mean/sd {mean[i]/sd[i]:+.5f}  positive share "
              f"{pos[i]/(pos[i]+neg[i]):.5f}  mean size when positive {psum[i]/pos[i]:.4f}, negative {nsum[i]/neg[i]:.4f}  "
              f"{'(sign changes under mirror)' if ODD[i] else ''}")
    print("correlations between a sign-changing and a sign-keeping component (zero under mirror symmetry):")
    for i in range(6):
        for j in range(6):
            if ODD[i] and not ODD[j]:
                print(f"  {NAMES[i]} x {NAMES[j]}: {corr[i, j]:+.4f}")
    print("for scale, correlations within the same class:")
    for i in range(6):
        for j in range(i + 1, 6):
            if ODD[i] == ODD[j]:
                print(f"  {NAMES[i]} x {NAMES[j]}: {corr[i, j]:+.4f}")

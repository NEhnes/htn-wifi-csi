"""
Feasibility test for 'Fault-Injected Reflexes': does timing jitter in the
assumed dt actually make a Kalman filter's REPORTED covariance dishonest
(overconfident)? Metric = NEES (Normalized Estimation Error Squared).
A consistent filter has E[NEES] = state_dim (=2 here). NEES far above the
chi-square upper bound => the filter's covariance is lying (overconfident).

We compare: filter told the TRUE dt each step (FPGA-timestamped) vs filter
told a NOMINAL dt while the real elapsed time was jittered (software clock).
"""
import numpy as np
rng = np.random.default_rng(0)

def run(nominal_dt, jitter_frac, n_steps=400, n_mc=300, q=5.0, r=0.04):
    # constant-velocity 1D: state [pos, vel]
    dim = 2
    chi2_lo, chi2_hi = 1.0, 3.0  # rough 95% bounds for 2-DoF NEES averaged over MC
    nees_true = []
    nees_jit = []
    for _ in range(n_mc):
        x = np.array([0.0, 1.0])            # true state
        Pt = np.eye(2)*0.1; xt = x.copy()   # filter A (true dt)
        Pj = np.eye(2)*0.1; xj = x.copy()   # filter B (nominal dt, real jittered)
        acc = []
        for k in range(n_steps):
            real_dt = nominal_dt*(1.0 + jitter_frac*rng.standard_normal())
            real_dt = max(real_dt, 1e-4)
            # --- true dynamics evolve over REAL dt ---
            F_real = np.array([[1, real_dt],[0,1]])
            w = rng.multivariate_normal([0,0],
                 q*np.array([[real_dt**3/3, real_dt**2/2],[real_dt**2/2, real_dt]]))
            x = F_real@x + w
            z = x[0] + rng.normal(0, np.sqrt(r))
            H = np.array([[1.0,0.0]])
            # --- Filter A: knows real_dt ---
            F=F_real; Q=q*np.array([[real_dt**3/3,real_dt**2/2],[real_dt**2/2,real_dt]])
            xt=F@xt; Pt=F@Pt@F.T+Q
            S=H@Pt@H.T+r; K=Pt@H.T/S; xt=xt+(K.flatten()*(z-H@xt)); Pt=Pt-K@H@Pt
            # --- Filter B: assumes nominal_dt (wrong) ---
            dtn=nominal_dt
            Fn=np.array([[1,dtn],[0,1]]); Qn=q*np.array([[dtn**3/3,dtn**2/2],[dtn**2/2,dtn]])
            xj=Fn@xj; Pj=Fn@Pj@Fn.T+Qn
            S=H@Pj@H.T+r; K=Pj@H.T/S; xj=xj+(K.flatten()*(z-H@xj)); Pj=Pj-K@H@Pj
            if k>50:
                et=x-xt; nees_true.append(et@np.linalg.solve(Pt,et))
                ej=x-xj; nees_jit.append(ej@np.linalg.solve(Pj,ej))
    return np.mean(nees_true), np.mean(nees_jit)

print("state dim = 2  => a CONSISTENT filter has mean-NEES ~ 2.0")
print("mean-NEES > ~2.4 means the reported covariance is OVERCONFIDENT (dishonest)\n")
print(f"{'jitter in dt':>16} | {'NEES true-dt':>12} | {'NEES nominal-dt':>15} | verdict")
for jf, label in [(0.001,'0.1% (QNX µs)'),(0.01,'1%'),(0.05,'5%'),(0.10,'10% (loaded SW)'),(0.20,'20% (heavy load)')]:
    nt, njj = run(0.01, jf)   # nominal_dt = 10 ms (100 Hz control loop)
    verdict = "covariance LIES" if njj>2.4 else "ok"
    print(f"{label:>16} | {nt:12.2f} | {njj:15.2f} | {verdict}")

"""
The mechanism that actually matters for 'Fault-Injected Reflexes':
when the AI/perception process CRASHES, measurements stop. The estimate must
keep inflating its covariance so the DISPLAYED uncertainty grows honestly.

Case A (FPGA/independent): the inflation loop runs on an independent clock and
        keeps growing P every tick -> displayed uncertainty tracks true error.
Case B (software, same stalled context): the crash also stalls the code that
        was supposed to inflate P, so the DISPLAYED P freezes at its last value
        while the true error keeps growing -> overconfident (covariance lies).
Metric: NEES using the DISPLAYED covariance during a 2s dropout.
"""
import numpy as np
rng=np.random.default_rng(1)
q=5.0; r=0.04; dt=0.01
def cov_step(P):
    F=np.array([[1,dt],[0,1]]); Q=q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
    return F@P@F.T+Q
n_mc=500; drop_steps=200  # 2 s dropout at 100 Hz
neesA=[]; neesB=[]
for _ in range(n_mc):
    x=np.array([0.0,1.0]); xh=x.copy()
    P=np.eye(2)*0.1
    # settle
    for _ in range(100):
        F=np.array([[1,dt],[0,1]]); Q=q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
        w=rng.multivariate_normal([0,0],Q); x=F@x+w
        z=x[0]+rng.normal(0,np.sqrt(r)); H=np.array([[1.0,0.0]])
        xh=F@xh; P=F@P@P.T*0+cov_step(P)
        S=H@P@H.T+r; K=P@H.T/S; xh=xh+(K.flatten()*(z-H@xh)); P=P-K@H@P
    P_frozen=P.copy(); P_live=P.copy(); xh_live=xh.copy()
    # DROPOUT: no measurements. true state evolves; estimate only predicts.
    for k in range(drop_steps):
        F=np.array([[1,dt],[0,1]]); Q=q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
        w=rng.multivariate_normal([0,0],Q); x=F@x+w
        xh_live=F@xh_live                 # predicted mean (both cases share it)
        P_live=cov_step(P_live)           # Case A: keeps inflating
        # Case B: P_frozen stays fixed (stalled inflation)
        if k>5:
            e=x-xh_live
            neesA.append(e@np.linalg.solve(P_live,e))
            neesB.append(e@np.linalg.solve(P_frozen,e))
print("During a 2s perception-process dropout (state dim 2, consistent~2.0):\n")
print(f"  Case A  FPGA/independent inflation : mean-NEES = {np.mean(neesA):6.2f}  (honest, ~bounded)")
print(f"  Case B  stalled software inflation : mean-NEES = {np.mean(neesB):6.2f}  (OVERCONFIDENT)")
print(f"\n  => Case B is {np.mean(neesB)/np.mean(neesA):.0f}x more overconfident: the displayed")
print("     covariance lies because it froze with the crashed process.")
print("\n  CONCLUSION: the FPGA's load-bearing role is GUARANTEED INDEPENDENT EXECUTION of")
print("  the inflation/safety loop when the AI dies - NOT clock-precision. Pitch it that way.")

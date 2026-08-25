"""Single source of truth for the integration dataflow (prediction -> decision).

The STGCN/STGAT predictions are produced by the model repos' run_infer.py, which
write the .npy files into THIS folder; the decision stage reads them locally, so
the pipeline has no sideways dependency on any other folder.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../integration
ROOT = HERE.parent                                # project root

# --- inputs (all local to this folder -> integration/ is fully standalone) ---
STGCN_PRED = HERE / "stgcn_pred.npy"             # written by ../STGCN/run_infer.py
STGAT_PRED = HERE / "stgat_pred.npy"             # written by ../STGAT/run_infer.py
# vendored copy of ../STGAT/data/METR-LA/adj_mx_dijsk.pkl (static road network;
# re-copy only if the network topology changes).
ADJ_PKL = HERE / "data" / "adj_mx_dijsk.pkl"

# --- outputs ---
# One sub-directory per road network. The agents are NOT interchangeable in practice
# even though the E-GAT weights carry no node-count dependency: METR-LA averages 1.5
# hops with relative edge times, Taichung 41 hops in seconds, and each has its own
# capacity_scale, so a checkpoint scored on one is meaningless on the other. A single
# shared default filename silently overwrote the other network's agent.
CKPT_DIR = HERE / "checkpoints"

# --- speed / ensemble ---
SPEED_MIN, SPEED_MAX = 1.0, 70.0      # mph clamp
W_STGCN, W_STGAT = 0.2, 0.8           # ensemble weights
ADJ_THRESHOLD = 0.0                   # keep directed edges with kernel weight > this
KNN = 0                               # k-NN sparsification: keep each node's k nearest out-neighbors (0 = dense; k>0 reduces spreading headroom)

# --- volume-delay (BPR) congestion model: t(load) = t0 * (1 + A*(load/cap)^B) ---
#     The standard Bureau-of-Public-Roads link cost. It is what makes the
#     "herding effect" mechanical: piling vehicles on one link inflates its time.
BPR_A, BPR_B = 0.15, 4.0
EDGE_CAPACITY = 18.0                  # vehicles per link before saturation (uniform proxy)

# --- global-penalty reward weights (proposal eq. 4) ---
#     R = -alpha*travel_time - lambda1*sum max(0, rho - rho_th)^2 - lambda2*Var(rho)
ALPHA = 1.0
LAMBDA_SAT = 0.5                      # lambda1: per-link saturation-overflow penalty
LAMBDA_VAR = 0.8                      # lambda2: load-spread (variance) penalty (raised 0.3->0.8 to push Gini down)
RHO_THRESHOLD = 0.85                  # saturation threshold (proposal uses 0.85)
PENALTY_SCALE = 12.0                  # puts the penalty terms on the same scale as edge time

# --- Taichung (real OSM road network, Map/graph_*_taichung.csv) ---
#     The CSV capacity is in veh/h (~1360-8000). At a few hundred/thousand vehicles
#     every edge sits near rho~0, so nothing congests and there is no herding to
#     suppress: scale it down into the abstract load range the BPR / eq.4 reward was
#     tuned on. `calibrate_taichung.py` reports the scale that puts the herding
#     baseline at a target worst-rho.
#     0.04 was calibrated on the FULL 20,390-edge export (~= worst-rho 3.0 at 800
#     vehicles) and does not carry over to the arena: the same vehicles now share
#     2,342 edges holding 5.5x less total capacity, so rho rises accordingly. 0.22 is
#     that ratio applied to 0.04 -- an ESTIMATE to keep run_compare readable, not a
#     calibration. Run `python calibrate_taichung.py` and replace it with the measured
#     value before reporting anything.
TAICHUNG_CAPACITY_SCALE = 0.0429
#     City routes run 30-70 hops (METR-LA's dense kernel needed only ~1.5), so the
#     60-hop default would abandon most trips before they reach their destination.
TAICHUNG_MAX_HOPS = 200
#     ~94% of the CSV rows have a blank free_flow_speed_kmh. Capture_Road_Node.py is
#     documented to write 50 km/h (the common Taichung urban limit) when OSM has no
#     maxspeed tag, but the current export leaves them empty — so the fallback here is
#     50 to match that intent and keep t0 consistent across the pipeline.
TAICHUNG_DEFAULT_SPEED_KMH = 50.0
#     Clamp on the models' raw output, in km/h (SPEED_MIN/MAX above are mph for
#     METR-LA). Routing cost is length/speed, so a near-zero or negative prediction
#     would blow the cost up or make it negative; 120 is already above any urban road.
TAICHUNG_SPEED_MIN_KMH, TAICHUNG_SPEED_MAX_KMH = 1.0, 120.0
#     What an edge with NO prediction is assumed to be doing. Only 14.8% of the arena
#     carries a TDX signal, so this choice decides how the other 85% compares.
#       "network_mean" — scale its free-flow time by the mean slowdown measured on the
#                        observed edges (0.49x here, i.e. ~25 km/h). Both groups then
#                        sit on one scale.
#       "free_flow"    — leave tpred = t0. Assumes every road without a sensor runs at
#                        the 50 km/h limit while measured roads report ~25, so the
#                        prediction-following baselines (2)(3)(4) route away from the
#                        instrumented arterials and came out 18% SLOWER than (1).
#     Keep both: "free_flow" is the sensitivity check that shows the choice matters.
TAICHUNG_TPRED_FALLBACK = "network_mean"

# --- experiment / demand ---
N_VEHICLES = 300
N_HOTSPOTS = 4                        # number of "city-center" sink nodes for the hotspot scenario
N_BATCHES = 30                        # incremental-assignment granularity (more = closer to system-optimal)
SCENARIO = "hotspot"                  # "random" | "hotspot" (hotspot funnels demand -> triggers herding)
SEED = 42

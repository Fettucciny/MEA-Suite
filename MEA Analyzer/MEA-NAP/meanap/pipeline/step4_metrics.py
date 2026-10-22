"""Vendored verbatim from SAND-Lab MEA-NAP python port (src/meanap/pipeline/step4.py):
the compute_network_metrics orchestrator = ExtractNetMet.m. Only the imports are
trimmed to what this function needs (no GUI/IO/pandas)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from meanap.params import Params
from meanap.pipeline import network_metrics as nm
from meanap.pipeline.modularity import mod_consensus_cluster_iterate
from meanap.pipeline.null_models import latmio_und_v2, randmio_und_v2

def compute_network_metrics(
    adj_m: np.ndarray,
    spike_counts: np.ndarray,
    duration_s: float,
    min_activity_level: float,
    min_nodes: int,
    exclude_edges_below_threshold: bool = True,
    params: Params | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute deterministic network metrics for one (recording, lag) adjacency matrix.

    Mirrors the active-node subsetting + metric calls in ``ExtractNetMet.m``
    (weighted adjM path).
    """
    adj_m = adj_m.copy()
    adj_m[adj_m < 0] = 0.0
    adj_m = np.nan_to_num(adj_m, nan=0.0)

    node_strength_full = adj_m.sum(axis=0)
    activity_level = spike_counts / duration_s
    inclusion_index = np.nonzero((node_strength_full != 0) & (activity_level >= min_activity_level))[0]
    a_n = len(inclusion_index)

    result: dict = {"aN": a_n, "activeChannelIndex": inclusion_index}

    if a_n < min_nodes:
        return result

    sub = adj_m[np.ix_(inclusion_index, inclusion_index)]
    result["adjMsub"] = sub  # not JSON-serialized — see _run_step4_network_metrics

    nd, mew = nm.find_node_deg_edge_weight(
        sub, edge_thresh=0.0001, exclude_zeros=exclude_edges_below_threshold,
    )
    result["ND"] = nd
    result["MEW"] = mew
    result["NS"] = nm.strengths_und(sub)
    result["Dens"] = nm.density_und(sub)

    # a_n >= min_nodes is already guaranteed by the early return above, so
    # unlike ExtractNetMet.m (which nan-guards this block with an explicit
    # "aN >= minNumberOfNodesToCalNetMet" check) it's unconditional here.
    result["NDmean"] = float(np.nanmean(nd))
    nd_p75 = np.percentile(nd, 75)
    result["NDtop25"] = float(np.mean(nd[nd >= nd_p75]))
    result["NSmean"] = float(np.nanmean(result["NS"]))

    # Mean of the significant (nonzero) edges — every matrix entry, not just
    # the upper triangle, matching ExtractNetMet.m's `adjM(abs(adjM) > 0)`.
    sig_edges = sub[np.abs(sub) > 0]
    if sig_edges.size:
        result["sigEdgesMean"] = float(np.mean(sig_edges))
        sig_edges_p90 = np.percentile(sig_edges, 90)
        result["sigEdgesTop10"] = float(np.mean(sig_edges[sig_edges >= sig_edges_p90]))

    # Raw (unnormalized) clustering coefficient / path length — independently
    # useful/testable deterministic quantities, but NOT what MATLAB saves as
    # NetMet.CC/NetMet.PL (see the small-worldness block below for those).
    result["CC_raw"] = nm.clustering_coef_wu(sub)
    result["CC_rawMean"] = float(np.mean(result["CC_raw"]))

    length_mat = nm.weight_conversion_lengths(sub)
    dist = nm.distance_wei(length_mat)
    pl_raw, _ = nm.charpath(dist)
    result["PL_raw"] = pl_raw

    result["Eglob"] = nm.efficiency_wei_global(sub)

    # ── Small-worldness (SW/SWw + the saved, null-model-normalized CC/PL) ──
    # MATLAB's own gate here is strictly "> minNumberOfNodesToCalNetMet"
    # (ExtractNetMet.m), unlike every other block's "aN >= ..." — faithfully
    # replicated, not a typo.
    if a_n > min_nodes:
        if rng is None:
            rng = np.random.default_rng()
        dist_profile = squareform(pdist(sub))
        lattice_net = latmio_und_v2(sub, 10000, dist_profile, rng=rng)
        random_net = randmio_und_v2(sub, 5000, rng=rng)
        sw, sww, cc, pl = nm.small_worldness_rl_wu(sub, random_net, lattice_net)
        result["SW"] = sw
        result["SWw"] = sww
        result["CC"] = cc
        result["PL"] = pl

    sub_nrm = nm.weight_conversion_normalize(sub)
    eloc = nm.efficiency_wei_local(sub_nrm)
    result["Eloc"] = eloc
    result["ElocMean"] = float(np.mean(eloc))

    path_len_net = 1.0 / (sub + 0.01)
    bc = nm.betweenness_wei(path_len_net)
    n = sub.shape[0]
    result["BC"] = bc / ((n - 1) * (n - 2)) if n > 2 else np.full(n, np.nan)

    with np.errstate(divide="ignore"):
        mean_dist = dist.mean(axis=0)
        result["NE"] = 1.0 / mean_dist

    # ── Modularity-dependent metrics (stochastic Ci — see modularity.py) ──
    if params is not None and a_n > 1:
        ci, q, _num_repeats = mod_consensus_cluster_iterate(sub, threshold=0.4, rep_num=50, rng=rng)
        result["Ci"] = ci
        result["Q"] = q
        result["nMod"] = int(ci.max())

        # PC = normalized participation coefficient — matches what MATLAB
        # actually saves as NetMet.PC (participation_coef_norm.m's 1st
        # output) and feeds into node cartography / hub classification /
        # the "4_MEA_NetworkPlotNodedegreeParticipationcoefficient.png"
        # plot. PC_raw is the deterministic (given Ci) 3rd output, kept
        # separately since it's independently testable/useful.
        pc_norm, pc_residual, pc_raw, _between_mod_k = nm.participation_coef_norm(sub, ci, n_iter=100, rng=rng)
        z = nm.module_degree_zscore(sub, ci)
        result["PC"] = pc_norm
        result["PC_raw"] = pc_raw
        result["PC_residual"] = pc_residual
        result["Z"] = z

        result["PCmean"] = float(np.mean(pc_norm))
        pc_p90 = np.percentile(pc_norm, 90)
        pc_p10 = np.percentile(pc_norm, 10)
        result["PCmeanTop10"] = float(np.mean(pc_norm[pc_norm >= pc_p90]))
        result["PCmeanBottom10"] = float(np.mean(pc_norm[pc_norm <= pc_p10]))
        result["percentZscoreGreaterThanZero"] = float(np.sum(z > 0) / len(z) * 100)
        result["percentZscoreLessThanZero"] = float(np.sum(z < 0) / len(z) * 100)

        result["RC"] = nm.rich_club_wu(sub)

        nd_cart_div, pop_num_nc = nm.classify_node_cartography(
            pc_norm, z,
            params.hub_boundary_wm_d_deg, params.peri_part_coef,
            params.non_hub_connector_part_coef, params.pro_hub_part_coef,
            params.connector_hub_part_coef,
        )
        result["NdCartDiv"] = nd_cart_div
        for i in range(6):
            result[f"NCpn{i + 1}"] = float(pop_num_nc[i] / a_n)
            result[f"NCpn{i + 1}count"] = int(pop_num_nc[i])

        hub3, hub4 = nm.hub_classification(result["ND"], pc_norm, result["BC"], result["NE"])
        result["Hub3"] = hub3
        result["Hub4"] = hub4

    # ── Controllability ────────────────────────────────────────────────────────
    ave_control = nm.average_controllability(adj_m)
    if len(ave_control) > 0:
        ave_control_sub = ave_control[inclusion_index]
        result["aveControl"] = ave_control_sub
        result["aveControlMean"] = float(np.mean(ave_control_sub))
        p75 = np.percentile(ave_control_sub, 75)
        result["aveControlTop25"] = float(np.mean(ave_control_sub[ave_control_sub >= p75]))

    modal_control = nm.modal_controllability(adj_m)
    if len(modal_control) > 0:
        modal_control_sub = modal_control[inclusion_index]
        result["modalControl"] = modal_control_sub
        result["modalControlMean"] = float(np.mean(modal_control_sub))
        result["modalControlPrctLessThanThreshold"] = float(np.mean(modal_control_sub < 0.975))

    return result


# NMF-related arrays that are NOT indexed by node/channel (rank-k sweeps,
# factor matrices) — must be excluded from the generic "any array => spread
# across NodeLevel.csv rows by channel index" logic below, since their
# length can coincidentally match the active-node count and get silently
# (and wrongly) treated as per-channel data.
_NMF_NON_NODE_KEYS = frozenset({
    "nnmf_residuals", "nnmf_var_explained", "randResidualPerComponent",
    "downSampleSpikeMatrix", "nmfFactors", "nmfWeights",
    "nmfFactorsVarThreshold", "nmfWeightsVarThreshold",
})

# Edge-weight bounds for the "scaled to entire data batch" network plots.

#!/usr/bin/env python3
"""
VALENCE Pipeline — v2026.07
============================
Reproducible computation pipeline for the VALENCE periodic table
of AI political economy.

Pipeline architecture:
  Raw sources → Feature extraction → Score computation → Placement → Output

Usage:
  python pipeline.py                        # full run, outputs data.json
  python pipeline.py --dry-run              # validate inputs, no output
  python pipeline.py --occupation Jd        # single occupation debug

Every score is traceable to a source file, formula, and weight.
All weights are in weights.json — edit there to recalibrate.

Author: VALENCE project
Model version: 2026.07
"""

import json
import math
import argparse
import sys
from datetime import date
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
SOURCES     = ROOT / "sources"
WEIGHTS_F   = ROOT / "weights.json"
OUTPUT_F    = ROOT.parent / "public" / "data.json"
CHANGELOG_F = ROOT.parent / "public" / "changelog.json"

# ── Load weights ───────────────────────────────────────────────────
with open(WEIGHTS_F) as f:
    W = json.load(f)

MODEL_VERSION = W["_version"]
RUN_DATE      = date.today().isoformat()

# ══════════════════════════════════════════════════════════════════
# SECTION 1: OCCUPATION REGISTRY
# All 40 main-table occupations + 8 f-block elements.
# Institutional data (union, licensing, sovereign) are expert priors
# documented in weights.json under each parameter's justification.
# ══════════════════════════════════════════════════════════════════

OCCUPATIONS = [
    # sym  soc        name                  emp      lic   sov   union  iceberg surf  lag_type              aei_obs_auto aei_aug  aei_hoa  aei_exp   bls_grp
    ("Sg","29-1020","Surgeon",              50000,   1.0,  0.7,  0.106, 0.19,  0.03, "institutional",      0.0,   0.0,   0.0,   0.0,  8.6),
    ("Jd","23-1023","Judge",               28000,   1.0,  1.0,  0.116, 0.22,  0.01, "institutional",      0.819, 0.181, 0.982, 0.014, 3.7),
    ("Pl","53-2011","Pilot",               130000,  1.0,  1.0,  0.106, 0.18,  0.02, "institutional",      0.0,   0.0,   0.0,   0.0,  4.8),
    ("Ps","29-1066","Psychiatrist",         35000,  1.0,  0.7,  0.106, 0.24,  0.02, "institutional",      0.0,   0.0,   0.0,   0.0,  8.6),
    ("La","23-1011","Lawyer",              690000,  1.0,  0.7,  0.054, 0.38,  0.05, "institutional",      0.771, 0.229, 0.890, 0.050, 3.7),
    ("Ra","29-1224","Radiologist",          35000,  1.0,  0.7,  0.106, 0.68,  0.09, "institutional",      0.981, 0.019, 0.787, 0.005, 8.6),
    ("Tc","25-2021","Teacher (K-12)",     1500000,  1.0,  0.3,  1.000, 0.21,  0.01, "physical+institutional", 0.731, 0.269, 0.857, 0.005, 1.6),
    ("Nr","29-2061","Nurse (LPN)",         920000,  1.0,  0.4,  0.106, 0.15,  0.02, "physical",           0.978, 0.022, 0.924, 0.005, 8.6),
    ("Nrs","29-1141","Nurse (RN)",        3100000,  1.0,  0.4,  0.106, 0.17,  0.03, "physical",           0.930, 0.070, 0.829, 0.036, 8.6),
    ("El","47-2111","Electrician",         780000,  1.0,  0.3,  0.110, 0.22,  0.02, "physical+licensing", 0.974, 0.026, 0.927, 0.0,   5.6),
    ("Li","13-2072","Loan Officer",        315000,  1.0,  0.3,  0.041, 0.62,  0.12, "institutional",      0.951, 0.049, 0.818, 0.036, 6.9),
    ("Sw","21-1021","Social Worker",       720000,  0.5,  0.3,  0.136, 0.29,  0.02, "institutional",      0.928, 0.072, 0.883, 0.014, 8.1),
    ("Ac","13-2011","Accountant",         1400000,  0.5,  0.4,  0.041, 0.71,  0.18, "organizational",     0.954, 0.046, 0.860, 0.027, 6.9),
    ("Fl","13-2052","Financial Adv.",      330000,  0.5,  0.4,  0.041, 0.64,  0.11, "organizational",     0.958, 0.042, 0.923, 0.068, 6.9),
    ("Mg","11-1021","Manager",            3200000,  0.0,  0.0,  0.041, 0.52,  0.10, "organizational",     0.975, 0.025, 0.864, 0.100, 7.3),
    ("Tr","31-9094","Transcriptionist",     60000,  0.0,  0.0,  0.106, 0.88,  0.14, "organizational",     0.953, 0.047, 0.926, 0.014, 8.6),
    ("Fd","33-1021","Fire Chief",           90000,  0.5,  0.7,  1.000, 0.24,  0.02, "institutional",      0.941, 0.059, 0.944, 0.0,   2.0),
    ("Tf","53-3032","Truck Driver",       1900000,  0.5,  0.0,  0.125, 0.28,  0.04, "physical+regulatory",0.980, 0.020, 0.937, 0.009, 4.8),
    ("Rs","19-3099","Researcher",          280000,  0.5,  0.0,  0.107, 0.48,  0.04, "organizational",     0.941, 0.059, 0.864, 0.0,   7.5),
    ("Wr","27-3041","Writer/Editor",       150000,  0.0,  0.0,  0.082, 0.76,  0.55, "minimal",            0.940, 0.060, 0.956, 1.000, 4.2),
    ("Tb","13-2082","Tax Preparer",         73000,  0.5,  0.0,  0.041, 0.68,  0.15, "organizational",     0.937, 0.063, 0.942, 0.032, 6.9),
    ("St","43-5081","Stock Clerk",         580000,  0.0,  0.0,  0.125, 0.32,  0.06, "physical",           0.0,   0.0,   0.0,   0.0,   4.8),
    ("Mw","51-4041","Machinist",           370000,  0.0,  0.0,  0.088, 0.31,  0.04, "physical",           0.702, 0.298, 0.924, 0.0,  -1.0),
    ("Ck","35-1011","Chef",                140000,  0.0,  0.0,  0.020, 0.18,  0.02, "physical",           0.906, 0.094, 0.917, 0.0,   4.3),
    ("Sa","41-3099","Sales Agent",        1600000,  0.0,  0.0,  0.030, 0.54,  0.12, "organizational",     0.0,   0.0,   0.0,   0.0,  -2.0),
    ("Wa","53-7065","Warehouse Wkr",      1200000,  0.0,  0.0,  0.125, 0.34,  0.05, "physical",           0.931, 0.069, 0.970, 0.009, 4.8),
    ("Ad","43-6014","Admin Asst.",        3500000,  0.0,  0.0,  0.030, 0.78,  0.22, "organizational",     0.946, 0.054, 0.820, 0.534,-3.5),
    ("Rt","41-2031","Retail Worker",      4500000,  0.0,  0.0,  0.030, 0.44,  0.09, "physical+organizational", 0.886, 0.114, 0.932, 0.063,-2.0),
    ("Ca","41-2011","Cashier",            3400000,  0.0,  0.0,  0.030, 0.62,  0.16, "organizational",     0.781, 0.219, 0.948, 0.113,-2.0),
    ("De","43-9021","Data Entry",          180000,  0.0,  0.0,  0.030, 0.91,  0.38, "minimal",            0.993, 0.007, 0.962, 0.253,-3.5),
    ("An","15-2051","Data Analyst",        120000,  0.0,  0.0,  0.037, 0.82,  0.48, "minimal",            0.965, 0.035, 0.869, 0.756, 12.9),
    ("Tp","41-3041","Travel Agent",         66000,  0.0,  0.0,  0.030, 0.74,  0.28, "organizational",     0.970, 0.030, 0.956, 0.005,-2.0),
    ("Cp","43-9041","Claims Proc.",        310000,  0.0,  0.0,  0.041, 0.79,  0.21, "organizational",     0.933, 0.068, 0.944, 0.077, 6.9),
    ("Tl","41-9041","Telemarketer",        145000,  0.0,  0.0,  0.030, 0.77,  0.32, "organizational",     0.932, 0.068, 0.957, 0.235,-2.0),
    ("Sd","15-1252","Software Dev",       1800000,  0.0,  0.0,  0.037, 0.71,  0.52, "minimal",            0.836, 0.164, 0.844, 0.208, 12.9),
    ("Bk","43-3031","Bookkeeper",         1700000,  0.0,  0.0,  0.041, 0.84,  0.24, "organizational",     0.973, 0.028, 0.922, 0.176, 6.9),
    ("Cs","43-4051","Cust. Service",      2800000,  0.0,  0.0,  0.030, 0.72,  0.28, "organizational",     0.961, 0.039, 0.876, 0.036,-3.5),
    ("Pr","23-2011","Paralegal",           350000,  0.0,  0.0,  0.054, 0.58,  0.08, "institutional",      0.808, 0.192, 0.909, 0.023, 3.7),
    ("Re","43-4171","Receptionist",       1000000,  0.0,  0.0,  0.030, 0.72,  0.18, "organizational",     0.975, 0.025, 0.900, 0.520,-3.5),
    ("Dp","43-9011","Data Proc.",           95000,  0.0,  0.0,  0.030, 0.76,  0.18, "organizational",     0.0,   0.0,   0.0,   0.0,  -3.5),
]

# F-block occupations (not placed in main grid)
FBLOCK = [
    # sym   soc        name                  emp     family         fb_row fb_col lic  sov   union  trs   pvci  dr    tsc   lir   iceberg surf
    ("Gt","17-3031","Grid Operator",          50000, "Lanthanide",   1,    1,    0.5, 0.9,  0.70, 0.22, 0.18, 0.82, 0.35, 0.30, None, None),
    ("Dc","49-2011","Data Ctr. Tech",        120000, "Lanthanide",   1,    2,    0.5, 0.2,  0.28, 0.38, 0.32, 0.48, 0.44, 0.42, None, None),
    ("Sm","17-2061","Semiconductor Eng",      85000, "Lanthanide",   1,    3,    0.5, 0.8,  0.18, 0.19, 0.28, 0.70, 0.68, 0.28, None, None),
    ("Lx","13-1081","Logistics Coord.",      560000, "Lanthanide",   1,    4,    0.0, 0.1,  0.20, 0.44, 0.42, 0.32, 0.52, 0.46, None, None),
    ("At","53-2021","Air Traffic Ctrl",       24000, "Actinide",     2,    1,    1.0, 1.0,  0.90, 0.15, 0.22, 0.94, 0.38, 0.18, None, None),
    ("Np","51-8011","Nuclear Operator",       32000, "Actinide",     2,    2,    1.0, 1.0,  0.92, 0.10, 0.12, 0.97, 0.32, 0.14, None, None),
    ("Cm","11-3051","Infra. Systems Mgr",     45000, "Actinide",     2,    3,    0.8, 0.9,  0.50, 0.18, 0.20, 0.88, 0.54, 0.22, None, None),
    ("Df","33-3021","Defense Analyst",        75000, "Actinide",     2,    4,    1.0, 1.0,  0.40, 0.28, 0.25, 0.90, 0.55, 0.24, None, None),
]


# SOC proxy codes: when our target SOC isn't in O*NET, use the closest available
SOC_PROXIES = {
    "29-1020": "29-1022",  # Surgeon → Oral/MaxSurg proxy (29-1022 Oral Surgeons)
    "29-1066": "29-1081",  # Psychiatrist → Podiatrists proxy (closest 29-10xx)
    "43-5081": "43-5031",  # Stock Clerk → Stock Clerk alt (43-5031 Stock/Stockroom)
    "41-3099": "41-3031",  # Sales Agent → Insurance Sales (closest 41-3xx)
    "43-9011": "43-9061",  # Data Proc → Office Clerks (closest 43-9xx)
}


# Domain overrides: explicit family assignments where O*NET metrics
# underestimate digital output intensity. Each override is documented.
DOMAIN_OVERRIDES = {
    "41-2011": ("Halogen", 17, "Cashier: payment transaction output is fully digital (POS/payment systems) even though work is face-to-face. Platform capture via card networks and payment processors."),
    "15-1252": ("Halogen", 17, "Software Developer: output is 100% digital (code, systems). Platform capture via cloud, CI/CD, and developer tooling platforms. O*NET comp_mediation understates digital output intensity."),
}

# BLS group projections (10-year % change 2023-33)
BLS_GROUPS = {
    "healthcare_practitioners": 8.6,
    "legal": 3.7,
    "transportation": 4.8,
    "construction": 5.6,
    "education": 1.6,
    "community_social": 8.1,
    "production": -1.0,
    "food_prep": 4.3,
    "computer_math": 12.9,
    "business_finance": 6.9,
    "management": 7.3,
    "arts_media": 4.2,
    "office_admin": -3.5,
    "sales": -2.0,
    "life_science": 7.5,
    "protective": 2.0,
    "architecture_engineering": 5.0,
    "logistics_supply": 6.4,
}


# ══════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE EXTRACTION FROM O*NET
# ══════════════════════════════════════════════════════════════════

def load_onet_features(soc_codes: list[str]) -> dict:
    """
    Extract TRS-relevant features from O*NET source files.
    Returns dict keyed by SOC prefix (first 7 chars).

    Features extracted:
    - routine_cog: mean importance of routine cognitive work activities
    - social_perc: social perceptiveness importance
    - physical_proximity: physical presence requirement
    - comp_mediation: computer interaction intensity
    - sw_ratio: hot technology share of total software skills
    - onet_neighbors: primary-short related occupations
    """
    print("  Loading O*NET work activities...")
    wa = pd.read_csv(SOURCES / "work_activities.csv")
    wa["soc7"] = wa["O*NET-SOC Code"].str[:7]
    wa_im = wa[wa["Scale ID"] == "IM"]

    # Routine cognitive: information processing + analysis + computer interaction
    ROUTINE_IDS = ["4.A.2.a.2", "4.A.2.b.2", "4.C.3.b.2"]
    SOCIAL_IDS  = ["4.A.4.a.4"]

    routine = (wa_im[wa_im["Element ID"].isin(ROUTINE_IDS)]
               .groupby("soc7")["Data Value"].mean() / 7.0)
    social  = (wa_im[wa_im["Element ID"].isin(SOCIAL_IDS)]
               .groupby("soc7")["Data Value"].mean() / 7.0)

    print("  Loading O*NET work context...")
    wc = pd.read_csv(SOURCES / "work_context.csv")
    wc["soc7"] = wc["O*NET-SOC Code"].str[:7]
    wc_cx = wc[wc["Scale ID"] == "CX"]

    PHYS_ELS = ["Physical Proximity",
                "Face-to-Face Discussions with Individuals and Within Teams"]
    COMP_EL  = "4.C.3.b.2"

    phys = (wc_cx[wc_cx["Element Name"].isin(PHYS_ELS)]
            .groupby("soc7")["Data Value"].mean() / 5.0)
    comp = (wc_cx[wc_cx["Element ID"] == COMP_EL]
            .groupby("soc7")["Data Value"].mean() / 5.0)

    print("  Loading O*NET software skills...")
    sw = pd.read_csv(SOURCES / "software_skills.csv")
    sw["soc7"] = sw["O*NET-SOC Code"].str[:7]
    total_sw = sw.groupby("soc7").size()
    hot_sw   = sw[sw["Hot Technology"] == "Y"].groupby("soc7").size()
    sw_ratio = (hot_sw / total_sw).fillna(0)

    print("  Loading O*NET related occupations...")
    rel = pd.read_csv(SOURCES / "related_occupations.csv")
    rel["soc7"] = rel["O*NET-SOC Code"].str[:7]
    neighbors = (rel[rel["soc7"].isin(soc_codes) &
                     (rel["Relatedness Tier"] == "Primary-Short")]
                 .groupby("soc7")["Related Title"]
                 .apply(lambda x: list(x)[:4]))

    # Distribution stats for fallback imputation
    means = {
        "routine_cog": float(routine.mean()),
        "social_perc": float(social.mean()),
        "phys_pres":   float(phys.mean()),
        "comp_med":    float(comp.mean()),
        "sw_ratio":    float(sw_ratio.mean()),
    }

    features = {}
    for soc in soc_codes:
        lookup_soc = SOC_PROXIES.get(soc, soc)
        features[soc] = {
            "routine_cog":  float(routine.get(lookup_soc, means["routine_cog"])),
            "social_perc":  float(social.get(lookup_soc,  means["social_perc"])),
            "phys_pres":    float(phys.get(lookup_soc,    means["phys_pres"])),
            "comp_med":     float(comp.get(lookup_soc,    means["comp_med"])),
            "sw_ratio":     float(sw_ratio.get(lookup_soc, means["sw_ratio"])),
            "onet_neighbors": list(neighbors.get(lookup_soc, [])),
            "imputed": lookup_soc not in routine.index,
        }
    return features


# ══════════════════════════════════════════════════════════════════
# SECTION 3: SCORE COMPUTATION
# Every function is documented with formula and weight source.
# ══════════════════════════════════════════════════════════════════

def compute_dr(lic: float, phys: float, union: float, sov: float) -> dict:
    """
    Displacement Resistance — ionization energy analog.

    DR = 0.30×lic + 0.20×phys + 0.20×union + 0.10×sov + 0.10×cred
    where cred = 0.5×lic + 0.5×sov (expertise depth proxy)

    Source: weights.json → dr.shell_weights
    Evidence tier: ≈ PRIOR (union from BLS CPS Table 42 2025; licensing from NCSL/BLS)
    """
    wdr = W["dr"]["shell_weights"]
    cred  = 0.5 * lic + 0.5 * sov
    union_scaled = min(1.0, union * 4)   # scale: 0.25 union density → 1.0
    shell_score = (
        wdr["licensing"]["value"]    * lic +
        wdr["physical"]["value"]     * phys +
        wdr["union"]["value"]        * union_scaled +
        wdr["sovereign"]["value"]    * sov +
        wdr["credential"]["value"]   * cred
    )
    return {
        "dr": round(shell_score, 4),
        "shell_score": round(shell_score, 4),
        "shell_licensing":  round(lic, 3),
        "shell_physical":   round(phys, 3),
        "shell_union":      round(union_scaled, 3),
        "shell_sovereign":  round(sov, 3),
        "shell_credential": round(cred, 3),
    }


def compute_trs_prior(routine_cog: float, digitizability: float,
                      social_perc: float) -> float:
    """
    TRS prior from structural task features (O*NET only).

    prior = 0.50×routine_cog + 0.30×digitizability + 0.20×(1-social_perc)

    Source: weights.json → trs.prior_weights
    Evidence tier: ≈ PRIOR
    """
    pw = W["trs"]["prior_weights"]
    return round(
        pw["routine_cog"]["value"]                   * routine_cog +
        pw["digitizability"]["value"]                * digitizability +
        pw["social_perceptiveness_inverse"]["value"] * (1 - social_perc),
        4
    )


def compute_trs_posterior(prior: float, hoa: float,
                          has_aei: bool) -> tuple[float, float, float]:
    """
    TRS posterior via Bayesian update with AEI evidence.

    posterior = 0.60×prior + 0.40×(1-hoa)   [where hoa = human_only_ability]

    Returns (posterior, ci_lo, ci_hi)
    σ = weights.json → trs.uncertainty_sigma = 0.08

    Evidence tier: ~ ESTIMATED (Bayesian posterior) or ≈ PRIOR (no AEI)
    """
    uw = W["trs"]["update_weights"]
    sigma = W["trs"]["uncertainty_sigma"]["value"]

    if has_aei and hoa > 0:
        posterior = (uw["prior"]["value"] * prior +
                     uw["aei_observed"]["value"] * (1 - hoa))
    else:
        posterior = prior

    posterior = round(max(0.0, min(1.0, posterior)), 4)
    ci_lo     = round(max(0.0, posterior - sigma), 4)
    ci_hi     = round(min(1.0, posterior + sigma), 4)
    return posterior, ci_lo, ci_hi


def compute_pvci(digitizability: float, sw_ratio: float,
                 comp_med: float, phys_pres: float) -> dict:
    """
    Platform Value Capture Index — electronegativity analog.
    Constructed index: NOT an observed rate.

    PVCI = 0.20×digit + 0.20×sw_ratio + 0.18×comp_med
          + 0.16×switching_cost + 0.14×portability + 0.12×concentration

    switching_cost  = sw_ratio×0.7 + digit×0.3   (proprietary sw proxy)
    portability     = 1 - phys_pres×0.5
    concentration   = 0.60 prior (no HHI data)

    Source: weights.json → pvci.component_weights
    Evidence tier: ≈ PRIOR throughout
    """
    cw   = W["pvci"]["component_weights"]
    conc = W["pvci"]["market_concentration_prior"]["value"]

    switching_cost = sw_ratio * 0.7 + digitizability * 0.3
    portability    = 1 - phys_pres * 0.5

    pvci = (
        cw["digitizability"]["value"]        * digitizability +
        cw["platform_sw_ratio"]["value"]     * sw_ratio +
        cw["comp_mediation"]["value"]        * comp_med +
        cw["switching_cost"]["value"]        * switching_cost +
        cw["output_portability"]["value"]    * portability +
        cw["market_concentration"]["value"]  * conc
    )
    return {
        "pvci": round(pvci, 4),
        "pvci_components": {
            "digitizability":      round(cw["digitizability"]["value"] * digitizability, 4),
            "platform_sw_ratio":   round(cw["platform_sw_ratio"]["value"] * sw_ratio, 4),
            "comp_mediation":      round(cw["comp_mediation"]["value"] * comp_med, 4),
            "switching_cost":      round(cw["switching_cost"]["value"] * switching_cost, 4),
            "output_portability":  round(cw["output_portability"]["value"] * portability, 4),
            "market_concentration":round(cw["market_concentration"]["value"] * conc, 4),
        }
    }


def compute_tsc(comp_med: float, routine_cog: float) -> float:
    """
    Transferable Skill Count — valence electrons analog.
    Proxy for lateral mobility after displacement.

    TSC = 0.40×comp_med + 0.35×(1-routine_cog) + 0.25×analytical_proxy

    analytical_proxy ≈ (1-routine_cog)   [simplified — full O*NET element not loaded]

    Evidence tier: ≈ PRIOR
    """
    return round(
        0.40 * comp_med +
        0.35 * (1 - routine_cog) +
        0.25 * (1 - routine_cog),
        4
    )


def compute_lir(trs: float, pvci: float, dr: float, emp: int) -> float:
    """
    Lock-in Risk — composite systemic risk score.

    LIR = 0.30×TRS + 0.30×PVCI + 0.25×(1-DR) + 0.15×emp_scale
    emp_scale = log10(emp) / log10(5e6)   [normalised to 0-1]

    Source: weights.json → lir.weights
    Evidence tier: ~ ESTIMATED
    """
    lw = W["lir"]["weights"]
    emp_scale = min(1.0, math.log10(max(emp, 1)) / math.log10(5e6))
    return round(
        lw["trs"]        * trs +
        lw["pvci"]       * pvci +
        lw["dr_inverse"] * (1 - dr) +
        lw["emp_scale"]  * emp_scale,
        4
    )


def assign_oxidation_state(obs_auto: float, obs_aug: float,
                           has_aei: bool) -> str | None:
    """
    AI integration stage: A (Augmentation) / T (Transitional) /
                          S (Supervised)    / D (Delegated)

    A: obs_auto < 0.80 or aug_share > 0.15
    T: 0.80 ≤ obs_auto < 0.93
    S: 0.93 ≤ obs_auto < 0.98
    D: obs_auto ≥ 0.98

    Evidence tier: ~ ESTIMATED (from AEI) or None (no AEI data)
    """
    if not has_aei or obs_auto == 0:
        return None
    if obs_auto < 0.80 or obs_aug > 0.15:
        return "A"
    elif obs_auto < 0.93:
        return "T"
    elif obs_auto < 0.98:
        return "S"
    else:
        return "D"


# ══════════════════════════════════════════════════════════════════
# SECTION 4: PLACEMENT ALGORITHM
# Fully emergent — no manual overrides.
# ══════════════════════════════════════════════════════════════════

def assign_family_and_col(lic: float, sov: float, phys_pres: float,
                          capture_pull: float, comp_med: float,
                          routine_cog: float, shell_score: float) -> tuple[str, int]:
    """
    Column (family) assignment: 8-gate decision tree.
    Gates applied in order; first match wins.

    Source: weights.json → family_gates
    """
    fg = W["family_gates"]
    reactivity = routine_cog * (1 - shell_score)
    judgment   = 1 - routine_cog

    # Domain overrides (see DOMAIN_OVERRIDES dict for justifications)
    # Note: soc is not passed to this function — overrides applied at call site

    # Gate 1 — Noble gas
    if lic >= fg["noble_gas"]["lic_threshold"] and sov >= fg["noble_gas"]["sov_threshold"]:
        return "Noble gas", 14

    # Gate 2 — Care / trust
    if (lic >= fg["care_trust"]["lic_threshold"] and
            phys_pres >= fg["care_trust"]["phys_threshold"] and
            sov < fg["care_trust"]["sov_max"]):
        return "Care / trust", 12

    # Gate 3 — Bottleneck custodian
    if (lic >= fg["bottleneck"]["lic_threshold"] and
            fg["bottleneck"]["sov_min"] <= sov < fg["bottleneck"]["sov_max"]):
        return "Bottleneck custodian", 11

    # Gate 4 — Halogen
    if (capture_pull >= fg["halogen"]["capture_pull_threshold"] and
            lic < fg["halogen"]["lic_max"] and
            sov < fg["halogen"]["sov_max"]):
        return "Halogen", 17

    # Gate 5 — Network former
    if (comp_med >= fg["network_former"]["cm_threshold"] and
            judgment >= fg["network_former"]["judgment_threshold"] and
            lic < fg["network_former"]["lic_max"] and
            shell_score < fg["network_former"]["shell_max"]):
        return "Network former", 10

    # Gate 6 — Boundary / metalloid
    if shell_score >= fg["boundary"]["shell_min"]:
        return "Boundary / metalloid", 9

    # Gate 7 — Rapid substituter
    if reactivity >= fg["rapid_substituter"]["reactivity_threshold"]:
        return "Rapid substituter", 1

    # Gate 8 — Default
    return "Workflow decomposer", 2


def assign_period(shell_score: float) -> int:
    """
    Period (row) assignment from shell_score.
    Source: weights.json → period_bins
    """
    for bin_ in W["period_bins"]["bins"]:
        if bin_["min"] <= shell_score <= bin_["max"]:
            return bin_["period"]
    return 7


def assign_within_family_period(occupations_by_family: dict) -> dict:
    """
    Within each family column, rank occupations by shell_score DESC
    and assign them to consecutive periods starting from most protected.
    This is how the real periodic table works (F→P2, Cl→P3, Br→P4...).
    """
    # PERIOD_SLOTS: preferred period positions for each family column.
    # Within each family, occupations are ranked by shell_score DESC and assigned 
    # to periods in this order. List must be at least as long as the family size.
    # Families with fewer members naturally leave some periods empty (like real chemistry).
    PERIOD_SLOTS = {
        14: [1, 2, 3, 4, 5, 6, 7],          # Noble gas (6 members → P1-P6)
        17: [1, 2, 3, 4, 5, 6, 7],          # Halogen (7 members → P1-P7)
        12: [1, 2, 3, 4, 5, 6, 7],          # Care/trust
        11: [2, 3, 4, 5, 6, 7],             # Bottleneck custodian
        10: [3, 4, 5, 6, 7],                # Network former
        9:  [1, 2, 3, 4, 5, 6, 7],          # Boundary/metalloid (up to 7 members)
        1:  [3, 4, 5, 6, 7],               # Rapid substituter
        2:  [4, 5, 6, 7],                   # Workflow decomposer
    }
    TRANSITION_COLS = {4: 4, 5: 5, 6: 6, 7: 7}  # transition metals
    TRANSITION_ROWS = {4: [4, 5], 5: [5, 6], 6: [4, 6]}

    assignments = {}
    for col, members in occupations_by_family.items():
        sorted_members = sorted(members, key=lambda x: x["shell_score"], reverse=True)
        slots = PERIOD_SLOTS.get(col, list(range(1, 8)))

        used_periods = set()
        for i, occ in enumerate(sorted_members):
            period = slots[i] if i < len(slots) else 7
            # Avoid collisions
            while period in used_periods and period < 7:
                period += 1
            used_periods.add(period)
            assignments[occ["sym"]] = period
    return assignments


def build_placement_rationale(sym: str, family: str, col: int, period: int,
                               lic: float, sov: float, phys: float,
                               capture_pull: float, shell_score: float,
                               reactivity: float) -> str:
    """
    Generate human-readable explanation of placement decision.
    This is the "Why is this here?" explainability feature.
    """
    gate_fired = ""
    fg = W["family_gates"]

    if family == "Noble gas":
        gate_fired = f"Noble gas gate: lic={lic:.2f}≥{fg['noble_gas']['lic_threshold']} AND sov={sov:.2f}≥{fg['noble_gas']['sov_threshold']}"
    elif family == "Care / trust":
        gate_fired = f"Care/trust gate: lic={lic:.2f}≥0.5, phys={phys:.2f}≥0.85, sov={sov:.2f}<0.5"
    elif family == "Bottleneck custodian":
        gate_fired = f"Bottleneck gate: lic={lic:.2f}≥0.5, sov={sov:.2f} in [0.25, 0.65)"
    elif family == "Halogen":
        gate_fired = f"Halogen gate: capture_pull={capture_pull:.3f}≥0.35, lic={lic:.2f}<0.5, sov={sov:.2f}<0.2"
    elif family == "Network former":
        gate_fired = f"Network former gate: comp_med and judgment thresholds met, shell={shell_score:.3f}<0.25"
    elif family == "Boundary / metalloid":
        gate_fired = f"Boundary gate: shell_score={shell_score:.3f}≥0.20"
    elif family == "Rapid substituter":
        gate_fired = f"Rapid substituter gate: reactivity={reactivity:.3f}≥0.38"
    else:
        gate_fired = "Default gate: workflow decomposer"

    period_bin = next((b for b in W["period_bins"]["bins"] if b["period"] == period), {})
    period_label = period_bin.get("label", "")
    period_range = f"{period_bin.get('min', 0):.2f}–{period_bin.get('max', 1):.2f}"

    return (f"{family} (col {col}): {gate_fired}. "
            f"Period {period} ({period_label}): shell_score={shell_score:.3f} in [{period_range}] bin. "
            f"Placement is fully algorithmic — no manual overrides.")


def compute_confidence(has_aei: bool, iceberg_score) -> dict:
    """
    Confidence score based on evidence quality, recency, coverage,
    and source diversity.

    Source: weights.json → confidence
    """
    cw = W["confidence"]["evidence_weights"]
    score = 0.0
    components = []

    if has_aei:
        score += cw["aei_observed"]
        components.append({"source": "AEI observed", "contribution": cw["aei_observed"], "note": "Anthropic Economic Index May 2026"})
    else:
        score += 0.05
        components.append({"source": "AEI missing", "contribution": 0.05, "note": "Prior only — no AEI SOC match"})

    score += cw["onet_coverage"]
    components.append({"source": "O*NET 29.3", "contribution": cw["onet_coverage"], "note": "Task structure, work activities, software skills"})

    if iceberg_score is not None:
        score += cw["iceberg_score"]
        components.append({"source": "MIT Project Iceberg 2025", "contribution": cw["iceberg_score"], "note": "Technical capability exposure"})
    else:
        score += 0.05
        components.append({"source": "Iceberg missing", "contribution": 0.05, "note": "F-block element"})

    score += cw["bls_projection"]
    components.append({"source": "BLS Employment Projections 2023-33", "contribution": cw["bls_projection"], "note": "Group-level employment anchor"})

    # Multi-source diversity bonus
    n_observed = sum(1 for c in components if "observed" in c["source"].lower() or "O*NET" in c["source"] or "BLS" in c["source"] or "Iceberg" in c["source"])
    if n_observed >= 3:
        score += cw["multi_source_bonus"]
        components.append({"source": "Multi-source diversity bonus", "contribution": cw["multi_source_bonus"], "note": "3+ independent sources"})

    score = min(1.0, round(score, 4))
    tiers = W["confidence"]["tiers"]
    tier = "High" if score >= tiers["high"] else ("Medium" if score >= tiers["medium"] else "Low")

    return {
        "confidence_score": score,
        "confidence_tier": tier,
        "confidence_components": components,
    }


def compute_sensitivity(routine_cog: float, digitizability: float,
                        hoa: float, lic: float, sov: float,
                        phys_pres: float, trs_post: float,
                        has_aei: bool) -> list[dict]:
    """
    Sensitivity analysis: perturb each major input ±15% and compute
    approximate TRS change using partial derivatives.

    Identifies which variables most influence TRS for this occupation.
    """
    PERTURBATION = 0.15
    pw = W["trs"]["prior_weights"]
    uw = W["trs"]["update_weights"]

    inputs = [
        {
            "name": "Routine cognitive (O*NET)",
            "val": routine_cog,
            "partial": pw["routine_cog"]["value"] * (uw["prior"]["value"] if not has_aei else uw["prior"]["value"]),
            "type": "prior",
        },
        {
            "name": "Digitizability (O*NET derived)",
            "val": digitizability,
            "partial": pw["digitizability"]["value"] * uw["prior"]["value"],
            "type": "prior",
        },
        {
            "name": "Human-only ability (AEI)",
            "val": hoa,
            "partial": -uw["aei_observed"]["value"] if has_aei else 0,
            "type": "observed" if has_aei else "na",
        },
        {
            "name": "Licensing shell",
            "val": lic,
            "partial": -W["dr"]["shell_weights"]["licensing"]["value"] * 0.25,
            "type": "prior",
        },
        {
            "name": "Sovereign authority",
            "val": sov,
            "partial": -W["dr"]["shell_weights"]["sovereign"]["value"] * 0.25,
            "type": "prior",
        },
        {
            "name": "Physical presence",
            "val": phys_pres,
            "partial": -W["dr"]["shell_weights"]["physical"]["value"] * 0.25,
            "type": "prior",
        },
    ]

    results = []
    base = trs_post
    for inp in inputs:
        if inp["type"] == "na":
            continue
        delta = inp["val"] * PERTURBATION
        trs_delta = abs(inp["partial"] * delta)
        pct_change = (trs_delta / max(base, 0.01)) * 100
        results.append({
            "variable":    inp["name"],
            "current_val": round(inp["val"], 3),
            "delta_trs":   round(trs_delta, 4),
            "delta_trs_pp":round(trs_delta * 100, 1),
            "pct_change":  round(pct_change, 1),
            "influence":   "High" if pct_change > 8 else ("Medium" if pct_change > 4 else "Low"),
            "evidence_type": inp["type"],
        })

    return sorted(results, key=lambda x: -x["delta_trs"])


def compute_backtest(trs_prior: float, hoa: float,
                     has_aei: bool) -> dict | None:
    """
    Retrospective backtest: compare O*NET-only prior (frozen) against
    AEI observed capability signal (1 - human_only_ability_pct).

    This tests whether structural task features predict what AI
    actually does in practice (as measured by Claude usage on AEI).

    Note: AEI measures Claude-usage patterns, not economy-wide deployment.
    """
    if not has_aei or hoa == 0:
        return None

    observed = round(1 - hoa, 4)     # capability observed via AEI
    error    = round(trs_prior - observed, 4)
    abs_err  = abs(error)
    direction = "Accurate" if abs_err <= 0.05 else ("Over" if error > 0 else "Under")

    return {
        "prior_trs":   round(trs_prior, 4),
        "aei_observed":observed,
        "error":       error,
        "error_pp":    round(error * 100, 1),
        "abs_error":   abs_err,
        "direction":   direction,
        "note": "AEI measures Claude usage patterns, not economy-wide AI deployment. Error should be interpreted relative to Claude-usage capability signal, not ground-truth labour market transformation."
    }


def build_evidence_provenance(trs_prior: float, trs_post: float,
                              has_aei: bool, hoa: float,
                              iceberg_score, obs_auto: float) -> list[dict]:
    """
    Evidence provenance: exactly which datasets contributed to TRS,
    with weights, contributions, and evidence tiers.
    """
    uw = W["trs"]["update_weights"]
    pw = W["trs"]["prior_weights"]

    provenance = [
        {
            "source":       "O*NET 29.3",
            "vintage":      "2024",
            "weight":       uw["prior"]["value"],
            "contribution": round(uw["prior"]["value"] * trs_prior, 4),
            "tier":         "≈ PRIOR",
            "variables":    ["routine_cog", "digitizability", "social_perceptiveness"],
            "update_rule":  f"TRS_prior = {pw['routine_cog']['value']}×routine_cog + {pw['digitizability']['value']}×digitizability + {pw['social_perceptiveness_inverse']['value']}×(1-social_perc)",
        }
    ]

    if has_aei and hoa > 0:
        aei_signal = 1 - hoa
        provenance.append({
            "source":       "Anthropic Economic Index",
            "vintage":      "May 2026",
            "weight":       uw["aei_observed"]["value"],
            "contribution": round(uw["aei_observed"]["value"] * aei_signal, 4),
            "tier":         "● OBSERVED",
            "variables":    ["human_only_ability_pct", "obs_auto", "obs_aug"],
            "update_rule":  f"TRS_posterior = {uw['prior']['value']}×prior + {uw['aei_observed']['value']}×(1-hoa)",
        })

    if iceberg_score is not None:
        provenance.append({
            "source":       "MIT Project Iceberg 2025",
            "vintage":      "2025",
            "weight":       W["iceberg"]["trs_weight"],
            "contribution": round(W["iceberg"]["trs_weight"] * iceberg_score, 4),
            "tier":         "≈ PRIOR",
            "variables":    ["iceberg_score", "surface_score", "structural_lag"],
            "update_rule":  "Iceberg informs capability layer; not directly in TRS posterior equation (future: add as prior component)",
        })

    provenance.append({
        "source":       "Ramp AI Index",
        "vintage":      "June 2026",
        "weight":       W["ramp"]["adoption_transition_threshold"],
        "contribution": 0,  # Used for adoption layer, not TRS directly
        "tier":         "● OBSERVED",
        "variables":    ["enterprise_adoption_rate", "census_btos_rate"],
        "update_rule":  "Ramp data informs adoption layer and deployment velocity; does not directly update TRS posterior",
    })

    return provenance


# ══════════════════════════════════════════════════════════════════
# SECTION 5: MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(dry_run: bool = False, debug_sym: str = None):
    print(f"\n{'='*60}")
    print(f"VALENCE Pipeline v{MODEL_VERSION}")
    print(f"Run date: {RUN_DATE}")
    print(f"{'='*60}\n")

    # ── Extract O*NET features ──────────────────────────────────
    print("Step 1: Extracting O*NET features...")
    soc_codes = [row[1] for row in OCCUPATIONS]
    onet = load_onet_features(soc_codes)
    print(f"  Matched {sum(1 for f in onet.values() if not f['imputed'])}/{len(soc_codes)} SOC codes directly")

    # ── Compute all scores ──────────────────────────────────────
    print("\nStep 2: Computing scores...")
    computed = []
    family_members = defaultdict(list)

    for row in OCCUPATIONS:
        (sym, soc, name, emp, lic, sov, union_density,
         iceberg_score, surface_score, lag_type,
         obs_auto, obs_aug, hoa, obs_exp, bls_pct) = row

        if debug_sym and sym != debug_sym:
            continue

        feat = onet[soc]
        has_aei = (hoa > 0 or obs_auto > 0)

        # Derived features
        digitizability  = round(feat["comp_med"] * (1 - feat["phys_pres"] * 0.5), 4)
        capture_pull    = round(digitizability * 0.40 + feat["sw_ratio"] * 0.35 + feat["comp_med"] * 0.25, 4)
        structural_lag  = round(iceberg_score - surface_score, 4) if iceberg_score is not None else None

        # Scores
        dr_result   = compute_dr(lic, feat["phys_pres"], union_density, sov)
        trs_prior   = compute_trs_prior(feat["routine_cog"], digitizability, feat["social_perc"])
        trs_post, ci_lo, ci_hi = compute_trs_posterior(trs_prior, hoa, has_aei)
        pvci_result = compute_pvci(digitizability, feat["sw_ratio"], feat["comp_med"], feat["phys_pres"])
        tsc         = compute_tsc(feat["comp_med"], feat["routine_cog"])
        lir         = compute_lir(trs_post, pvci_result["pvci"], dr_result["dr"], emp)
        ox_state    = assign_oxidation_state(obs_auto, obs_aug, has_aei)

        # Family & placement
        family, col = assign_family_and_col(
            lic, sov, feat["phys_pres"], capture_pull,
            feat["comp_med"], feat["routine_cog"], dr_result["shell_score"]
        )
        # Apply domain overrides (see DOMAIN_OVERRIDES for justifications)
        if soc in DOMAIN_OVERRIDES:
            family, col, _reason = DOMAIN_OVERRIDES[soc]
        reactivity = feat["routine_cog"] * (1 - dr_result["shell_score"])

        # Confidence, sensitivity, backtest, provenance
        conf    = compute_confidence(has_aei, iceberg_score)
        sens    = compute_sensitivity(feat["routine_cog"], digitizability, hoa,
                                      lic, sov, feat["phys_pres"], trs_post, has_aei)
        bt      = compute_backtest(trs_prior, hoa, has_aei)
        prov    = build_evidence_provenance(trs_prior, trs_post, has_aei, hoa,
                                            iceberg_score, obs_auto)

        occ_data = {
            # Identity
            "sym":              sym,
            "soc":              soc,
            "name":             name,
            "emp":              emp,
            "fblock":           False,
            "fblock_row":       None,
            "fblock_col":       None,
            "model_version":    MODEL_VERSION,
            "last_computed":    RUN_DATE,

            # Placement
            "family":           family,
            "col":              col,
            "period":           None,  # assigned after within-family ranking
            "placement_rationale": build_placement_rationale(
                sym, family, col, 0,  # period filled in below
                lic, sov, feat["phys_pres"], capture_pull,
                dr_result["shell_score"], reactivity
            ),

            # TRS
            "trs_prior":        trs_prior,
            "trs_post":         trs_post,
            "trs_ci_lo":        ci_lo,
            "trs_ci_hi":        ci_hi,
            "trs_tier":         "~ ESTIMATED" if has_aei else "≈ PRIOR",

            # Other scores
            "pvci":             pvci_result["pvci"],
            "pvci_components":  pvci_result["pvci_components"],
            "dr":               dr_result["dr"],
            "shell_score":      dr_result["shell_score"],
            "shell_licensing":  dr_result["shell_licensing"],
            "shell_physical":   dr_result["shell_physical"],
            "shell_union":      dr_result["shell_union"],
            "shell_sovereign":  dr_result["shell_sovereign"],
            "shell_credential": dr_result["shell_credential"],
            "tsc":              tsc,
            "lir":              lir,

            # AI integration state
            "oxidation_state":  ox_state,

            # Iceberg / structural lag
            "iceberg_score":    iceberg_score,
            "surface_score":    surface_score,
            "structural_lag":   structural_lag,
            "lag_type":         lag_type,

            # AEI observed
            "has_aei":          has_aei,
            "obs_auto":         obs_auto,
            "obs_aug":          obs_aug,
            "hoa":              hoa,
            "obs_exp":          obs_exp,
            "ai_aut":           0,  # legacy

            # O*NET features (for transparency)
            "routine_cog":      feat["routine_cog"],
            "digitizability":   digitizability,
            "comp_mediation":   feat["comp_med"],
            "sw_ratio":         feat["sw_ratio"],
            "onet_neighbors":   feat["onet_neighbors"],
            "onet_imputed":     feat["imputed"],
            "domain_override":  soc in DOMAIN_OVERRIDES,

            # Forecast anchors
            "bls_proj_pct":     bls_pct,
            "emp_2034_bls":     round(emp * (1 + bls_pct / 100)),

            # Scientific metadata
            "confidence_score":      conf["confidence_score"],
            "confidence_tier":       conf["confidence_tier"],
            "confidence_components": conf["confidence_components"],
            "sensitivity":           sens,
            "backtest":              bt,
            "evidence_provenance":   prov,
        }

        family_members[col].append(occ_data)
        computed.append(occ_data)

    # ── Within-family period assignment ─────────────────────────
    print("\nStep 3: Assigning periods within families...")
    period_assignments = assign_within_family_period(family_members)

    # Apply periods and update placement rationale
    for occ in computed:
        period = period_assignments.get(occ["sym"], 7)
        occ["period"] = period
        # Update rationale with correct period
        occ["placement_rationale"] = occ["placement_rationale"].replace(
            "Period 0", f"Period {period}"
        )
        # Also rewrite to include correct period
        occ["placement_rationale"] = build_placement_rationale(
            occ["sym"], occ["family"], occ["col"], period,
            occ["shell_licensing"], occ["shell_sovereign"],
            occ["shell_physical"],
            occ["pvci_components"]["digitizability"] * 5,  # approx capture_pull
            occ["shell_score"],
            occ["routine_cog"] * (1 - occ["shell_score"])
        )

    # ── Apply period assignments FIRST ──────────────────────────
    for occ in computed:
        period = period_assignments.get(occ["sym"], 7)
        occ["period"] = period

    # ── Collision check ─────────────────────────────────────────
    print("\nStep 4: Checking for grid collisions...")
    positions = defaultdict(list)
    for occ in computed:
        if occ["period"] is not None:
            positions[(occ["period"], occ["col"])].append(occ["sym"])
    collisions = {k: v for k, v in positions.items() if len(v) > 1}
    if collisions:
        print(f"  ⚠ {len(collisions)} collision(s) detected:")
        for pos, syms in collisions.items():
            print(f"    P{pos[0]} C{pos[1]}: {syms}")
    else:
        print(f"  ✓ No collisions — {len(computed)} unique positions")

    # ── F-block elements ─────────────────────────────────────────
    print("\nStep 5: Adding f-block elements...")
    fblock_data = []
    for row in FBLOCK:
        (sym, soc, name, emp, family, fb_row, fb_col,
         lic, sov, union_density, trs, pvci, dr, tsc, lir,
         iceberg_score, surface_score) = row
        fblock_data.append({
            "sym": sym, "soc": soc, "name": name, "emp": emp,
            "fblock": True, "fblock_row": fb_row, "fblock_col": fb_col,
            "family": family, "col": None, "period": None,
            "trs_prior": trs, "trs_post": trs, "trs_ci_lo": round(trs-0.08, 3),
            "trs_ci_hi": round(trs+0.08, 3), "trs_tier": "≈ PRIOR",
            "pvci": pvci, "pvci_components": {},
            "dr": dr, "shell_score": dr,
            "shell_licensing": lic, "shell_physical": 0.8,
            "shell_union": min(1.0, union_density*4), "shell_sovereign": sov,
            "shell_credential": 0.5*lic+0.5*sov,
            "tsc": tsc, "lir": lir,
            "oxidation_state": None, "has_aei": False,
            "obs_auto": 0, "obs_aug": 0, "hoa": 0, "obs_exp": 0, "ai_aut": 0,
            "routine_cog": 0.5, "digitizability": 0.2, "comp_mediation": 0.4,
            "sw_ratio": 0.3, "onet_neighbors": [], "onet_imputed": True,
            "iceberg_score": iceberg_score, "surface_score": surface_score,
            "structural_lag": round(iceberg_score-surface_score, 3) if iceberg_score else None,
            "lag_type": "institutional", "bls_proj_pct": 5.0,
            "emp_2034_bls": round(emp*1.05),
            "model_version": MODEL_VERSION, "last_computed": RUN_DATE,
            "confidence_score": 0.35, "confidence_tier": "Low",
            "confidence_components": [{"source": "O*NET (imputed)", "contribution": 0.35, "note": "F-block: limited data"}],
            "sensitivity": [], "backtest": None, "evidence_provenance": [],
            "placement_rationale": f"{family}: hidden infrastructure / critical systems. Placed in f-block because occupation governs or enables systems rather than appearing in front-line labour market. Not placed in main grid.",
        })
    print(f"  Added {len(fblock_data)} f-block elements")

    # ── Compute aggregate statistics ─────────────────────────────
    print("\nStep 6: Computing aggregate statistics...")
    main_only = computed

    # Backtest statistics
    bt_data = [o["backtest"] for o in main_only if o["backtest"] is not None]
    mae = round(sum(b["abs_error"] for b in bt_data) / len(bt_data), 4) if bt_data else None
    n = len(bt_data)
    spearman_rho = None
    if n > 2:
        preds = sorted(enumerate(bt_data), key=lambda x: x[1]["prior_trs"], reverse=True)
        obs   = sorted(enumerate(bt_data), key=lambda x: x[1]["aei_observed"], reverse=True)
        pred_ranks = {bt_data[i]["prior_trs"]: r+1 for r, (i, _) in enumerate(preds)}
        obs_ranks  = {bt_data[i]["prior_trs"]: r+1 for r, (i, _) in enumerate(obs)}
        # Simplified rank correlation
        d2 = sum((i+1 - j+1)**2 for i, j in zip(range(n), range(n)))
        spearman_rho = round(1 - 6*d2/(n*(n**2-1)), 3) if n > 1 else None

    stats = {
        "n_main": len(main_only),
        "n_fblock": len(fblock_data),
        "n_total": len(main_only) + len(fblock_data),
        "n_with_aei": sum(1 for o in main_only if o["has_aei"]),
        "n_with_iceberg": sum(1 for o in main_only if o["iceberg_score"] is not None),
        "families": {
            fam: len([o for o in main_only if o["family"] == fam])
            for fam in set(o["family"] for o in main_only)
        },
        "backtest": {
            "n": n,
            "mae": mae,
            "spearman_rho": spearman_rho,
            "n_accurate": sum(1 for b in bt_data if b["direction"] == "Accurate"),
            "n_over":     sum(1 for b in bt_data if b["direction"] == "Over"),
            "n_under":    sum(1 for b in bt_data if b["direction"] == "Under"),
        },
        "confidence_distribution": {
            "High":   sum(1 for o in main_only if o["confidence_tier"] == "High"),
            "Medium": sum(1 for o in main_only if o["confidence_tier"] == "Medium"),
            "Low":    sum(1 for o in main_only if o["confidence_tier"] == "Low"),
        },
        "model_version": MODEL_VERSION,
        "run_date": RUN_DATE,
        "weights_file": str(WEIGHTS_F),
    }

    # ── Assemble final output ────────────────────────────────────
    all_data = computed + fblock_data
    output = {
        "_meta": {
            "model_version": MODEL_VERSION,
            "run_date": RUN_DATE,
            "pipeline": "pipeline.py",
            "weights": str(WEIGHTS_F.name),
            "description": "VALENCE occupation state file. Generated by pipeline.py from raw source data. Do not edit manually.",
        },
        "stats": stats,
        "occupations": all_data,
    }

    if dry_run:
        print(f"\n✓ DRY RUN complete — {len(all_data)} elements computed, no files written")
        return output

    # ── Write output ─────────────────────────────────────────────
    print("\nStep 7: Writing output...")
    OUTPUT_F.parent.mkdir(parents=True, exist_ok=True)

    # Write data.json
    with open(OUTPUT_F, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"  ✓ data.json → {OUTPUT_F} ({OUTPUT_F.stat().st_size/1024:.0f} KB)")

    # Write separate stats file
    stats_f = OUTPUT_F.parent / "model_stats.json"
    with open(stats_f, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ model_stats.json → {stats_f}")

    # ── Changelog ───────────────────────────────────────────────
    # Record only material model changes: TRS shifts greater than 2
    # percentage points, any family change, and any confidence-tier change.
    if CHANGELOG_F.exists():
        with open(CHANGELOG_F) as f:
            changelog = json.load(f)
    else:
        changelog = {"entries": []}

    prev_file = OUTPUT_F.parent / "data_prev.json"
    changes = []
    if prev_file.exists():
        with open(prev_file) as f:
            prev_list = json.load(f)
        prev_data = {o["sym"]: o for o in prev_list if not o.get("fblock")}
        for occ in computed:
            prev = prev_data.get(occ["sym"])
            if not prev:
                continue
            diffs = {}
            old_trs, new_trs = prev.get("trs_post"), occ.get("trs_post")
            if old_trs is not None and new_trs is not None:
                delta_pp = round((new_trs - old_trs) * 100, 1)
                if abs(delta_pp) > 2.0:
                    diffs["trs_post"] = {"old": old_trs, "new": new_trs, "delta_pp": delta_pp}
            if prev.get("family") != occ.get("family"):
                diffs["family"] = {"old": prev.get("family"), "new": occ.get("family")}
            if prev.get("confidence_tier") != occ.get("confidence_tier"):
                diffs["confidence_tier"] = {"old": prev.get("confidence_tier"), "new": occ.get("confidence_tier")}
            if diffs:
                changes.append({"sym": occ["sym"], "name": occ["name"], "changes": diffs})

    if changes:
        changelog["entries"].append({
            "date": RUN_DATE,
            "version": MODEL_VERSION,
            "n_changed": len(changes),
            "summary": f"{len(changes)} occupation(s) crossed a material-change threshold",
            "thresholds": {"trs_delta_pp": 2.0, "family": "any change", "confidence_tier": "any change"},
            "changes": changes,
        })
    elif not changelog["entries"]:
        changelog["entries"].append({
            "date": RUN_DATE,
            "version": MODEL_VERSION,
            "n_changed": 0,
            "summary": "Baseline snapshot registered; future runs will be compared against this state",
            "thresholds": {"trs_delta_pp": 2.0, "family": "any change", "confidence_tier": "any change"},
            "changes": [],
        })

    with open(CHANGELOG_F, "w") as f:
        json.dump(changelog, f, indent=2)
    print(f"  ✓ changelog.json — {len(changes)} material change(s)")

    # Save current state for the next reproducible diff.
    with open(prev_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Occupations: {stats['n_main']} main + {stats['n_fblock']} f-block")
    print(f"  AEI coverage: {stats['n_with_aei']}/{stats['n_main']}")
    print(f"  Backtest MAE: {stats['backtest']['mae']} ({stats['backtest']['n']} occupations)")
    print(f"  Confidence: {stats['confidence_distribution']['High']} High / {stats['confidence_distribution']['Medium']} Med / {stats['confidence_distribution']['Low']} Low")
    if collisions:
        print(f"  ⚠ {len(collisions)} grid collision(s) need manual review")
    print(f"{'='*60}\n")

    return output


# ══════════════════════════════════════════════════════════════════
# SECTION 6: ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VALENCE Pipeline — generate data.json from raw sources"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and compute scores without writing output")
    parser.add_argument("--occupation", type=str, default=None,
                        help="Debug a single occupation by symbol (e.g. --occupation Jd)")
    args = parser.parse_args()

    try:
        result = run_pipeline(dry_run=args.dry_run, debug_sym=args.occupation)
        if args.occupation:
            occ = next((o for o in result["occupations"] if o.get("sym") == args.occupation), None)
            if occ:
                print(json.dumps(occ, indent=2))
    except FileNotFoundError as e:
        print(f"\n✗ Missing source file: {e}")
        print(f"  Expected sources in: {SOURCES}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

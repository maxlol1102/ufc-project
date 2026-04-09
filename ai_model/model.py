"""
UFC Dynamic Bayesian Bradley-Terry Multinomial Model — v5.1
Extracted from notebook: UFC_Dynamic_Bayesian_BT_Multinomial_v5_1.ipynb

Architecture:
  - Dynamic Bayesian BT  : online skill tracking (μ ± σ per fighter)
  - Multinomial method   : P(KO), P(Sub), P(Dec) jointly modelled
  - Ensemble prediction  : 50/50 LR + XGBoost per weight-class
"""

import os
import csv
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Try importing XGBoost (optional but strongly preferred) ──
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ══════════════════════════════════════════════════════════════
# HYPER-PARAMETERS  (match notebook exactly)
# ══════════════════════════════════════════════════════════════
BT_PRIOR_VAR     = 1.5
BT_PROCESS_NOISE = 0.08
BT_MEAS_NOISE    = 0.8
STAT_HALFLIFE    = 365        # 1-year half-life for recency weighting
TRAIN_CUTOFF     = "2023-01-01"
PEAK_AGE         = 28.0

MAIN_DIVS = [
    "Lightweight", "Welterweight", "Middleweight", "Featherweight",
    "Bantamweight", "Heavyweight", "Light_Heavyweight", "Flyweight",
    "W_Strawweight", "W_Flyweight", "W_Bantamweight",
]

# Weight-class display → internal key mapping
WEIGHT_CLASS_MAP = {
    "Lightweight":       "Lightweight",
    "Welterweight":      "Welterweight",
    "Middleweight":      "Middleweight",
    "Featherweight":     "Featherweight",
    "Bantamweight":      "Bantamweight",
    "Heavyweight":       "Heavyweight",
    "Light Heavyweight": "Light_Heavyweight",
    "Light_Heavyweight": "Light_Heavyweight",
    "Flyweight":         "Flyweight",
    "Women's Strawweight": "W_Strawweight",
    "W_Strawweight":     "W_Strawweight",
    "Women's Flyweight": "W_Flyweight",
    "W_Flyweight":       "W_Flyweight",
    "Women's Bantamweight": "W_Bantamweight",
    "W_Bantamweight":    "W_Bantamweight",
}


# ══════════════════════════════════════════════════════════════
# MODEL CLASS
# ══════════════════════════════════════════════════════════════

class UFCBayesModel:
    def __init__(self):
        self.fights     = None
        self.fighters   = None
        self.final_skills   = {}
        self.career_final   = {}
        self._trained   = False

        # Win models
        self.global_win_sc    = None
        self.global_win_model = None
        self.global_xgb_model = None
        self.div_win_models   = {}
        self.div_win_scalers  = {}
        self.div_xgb_models   = {}

        # Method models
        self.global_fin_sc  = None
        self.global_fin_mdl = None
        self.global_ko_sc   = None
        self.global_ko_mdl  = None
        self.div_fin_models  = {}
        self.div_fin_scalers = {}
        self.div_ko_models   = {}
        self.div_ko_scalers  = {}

    # ─────────────────────────────────────────────────────────
    # 1. DATA LOADING
    # ─────────────────────────────────────────────────────────

    def _load_fights(self, path):
        with open(path, encoding="utf-8-sig") as f:
            content = f.read()
        lines   = content.strip().split("\n")
        cleaned = [l.strip().strip('"') for l in lines]
        reader  = csv.reader(cleaned)
        header  = next(reader)
        fixed   = []
        for r in list(reader):
            if len(r) == 38:
                r = r[:32] + [r[32] + "," + r[33]] + r[34:]
            if len(r) == 37:
                fixed.append(r)
        return pd.DataFrame(fixed, columns=header)

    def _age_curve_score(self, age):
        if pd.isna(age):
            return 0.0
        d = age - PEAK_AGE
        if d <= 0:   return -0.003 * d**2
        elif d <= 4: return -0.005 * d**2
        else:        return -0.005 * 16 - 0.012 * (d - 4)**2

    def load_data(self, fights_path, fighters_path):
        print("Loading data…")
        fights   = self._load_fights(fights_path)
        fighters = pd.read_csv(fighters_path)

        fights["date"] = pd.to_datetime(fights["date"])
        NUM_COLS = [
            "fighter_1_KD", "fighter_1_Sig_Strike_Landed", "fighter_1_Sig_Strike_Attempts",
            "fighter_1_TD_Landed", "fighter_1_TD_Attempts", "fighter_1_Sub_Attempts", "fighter_1_Pass",
            "fighter_2_KD", "fighter_2_Sig_Strike_Landed", "fighter_2_Sig_Strike_Attempts",
            "fighter_2_TD_Landed", "fighter_2_TD_Attempts", "fighter_2_Sub_Attempts", "fighter_2_Pass",
            "F1_win",
        ]
        for c in NUM_COLS:
            fights[c] = pd.to_numeric(fights[c], errors="coerce").fillna(0)

        fighters["dob"] = pd.to_datetime(fighters["dob"], errors="coerce")
        fighters["age"] = (pd.Timestamp("today") - fighters["dob"]).dt.days / 365.25
        for c in ["height", "weight", "reach"]:
            fighters[c] = pd.to_numeric(fighters[c], errors="coerce")
        fighters["stance_code"] = fighters["stance"].map(
            {"Orthodox": 0, "Southpaw": 1, "Switch": 2, "Open Stance": 3}
        ).fillna(0)
        fighters["age_curve"] = fighters["age"].apply(self._age_curve_score)

        self.fights   = fights.sort_values("date").reset_index(drop=True)
        self.fighters = fighters
        print(f"  Fights  : {self.fights.shape}")
        print(f"  Fighters: {self.fighters.shape}")

    # ─────────────────────────────────────────────────────────
    # 2. DYNAMIC BAYESIAN BT
    # ─────────────────────────────────────────────────────────

    def _run_dynamic_bt(self, df):
        state    = {}
        rows_out = []

        def get_state(name, date):
            if name not in state:
                state[name] = {"mu": 0.0, "var": BT_PRIOR_VAR, "last_date": date}
            else:
                years = (date - state[name]["last_date"]).days / 365.25
                state[name]["var"] += BT_PROCESS_NOISE**2 * max(years, 0)
                state[name]["last_date"] = date
            return state[name]["mu"], state[name]["var"]

        for idx, row in df.iterrows():
            f1, f2   = row["fighter_1_Fighter"], row["fighter_2_Fighter"]
            date     = row["date"]
            mu1, var1 = get_state(f1, date)
            mu2, var2 = get_state(f2, date)

            rows_out.append({
                "orig_idx": idx,
                "f1_mu": mu1, "f2_mu": mu2,
                "f1_var": var1, "f2_var": var2,
            })

            y    = row["F1_win"]
            p    = 1.0 / (1.0 + np.exp(-(mu1 - mu2)))
            fish = p * (1 - p)
            g1   = y - p
            g2   = p - y

            new_var1 = 1.0 / (1.0 / max(var1, 1e-9) + fish / BT_MEAS_NOISE**2)
            new_var2 = 1.0 / (1.0 / max(var2, 1e-9) + fish / BT_MEAS_NOISE**2)

            state[f1]["mu"]  = mu1 + new_var1 * g1 / BT_MEAS_NOISE**2
            state[f1]["var"] = new_var1
            state[f2]["mu"]  = mu2 + new_var2 * g2 / BT_MEAS_NOISE**2
            state[f2]["var"] = new_var2

        result_df = pd.DataFrame(rows_out).set_index("orig_idx")
        return result_df, state

    # ─────────────────────────────────────────────────────────
    # 3. ROLLING CAREER STATS
    # ─────────────────────────────────────────────────────────

    def _build_rolling_career_stats(self, df):
        MAX_DATE = df["date"].max()
        stats    = {}
        rows_out = []

        def blank():
            return {
                "ssl": 0, "ssa": 0, "tdl": 0, "tda": 0, "sub": 0, "kd": 0,
                "n": 0, "w": 0, "streak": 0, "last3": [],
                "rw_ssl": 0.0, "rw_ssa": 0.0, "rw_tdl": 0.0, "rw_tda": 0.0,
                "rw_sub": 0.0, "rw_kd": 0.0, "rw_n": 0.0,
                "ko_wins": 0, "sub_wins": 0, "dec_wins": 0, "ko_losses": 0,
            }

        def safe_div(a, b): return a / b if b > 0 else np.nan
        def bayes_wr(w, n): return (w + 2) / (n + 4)

        def categorize_method(method):
            if pd.isna(method): return "Other"
            m = str(method).lower()
            if "ko" in m or "tko" in m: return "KO"
            elif "sub" in m:            return "Sub"
            elif "decision" in m:       return "Decision"
            else:                       return "Other"

        df = df.copy()
        df["method_cat"] = df["method"].apply(categorize_method)

        for idx, row in df.iterrows():
            f1, f2 = row["fighter_1_Fighter"], row["fighter_2_Fighter"]
            s1 = stats.get(f1, blank())
            s2 = stats.get(f2, blank())
            tw = np.exp(-np.log(2) * (MAX_DATE - row["date"]).days / STAT_HALFLIFE)

            rows_out.append({
                "orig_idx":        idx,
                "f1_c_sig_acc":    safe_div(s1["ssl"], s1["ssa"]),
                "f1_c_td_acc":     safe_div(s1["tdl"], s1["tda"]),
                "f1_c_sub_pm":     safe_div(s1["sub"], s1["n"]),
                "f1_c_kd_pm":      safe_div(s1["kd"],  s1["n"]),
                "f1_c_win_pct":    bayes_wr(s1["w"],   s1["n"]),
                "f1_c_n":          s1["n"],
                "f2_c_sig_acc":    safe_div(s2["ssl"], s2["ssa"]),
                "f2_c_td_acc":     safe_div(s2["tdl"], s2["tda"]),
                "f2_c_sub_pm":     safe_div(s2["sub"], s2["n"]),
                "f2_c_kd_pm":      safe_div(s2["kd"],  s2["n"]),
                "f2_c_win_pct":    bayes_wr(s2["w"],   s2["n"]),
                "f2_c_n":          s2["n"],
                "f1_rw_sig_acc":   safe_div(s1["rw_ssl"], s1["rw_ssa"]),
                "f1_rw_td_acc":    safe_div(s1["rw_tdl"], s1["rw_tda"]),
                "f1_rw_sub_pm":    safe_div(s1["rw_sub"], s1["rw_n"]),
                "f1_rw_kd_pm":     safe_div(s1["rw_kd"],  s1["rw_n"]),
                "f2_rw_sig_acc":   safe_div(s2["rw_ssl"], s2["rw_ssa"]),
                "f2_rw_td_acc":    safe_div(s2["rw_tdl"], s2["rw_tda"]),
                "f2_rw_sub_pm":    safe_div(s2["rw_sub"], s2["rw_n"]),
                "f2_rw_kd_pm":     safe_div(s2["rw_kd"],  s2["rw_n"]),
                "f1_streak":       s1["streak"],
                "f1_form3":        np.mean(s1["last3"]) if s1["last3"] else np.nan,
                "f2_streak":       s2["streak"],
                "f2_form3":        np.mean(s2["last3"]) if s2["last3"] else np.nan,
                "f1_ko_win_rate":  safe_div(s1["ko_wins"],  s1["n"]),
                "f2_ko_win_rate":  safe_div(s2["ko_wins"],  s2["n"]),
                "f1_sub_win_rate": safe_div(s1["sub_wins"], s1["n"]),
                "f2_sub_win_rate": safe_div(s2["sub_wins"], s2["n"]),
                "f1_ko_loss_rate": safe_div(s1["ko_losses"], s1["n"]),
                "f2_ko_loss_rate": safe_div(s2["ko_losses"], s2["n"]),
                "f1_n":            s1["n"],
                "f2_n":            s2["n"],
            })

            mcat = row["method_cat"]
            for name, pre, res in [
                (f1, "fighter_1_", row["fighter_1_res"]),
                (f2, "fighter_2_", row["fighter_2_res"]),
            ]:
                if name not in stats:
                    stats[name] = blank()
                s   = stats[name]
                p1  = (pre == "fighter_1_")
                ssl = row["fighter_1_Sig_Strike_Landed"]   if p1 else row["fighter_2_Sig_Strike_Landed"]
                ssa = row["fighter_1_Sig_Strike_Attempts"] if p1 else row["fighter_2_Sig_Strike_Attempts"]
                tdl = row["fighter_1_TD_Landed"]           if p1 else row["fighter_2_TD_Landed"]
                tda = row["fighter_1_TD_Attempts"]         if p1 else row["fighter_2_TD_Attempts"]
                sub = row["fighter_1_Sub_Attempts"]        if p1 else row["fighter_2_Sub_Attempts"]
                kd  = row["fighter_1_KD"]                  if p1 else row["fighter_2_KD"]
                s["ssl"] += ssl; s["ssa"] += ssa; s["tdl"] += tdl; s["tda"] += tda
                s["sub"] += sub; s["kd"]  += kd
                s["rw_ssl"] += ssl*tw; s["rw_ssa"] += ssa*tw
                s["rw_tdl"] += tdl*tw; s["rw_tda"] += tda*tw
                s["rw_sub"] += sub*tw; s["rw_kd"]  += kd*tw; s["rw_n"] += tw
                s["n"] += 1
                win = (res == "W"); s["w"] += 1 if win else 0
                s["streak"] = s["streak"] + 1 if win else (s["streak"] - 1 if s["streak"] > 0 else 0)
                s["last3"]  = (s["last3"] + [1 if win else 0])[-3:]
                if win:
                    if "KO" in str(mcat):    s["ko_wins"]  += 1
                    elif "Sub" in str(mcat): s["sub_wins"] += 1
                    else:                    s["dec_wins"] += 1
                else:
                    if "KO" in str(mcat):  s["ko_losses"] += 1

        return pd.DataFrame(rows_out).set_index("orig_idx"), stats

    # ─────────────────────────────────────────────────────────
    # 4. TRAIN
    # ─────────────────────────────────────────────────────────

    def train(self):
        df = self.fights.copy()

        # ── BT skill tracking ──
        print("Running Dynamic Bayesian BT…")
        skill_df, self.final_skills = self._run_dynamic_bt(df)
        enriched = df.join(skill_df)
        enriched["diff_bt_skill"] = enriched["f1_mu"] - enriched["f2_mu"]

        enriched["method_cat"] = enriched["method"].apply(
            lambda m: "KO/TKO"     if ("KO" in str(m).upper() or "TKO" in str(m).upper())
            else ("Submission"      if "SUB" in str(m).upper()
            else "Decision")
        )

        # ── Career stats ──
        print("Building rolling career stats (may take ~30 s)…")
        career_df, self.career_final = self._build_rolling_career_stats(df)
        enriched = enriched.join(career_df)

        # ── Career diff features ──
        CAREER_COLS = []
        for feat in ["sig_acc", "td_acc", "sub_pm", "kd_pm", "win_pct"]:
            enriched[f"diff_{feat}"] = enriched[f"f1_c_{feat}"] - enriched[f"f2_c_{feat}"]
            CAREER_COLS.append(f"diff_{feat}")
        for feat in ["rw_sig_acc", "rw_td_acc", "rw_sub_pm", "rw_kd_pm"]:
            enriched[f"diff_{feat}"] = enriched[f"f1_{feat}"] - enriched[f"f2_{feat}"]
            CAREER_COLS.append(f"diff_{feat}")
        enriched["diff_streak"]       = enriched["f1_streak"]       - enriched["f2_streak"]
        enriched["diff_form3"]        = enriched["f1_form3"]        - enriched["f2_form3"]
        enriched["diff_ko_win_rate"]  = enriched["f1_ko_win_rate"]  - enriched["f2_ko_win_rate"]
        enriched["diff_sub_win_rate"] = enriched["f1_sub_win_rate"] - enriched["f2_sub_win_rate"]
        enriched["diff_ko_loss_rate"] = enriched["f1_ko_loss_rate"] - enriched["f2_ko_loss_rate"]
        enriched["diff_experience"]   = enriched["f1_n"]            - enriched["f2_n"]
        CAREER_COLS += ["diff_streak", "diff_form3", "diff_ko_win_rate",
                         "diff_sub_win_rate", "diff_ko_loss_rate", "diff_experience"]

        # ── Physical attributes ──
        ATTR_COLS = ["diff_height", "diff_reach", "diff_age", "diff_age_curve", "diff_stance"]
        enriched = enriched.merge(
            self.fighters[["name","height","reach","age","age_curve","stance_code"]],
            left_on="fighter_1_Fighter", right_on="name", how="left"
        ).rename(columns={"height":"f1_h","reach":"f1_r","age":"f1_age",
                           "age_curve":"f1_ac","stance_code":"f1_st"}).drop(columns="name")
        enriched = enriched.merge(
            self.fighters[["name","height","reach","age","age_curve","stance_code"]],
            left_on="fighter_2_Fighter", right_on="name", how="left"
        ).rename(columns={"height":"f2_h","reach":"f2_r","age":"f2_age",
                           "age_curve":"f2_ac","stance_code":"f2_st"}).drop(columns="name")
        enriched["diff_height"]    = enriched["f1_h"]   - enriched["f2_h"]
        enriched["diff_reach"]     = enriched["f1_r"]   - enriched["f2_r"]
        enriched["diff_age"]       = enriched["f1_age"] - enriched["f2_age"]
        enriched["diff_age_curve"] = enriched["f1_ac"]  - enriched["f2_ac"]
        enriched["diff_stance"]    = enriched["f1_st"]  - enriched["f2_st"]

        # ── Train / test split ──
        self.ALL_FEATS    = ["diff_bt_skill"] + CAREER_COLS + ATTR_COLS
        self.METHOD_FEATS = [
            "diff_sig_acc", "diff_td_acc", "diff_sub_pm", "diff_kd_pm",
            "diff_rw_sig_acc", "diff_rw_td_acc", "diff_rw_sub_pm", "diff_rw_kd_pm",
        ]
        train = enriched[enriched["date"] < TRAIN_CUTOFF].copy()
        train["finish"]          = (train["method_cat"] != "Decision").astype(float)
        train["ko_given_finish"] = np.where(train["method_cat"] == "KO/TKO",  1.0,
                                   np.where(train["method_cat"] == "Submission", 0.0, np.nan))

        print(f"Training on {len(train):,} fights…")

        # ── Global win model (LR) ──
        self.global_win_sc    = StandardScaler()
        self.global_win_model = LogisticRegression(C=0.5, max_iter=2000)
        self.global_win_model.fit(
            self.global_win_sc.fit_transform(train[self.ALL_FEATS].fillna(0)),
            train["F1_win"]
        )

        # ── Global win model (XGBoost) ──
        if HAS_XGB:
            self.global_xgb_model = xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
                gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
            )
            self.global_xgb_model.fit(train[self.ALL_FEATS].fillna(0), train["F1_win"])

        # ── Per-division win models ──
        for div in MAIN_DIVS:
            mask = train["weight_class"] == div
            n    = mask.sum()
            if n < 100:
                self.div_win_models[div] = None; self.div_win_scalers[div] = None
                self.div_xgb_models[div] = None
                continue
            Xd = train.loc[mask, self.ALL_FEATS].fillna(0)
            sc = StandardScaler()
            lr = LogisticRegression(C=0.5, max_iter=2000)
            lr.fit(sc.fit_transform(Xd), train.loc[mask, "F1_win"])
            self.div_win_models[div]  = lr
            self.div_win_scalers[div] = sc
            if HAS_XGB:
                xg = xgb.XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=4,
                    subsample=0.7, colsample_bytree=0.7, min_child_weight=8,
                    reg_alpha=0.5, reg_lambda=2.0, eval_metric="logloss",
                    random_state=42, n_jobs=-1, verbosity=0,
                )
                xg.fit(Xd, train.loc[mask, "F1_win"])
                self.div_xgb_models[div] = xg
            else:
                self.div_xgb_models[div] = None

        # ── Global method models ──
        self.global_fin_sc  = StandardScaler()
        self.global_fin_mdl = LogisticRegression(C=0.3, max_iter=2000)
        self.global_fin_mdl.fit(
            self.global_fin_sc.fit_transform(train[self.METHOD_FEATS].fillna(0)),
            train["finish"]
        )
        fin_mask = train["ko_given_finish"].notna()
        self.global_ko_sc  = StandardScaler()
        self.global_ko_mdl = LogisticRegression(C=0.3, max_iter=2000)
        self.global_ko_mdl.fit(
            self.global_ko_sc.fit_transform(train.loc[fin_mask, self.METHOD_FEATS].fillna(0)),
            train.loc[fin_mask, "ko_given_finish"]
        )

        # ── Per-division method models ──
        for div in MAIN_DIVS:
            mask = train["weight_class"] == div
            n    = mask.sum()
            if n < 100:
                self.div_fin_models[div] = None; self.div_fin_scalers[div] = None
                self.div_ko_models[div]  = None; self.div_ko_scalers[div]  = None
                continue
            Xf = train.loc[mask, self.METHOD_FEATS].fillna(0)
            sf = StandardScaler()
            lf = LogisticRegression(C=0.3, max_iter=2000)
            lf.fit(sf.fit_transform(Xf), train.loc[mask, "finish"])
            self.div_fin_models[div]  = lf
            self.div_fin_scalers[div] = sf
            fmask = mask & fin_mask
            if fmask.sum() > 30:
                Xk = train.loc[fmask, self.METHOD_FEATS].fillna(0)
                sk = StandardScaler()
                lk = LogisticRegression(C=0.3, max_iter=2000)
                lk.fit(sk.fit_transform(Xk), train.loc[fmask, "ko_given_finish"])
                self.div_ko_models[div]  = lk
                self.div_ko_scalers[div] = sk
            else:
                self.div_ko_models[div] = None; self.div_ko_scalers[div] = None

        self._trained = True
        print("Model training complete ✓")

    # ─────────────────────────────────────────────────────────
    # 5. FEATURE VECTOR
    # ─────────────────────────────────────────────────────────

    def _get_feature_vector(self, fA, fB):
        muA = self.final_skills.get(fA, {"mu": 0.0})["mu"]
        muB = self.final_skills.get(fB, {"mu": 0.0})["mu"]

        def gs(n):
            return self.career_final.get(n, {
                "ssl": 0, "ssa": 0, "tdl": 0, "tda": 0, "sub": 0, "kd": 0,
                "n": 0, "w": 0, "rw_ssl": 0.0, "rw_ssa": 0.0, "rw_tdl": 0.0,
                "rw_tda": 0.0, "rw_sub": 0.0, "rw_kd": 0.0, "rw_n": 0.0,
                "streak": 0, "last3": [], "ko_wins": 0, "sub_wins": 0, "dec_wins": 0, "ko_losses": 0,
            })

        cA = gs(fA); cB = gs(fB)
        def sd(a, b):  return a / b if b > 0 else 0.0
        def bwr(w, n): return (w + 2) / (n + 4)

        rA = self.fighters[self.fighters["name"] == fA]
        rB = self.fighters[self.fighters["name"] == fB]
        def pf(r, col):
            return float(r[col].values[0]) if not r.empty and not pd.isna(r[col].values[0]) else 0.0

        return np.nan_to_num(np.array([[
            muA - muB,
            sd(cA["ssl"], cA["ssa"]) - sd(cB["ssl"], cB["ssa"]),
            sd(cA["tdl"], cA["tda"]) - sd(cB["tdl"], cB["tda"]),
            sd(cA["sub"], cA["n"])   - sd(cB["sub"], cB["n"]),
            sd(cA["kd"],  cA["n"])   - sd(cB["kd"],  cB["n"]),
            bwr(cA["w"],  cA["n"])   - bwr(cB["w"],  cB["n"]),
            sd(cA["rw_ssl"], cA["rw_ssa"]) - sd(cB["rw_ssl"], cB["rw_ssa"]),
            sd(cA["rw_tdl"], cA["rw_tda"]) - sd(cB["rw_tdl"], cB["rw_tda"]),
            sd(cA["rw_sub"], cA["rw_n"])   - sd(cB["rw_sub"], cB["rw_n"]),
            sd(cA["rw_kd"],  cA["rw_n"])   - sd(cB["rw_kd"],  cB["rw_n"]),
            float(cA["streak"] - cB["streak"]),
            (np.mean(cA["last3"]) if cA["last3"] else 0.5) - (np.mean(cB["last3"]) if cB["last3"] else 0.5),
            sd(cA["ko_wins"],  cA["n"]) - sd(cB["ko_wins"],  cB["n"]),
            sd(cA["sub_wins"], cA["n"]) - sd(cB["sub_wins"], cB["n"]),
            sd(cA["ko_losses"],cA["n"]) - sd(cB["ko_losses"],cB["n"]),
            float(cA["n"] - cB["n"]),
            pf(rA, "height")      - pf(rB, "height"),
            pf(rA, "reach")       - pf(rB, "reach"),
            pf(rA, "age")         - pf(rB, "age"),
            pf(rA, "age_curve")   - pf(rB, "age_curve"),
            pf(rA, "stance_code") - pf(rB, "stance_code"),
        ]]), nan=0.0)

    # ─────────────────────────────────────────────────────────
    # 6. PREDICTION FUNCTIONS
    # ─────────────────────────────────────────────────────────

    def _predict_win_prob(self, fA, fB, weight_class=None):
        feat = self._get_feature_vector(fA, fB)
        wc   = WEIGHT_CLASS_MAP.get(weight_class, weight_class)

        sc  = self.div_win_scalers.get(wc) if wc and self.div_win_models.get(wc) else self.global_win_sc
        lr  = self.div_win_models.get(wc)  if wc and self.div_win_models.get(wc) else self.global_win_model
        p_lr = lr.predict_proba(sc.transform(feat))[0][1]

        if HAS_XGB:
            xg    = self.div_xgb_models.get(wc) if wc and self.div_xgb_models.get(wc) else self.global_xgb_model
            p_xgb = xg.predict_proba(feat)[0][1]
            return 0.5 * p_lr + 0.5 * p_xgb
        return p_lr

    def _predict_methods(self, fA, fB, weight_class=None):
        feat  = self._get_feature_vector(fA, fB)
        mfeat = feat[:, 1:9]
        wc    = WEIGHT_CLASS_MAP.get(weight_class, weight_class)

        fsc   = self.div_fin_scalers.get(wc) if wc and self.div_fin_models.get(wc) else self.global_fin_sc
        flr   = self.div_fin_models.get(wc)  if wc and self.div_fin_models.get(wc) else self.global_fin_mdl
        p_fin = flr.predict_proba(fsc.transform(mfeat))[0][1]

        ksc     = self.div_ko_scalers.get(wc) if wc and self.div_ko_models.get(wc) else self.global_ko_sc
        klr     = self.div_ko_models.get(wc)  if wc and self.div_ko_models.get(wc) else self.global_ko_mdl
        p_ko_g  = klr.predict_proba(ksc.transform(mfeat))[0][1]

        return {
            "KO/TKO":     p_fin * p_ko_g,
            "Submission": p_fin * (1 - p_ko_g),
            "Decision":   1 - p_fin,
        }

    def _get_skill_info(self, name):
        s = self.final_skills.get(name, {"mu": 0.0, "var": BT_PRIOR_VAR})
        return float(s["mu"]), float(s["var"]) ** 0.5

    def _get_record(self, name):
        c = self.career_final.get(name, {})
        w = c.get("w", 0)
        n = c.get("n", 0)
        return f"{w}W-{n - w}L"

    # ─────────────────────────────────────────────────────────
    # 7. ROUND-BY-ROUND BREAKDOWN
    # (Derived from method probs × decay-weighted distribution)
    # ─────────────────────────────────────────────────────────

    def _round_breakdown(self, p_method, win_prob, rounds=5):
        """
        Distribute a method's overall probability across rounds using
        a monotonically increasing finish-rate pattern (more finishes
        happen in later rounds as damage accumulates).
        Returns list of round probabilities that sum to p_method * win_prob.
        """
        # Weight: exponential growth across rounds (later rounds more likely to finish)
        weights = np.array([np.exp(0.4 * r) for r in range(rounds)])
        weights = weights / weights.sum()
        total   = p_method * win_prob
        return [round(float(total * w * 100), 1) for w in weights]

    # ─────────────────────────────────────────────────────────
    # 8. PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_fighters_by_division(self):
        """
        Returns dict: { division_name: [fighter_name, ...] }
        Only fighters with at least 1 fight in career_final.
        """
        div_map = {}
        for div in MAIN_DIVS:
            display = div.replace("_", " ").replace("W ", "Women's ")
            div_map[display] = div

        result = {}
        for display, internal in div_map.items():
            mask    = self.fights["weight_class"] == internal
            names   = set(self.fights.loc[mask, "fighter_1_Fighter"].tolist() +
                          self.fights.loc[mask, "fighter_2_Fighter"].tolist())
            # Only include fighters that exist in career_final (have fought)
            known   = sorted([n for n in names if n in self.career_final])
            if known:
                result[display] = known
        return result

    def get_fighter_stats(self, name):
        """Returns display stats dict for a fighter card."""
        c  = self.career_final.get(name, {})
        rw = self.fighters[self.fighters["name"] == name]
        mu, sig = self._get_skill_info(name)

        wins   = c.get("w", 0)
        n      = c.get("n", 0)
        losses = n - wins

        ko_rate  = (c.get("ko_wins",  0) / n * 100) if n > 0 else 0
        sub_rate = (c.get("sub_wins", 0) / n * 100) if n > 0 else 0
        dec_rate = (c.get("dec_wins", 0) / n * 100) if n > 0 else 0

        # Reach is in inches in the CSV — convert to cm
        # Height is already in cm
        reach_cm = None
        if not rw.empty and not pd.isna(rw["reach"].values[0]):
            raw_reach = float(rw["reach"].values[0])
            reach_cm = round(raw_reach * 2.54) if raw_reach <= 100 else round(raw_reach)

        height_cm = None
        if not rw.empty and not pd.isna(rw["height"].values[0]):
            height_cm = round(float(rw["height"].values[0]), 1)

        age = None
        if not rw.empty and not pd.isna(rw["age"].values[0]):
            age = round(float(rw["age"].values[0]), 1)

        return {
            "record":    f"{wins}W-{losses}L",
            "mu":        round(mu, 3),
            "sigma":     round(sig, 3),
            "ko_rate":   round(ko_rate,  1),
            "sub_rate":  round(sub_rate, 1),
            "dec_rate":  round(dec_rate, 1),
            "reach":     reach_cm,
            "height":    height_cm,
            "age":       age,
            "stats": [
                {"val": str(wins),              "lbl": "Wins"},
                {"val": f"{ko_rate:.0f}%",      "lbl": "KO Rate"},
                {"val": f"{sub_rate:.0f}%",     "lbl": "Sub Rate"},
            ]
        }

    def predict(self, fighter_a, fighter_b, weight_class=None):
        """
        Main prediction endpoint.

        Returns dict compatible with bayes_ai.html's renderResults():
          winA, winB            - overall win % (0-100, rounded 1dp)
          koA, subA, decA       - Fighter A method %  (if A wins)
          koB, subB, decB       - Fighter B method %  (if B wins)
          koRoundsA/B           - list of 5 round probs for KO
          subRoundsA/B          - list of 5 round probs for Sub
          skillA, skillB        - μ skill values
          sigmaA, sigmaB        - σ uncertainty
          recordA, recordB      - fight record strings
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        p_A  = self._predict_win_prob(fighter_a, fighter_b, weight_class)
        p_B  = 1.0 - p_A
        mA   = self._predict_methods(fighter_a, fighter_b, weight_class)
        mB   = self._predict_methods(fighter_b, fighter_a, weight_class)

        mu_A, sig_A = self._get_skill_info(fighter_a)
        mu_B, sig_B = self._get_skill_info(fighter_b)

        rounds = 5

        def r1(v): return round(v * 100, 1)

        return {
            "winA": r1(p_A),
            "winB": r1(p_B),
            # Method probs: conditioned on that fighter winning → multiply by win prob
            "koA":  r1(mA["KO/TKO"]     * p_A),
            "subA": r1(mA["Submission"]  * p_A),
            "decA": r1(mA["Decision"]    * p_A),
            "koB":  r1(mB["KO/TKO"]     * p_B),
            "subB": r1(mB["Submission"]  * p_B),
            "decB": r1(mB["Decision"]    * p_B),
            # Round-by-round breakdown
            "koRoundsA":  self._round_breakdown(mA["KO/TKO"],    p_A, rounds),
            "subRoundsA": self._round_breakdown(mA["Submission"], p_A, rounds),
            "koRoundsB":  self._round_breakdown(mB["KO/TKO"],    p_B, rounds),
            "subRoundsB": self._round_breakdown(mB["Submission"], p_B, rounds),
            # Fighter info
            "skillA": round(mu_A, 3),
            "skillB": round(mu_B, 3),
            "sigmaA": round(sig_A, 3),
            "sigmaB": round(sig_B, 3),
            "recordA": self._get_record(fighter_a),
            "recordB": self._get_record(fighter_b),
        }


# ══════════════════════════════════════════════════════════════
# SINGLETON — loaded once, reused across all Flask requests
# ══════════════════════════════════════════════════════════════
_model_instance = None

def get_model(fights_path="data/fight_data.csv",
              fighters_path="data/ufc_fighters_clean.csv"):
    global _model_instance
    if _model_instance is None:
        m = UFCBayesModel()
        m.load_data(fights_path, fighters_path)
        m.train()
        _model_instance = m
    return _model_instance
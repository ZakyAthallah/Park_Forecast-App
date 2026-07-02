# =============================================================
# SISTEM PREDIKSI PARKIRAN - VSCode / Desktop Edition
# Versi 2.0 — GUI Tkinter (tidak butuh browser/Flask)
#
# CARA JALANKAN:
#   pip install -r requirements.txt
#   python parking_forecast_legacy.py
# =============================================================

import os
import sys
import shutil
import sqlite3
import warnings
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================
# KONFIGURASI KOLOM — sesuaikan dengan nama kolom CSV Anda
# =============================================================
DATA_DIR     = Path(__file__).parent / "data"
DB_FILE      = DATA_DIR / "parkiran.db"
TRAINING_CSV = DATA_DIR / "Dataset.csv"

DATE_COL      = "Tanggal"
DAY_COL       = "Hari"
CONDITION_COL = "Kondisi Parkiran"
TARGET_COL    = "Jumlah Total"
STATUS_COL    = "Status Minggu"

COL_MOBIL     = "Jumlah Roda 4"
COL_MOTOR     = "Jumlah Roda 2"
KAPASITAS_MAX = 5000

# =============================================================
# TEMA WARNA
# =============================================================
C = {
    "bg":      "#c2c8ce",
    "panel":   "#b5bcc4",
    "card":    "#a2acb8",
    "border":  "#768698",
    "gold":    "#b45309",
    "green":   "#15803d",
    "teal":    "#0e7490",
    "steel":   "#475569",
    "blue":    "#1d4ed8",
    "white":   "#e8ecf0",
    "gray":    "#505d70",
    "red":     "#dc2626",
    "yellow":  "#ca8a04",
    "lime":    "#16a34a",
    "orange":  "#ea580c",
    "purple":  "#7c3aed",
    "text":    "#1e293b",
}

STATUS_COLOR = {
    "PENUH":        "#dc2626",
    "SANGAT RAMAI": "#ea580c",
    "RAMAI":        "#ca8a04",
    "TERSEDIA":     "#16a34a",
    "SEPI":         "#16a34a",
    "KOSONG":       "#94a3b8",
}

MODEL_COLOR = {
    "XGBoost": "#f97316",
    "Aktual":  "#3b82f6",
}

AVAILABLE_MODELS = ["XGBoost"]
MODEL_OPTIONS = ["Gabungan"] + AVAILABLE_MODELS

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def get_condition_label(total, kapasitas):
    if total <= 0:     return "Kosong"
    if kapasitas <= 0: return "Sepi"
    r = total / kapasitas
    if r >= 0.75:  return "Sangat Ramai"
    if r >= 0.40:  return "Ramai"
    return "Sepi"

def classify_parking_conditions(df):
    data = df.copy()
    data["_dow"] = pd.to_datetime(data[DATE_COL]).dt.dayofweek

    def ratio_label(total):
        r = total / KAPASITAS_MAX
        if r >= 0.75: return "Sangat Ramai"
        if r >= 0.40: return "Ramai"
        return "Sepi"

    y = data[TARGET_COL].apply(ratio_label)

    feat_cols = [TARGET_COL, "_dow"]
    if COL_MOBIL in data.columns: feat_cols.append(COL_MOBIL)
    if COL_MOTOR in data.columns: feat_cols.append(COL_MOTOR)

    X = data[feat_cols].fillna(0)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=5)
    clf.fit(X, y)

    return clf.predict(X), clf, feat_cols

def mape_score(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    mask = yt != 0
    return np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100 if mask.sum() else np.nan

def evaluate(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mape_score(y_true, y_pred),
    )

HARI_ID = {0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",4:"Jumat",5:"Sabtu",6:"Minggu"}

# =============================================================
# FORECASTING FUNCTIONS
# =============================================================
def split_tt(data, ratio=0.2):
    data = data.sort_values(DATE_COL).reset_index(drop=True)
    n = max(1, int(len(data) * ratio))
    return data.iloc[:-n].copy(), data.iloc[-n:].copy()



def xgb_features(data):
    data = data.copy().sort_values(DATE_COL).reset_index(drop=True)
    data["dayofweek"] = data[DATE_COL].dt.dayofweek
    data["hari_code"] = data[DAY_COL].astype("category").cat.codes if DAY_COL in data.columns else 0
    data["status_code"] = (data[STATUS_COL].astype("category").cat.codes
                           if STATUS_COL in data.columns else 0)
    data["lag_1"] = data[TARGET_COL].shift(1)
    data["roll3"] = data[TARGET_COL].shift(1).rolling(3, min_periods=1).mean()
    if DAY_COL in data.columns and STATUS_COL in data.columns:
        data["last_ds"] = data.groupby([DAY_COL, STATUS_COL])[TARGET_COL].shift(1)
        data["last_s"]  = data.groupby(STATUS_COL)[TARGET_COL].shift(1)
    else:
        data["last_ds"] = data[TARGET_COL].shift(1)
        data["last_s"]  = data[TARGET_COL].shift(1)
    lag1 = data[TARGET_COL].shift(1)
    data["last_ds"] = data["last_ds"].fillna(data["last_s"]).fillna(lag1)
    data["last_s"]  = data["last_s"].fillna(lag1)
    return data

XGB_FEAT = ["dayofweek", "hari_code", "status_code", "lag_1", "roll3", "last_ds", "last_s"]

def xgb_forecast_fn(train, test):
    full = xgb_features(pd.concat([train, test], ignore_index=True))
    lo, hi = train[TARGET_COL].min(), train[TARGET_COL].max()
    n_tr, preds, model = len(train), [], None
    for i in range(len(test)):
        idx = n_tr + i
        if model is None or i % 5 == 0:
            sl = full.iloc[:idx].dropna(subset=XGB_FEAT + [TARGET_COL])
            if len(sl) >= 20:
                model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                     subsample=0.9, colsample_bytree=0.9,
                                     random_state=42, objective="reg:squarederror", verbosity=0)
                model.fit(sl[XGB_FEAT], sl[TARGET_COL])
            else:
                model = None
        row = full.iloc[[idx]]
        p = row["last_ds"].iloc[0] if model is None or row[XGB_FEAT].isna().sum().sum() > 0 \
            else model.predict(row[XGB_FEAT])[0]
        preds.append(float(np.clip(p, lo, hi)))
    return preds



def xgb_train_predict(train):
    """In-sample XGBoost predictions on the training set."""
    data = xgb_features(train.copy())
    lo, hi = train[TARGET_COL].min(), train[TARGET_COL].max()
    valid = data.dropna(subset=XGB_FEAT + [TARGET_COL])
    if len(valid) < 20 or XGBRegressor is None:
        return None, None
    try:
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=42, objective="reg:squarederror", verbosity=0)
        model.fit(valid[XGB_FEAT], valid[TARGET_COL])
        preds = np.clip(model.predict(valid[XGB_FEAT]), lo, hi)
        return valid[TARGET_COL].values, preds
    except Exception:
        return None, None


def _xgb_run_col(split_dict, target_col):
    """Run XGBoost for any numeric target column.
    Returns (eval_df, pred_by_cond) where pred_by_cond = {cond: DataFrame[DATE_COL,'Aktual','XGBoost']}.
    """
    _F = ["_dow", "_hari", "_stat", "_lag1", "_r3", "_lds", "_ls"]

    def _featurize(data):
        d = data.copy().sort_values(DATE_COL).reset_index(drop=True)
        d["_dow"]  = d[DATE_COL].dt.dayofweek
        d["_hari"] = d[DAY_COL].astype("category").cat.codes if DAY_COL in d.columns else 0
        d["_stat"] = d[STATUS_COL].astype("category").cat.codes if STATUS_COL in d.columns else 0
        lag1 = d[target_col].shift(1)
        d["_lag1"] = lag1
        d["_r3"]   = lag1.rolling(3, min_periods=1).mean()
        if DAY_COL in d.columns and STATUS_COL in d.columns:
            d["_lds"] = d.groupby([DAY_COL, STATUS_COL])[target_col].shift(1)
            d["_ls"]  = d.groupby(STATUS_COL)[target_col].shift(1)
        else:
            d["_lds"] = lag1
            d["_ls"]  = lag1
        d["_lds"] = d["_lds"].fillna(d["_ls"]).fillna(lag1)
        d["_ls"]  = d["_ls"].fillna(lag1)
        return d

    _EMPTY_E = pd.DataFrame(columns=["Kondisi Parkiran", "Data Test", "MAE", "RMSE", "MAPE (%)"])
    erows, all_a, all_p, pred_by_cond = [], [], [], {}
    for cond, (tr, te) in split_dict.items():
        if target_col not in tr.columns or target_col not in te.columns:
            continue
        full = _featurize(pd.concat([tr, te], ignore_index=True))
        lo, hi = float(tr[target_col].min()), float(tr[target_col].max())
        n_tr, preds, mdl = len(tr), [], None
        for i in range(len(te)):
            idx = n_tr + i
            if mdl is None or i % 5 == 0:
                sl = full.iloc[:idx].dropna(subset=_F + [target_col])
                if len(sl) >= 20:
                    mdl = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                       subsample=0.9, colsample_bytree=0.9,
                                       random_state=42, objective="reg:squarederror", verbosity=0)
                    mdl.fit(sl[_F], sl[target_col])
                else:
                    mdl = None
            row = full.iloc[[idx]]
            p = row["_lds"].iloc[0] if mdl is None or row[_F].isna().sum().sum() > 0 \
                else float(mdl.predict(row[_F])[0])
            preds.append(float(np.clip(p, lo, hi)))

        # Store full prediction series for chart
        rdf_pred = te[[DATE_COL]].copy()
        rdf_pred["Aktual"]  = te[target_col].values
        rdf_pred["XGBoost"] = preds
        pred_by_cond[cond]  = rdf_pred

        act = te[target_col].values
        valid = ~np.isnan(act.astype(float))
        if valid.sum() == 0: continue
        a, pv = act[valid], np.array(preds)[valid]
        mae_v, rmse_v, mape_v = evaluate(a, pv)
        erows.append({"Kondisi Parkiran": cond, "Data Test": int(valid.sum()),
                      "MAE": round(mae_v, 2), "RMSE": round(rmse_v, 2), "MAPE (%)": round(mape_v, 2)})
        all_a.extend(a.tolist()); all_p.extend(pv.tolist())

    if not erows: return _EMPTY_E, pred_by_cond
    df_e = pd.DataFrame(erows).sort_values("RMSE").reset_index(drop=True)
    if all_a:
        mae_a, rmse_a, mape_a = evaluate(all_a, all_p)
        df_e = pd.concat([df_e, pd.DataFrame([{"Kondisi Parkiran": "Semua", "Data Test": len(all_a),
                                                "MAE": round(mae_a, 2), "RMSE": round(rmse_a, 2),
                                                "MAPE (%)": round(mape_a, 2)}])], ignore_index=True)
    return df_e, pred_by_cond


def predict_next_days(df, n=2):
    today = df[DATE_COL].max()
    results = []
    for offset in range(1, n + 1):
        tgt = today + pd.Timedelta(days=offset)
        day_name = HARI_ID[tgt.dayofweek]
        sd = df[df[DAY_COL] == day_name] if DAY_COL in df.columns else df
        val = sd[TARGET_COL].mean() if len(sd) >= 1 else df[TARGET_COL].mean()
        results.append((tgt, day_name, float(val) if not np.isnan(val) else df[TARGET_COL].mean()))
    return results


# =============================================================
# DATABASE HELPERS
# =============================================================
_DB_COLS = [DATE_COL, DAY_COL, TARGET_COL, COL_MOBIL, COL_MOTOR, STATUS_COL, CONDITION_COL]

def db_ensure_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS parkiran (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            "{DATE_COL}"      TEXT NOT NULL UNIQUE,
            "{DAY_COL}"       TEXT,
            "{TARGET_COL}"    INTEGER NOT NULL,
            "{COL_MOBIL}"     INTEGER,
            "{COL_MOTOR}"     INTEGER,
            "{STATUS_COL}"    TEXT,
            "{CONDITION_COL}" TEXT
        )
    """)
    conn.commit()

def db_load(db_path):
    conn = sqlite3.connect(db_path)
    db_ensure_table(conn)
    df = pd.read_sql(f'SELECT * FROM parkiran ORDER BY "{DATE_COL}"', conn)
    conn.close()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df[DATE_COL]   = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df.dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)

def db_import_df(db_path, df):
    conn = sqlite3.connect(db_path)
    db_ensure_table(conn)
    df_copy = df.copy()
    df_copy[DATE_COL] = df_copy[DATE_COL].dt.strftime("%Y-%m-%d")
    keep = [c for c in _DB_COLS if c in df_copy.columns]
    df_copy[keep].to_sql("parkiran", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def db_insert_row(db_path, row_dict):
    conn = sqlite3.connect(db_path)
    db_ensure_table(conn)
    cols_sql   = ", ".join(f'"{k}"' for k in row_dict)
    placeholders = ", ".join("?" for _ in row_dict)
    conn.execute(
        f"INSERT OR REPLACE INTO parkiran ({cols_sql}) VALUES ({placeholders})",
        list(row_dict.values())
    )
    conn.commit()
    conn.close()


# =============================================================
# MAIN APP
# =============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistem Prediksi Parkiran v2.0")
        self.configure(bg=C["bg"])
        self.state("zoomed")
        self.minsize(1100, 700)

        self.df                   = None   # historical CSV — for model training
        self.df_live              = None   # DB 30-day window — for dashboard & lag
        self.condition_classifier = None
        self.db_path              = None
        self.results              = {}
        self.results_mobil        = {}
        self.results_motor        = {}
        self.evaluation_df        = None
        self.eval_results         = {}
        self.dt_stats             = []
        self._pred_future         = []
        self._pred_7days          = []

        self._topbar()
        self._notebook()
        self.after(100, self._auto_load_csv)

    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------
    def _topbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="🅿  SISTEM PREDIKSI PARKIRAN",
                 font=("Segoe UI", 14, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="left", padx=18, pady=12)

        specs = [
            ("💾 Simpan Hasil",    C["steel"],  self._save,                "btn_save"),
            ("➕ Tambah Data",    C["teal"],   self._show_add_data_dialog, "btn_add"),
            ("📂 Buka CSV",       C["gold"],   self._load_csv,             "btn_load"),
            ("🗄 Database",       C["purple"], self._open_database,        "btn_db"),
            ("🔄 Sync DB",        C["green"],  self._sync_from_ref_db,     "btn_sync"),
        ]
        for txt, color, cmd, attr in specs:
            b = tk.Button(bar, text=txt, font=("Segoe UI", 10, "bold"),
                          bg=color, fg=C["white"], relief="flat",
                          activebackground=color, activeforeground=C["white"],
                          padx=14, pady=6, cursor="hand2", command=cmd)
            b.pack(side="right", padx=6, pady=8)
            setattr(self, attr, b)

        self.btn_save.config(state="disabled")
        self.btn_add.config(state="disabled")

        self.lbl_file = tk.Label(bar, text="Belum ada file dipilih",
                                 font=("Segoe UI", 9), bg=C["panel"], fg=C["gray"])
        self.lbl_file.pack(side="right", padx=14)

        # Progress bar
        pb_frame = tk.Frame(self, bg=C["bg"])
        pb_frame.pack(fill="x")
        self.prog_var = tk.DoubleVar()
        s = ttk.Style(); s.theme_use("clam")
        s.configure("G.Horizontal.TProgressbar",
                    troughcolor=C["panel"], background=C["green"],
                    lightcolor=C["green"], darkcolor=C["green"])
        ttk.Progressbar(pb_frame, variable=self.prog_var, maximum=100,
                        style="G.Horizontal.TProgressbar").pack(fill="x")
        self.lbl_prog = tk.Label(pb_frame, text="Pilih file CSV untuk memulai.",
                                 font=("Segoe UI", 8), bg=C["bg"], fg=C["gray"])
        self.lbl_prog.pack(anchor="w", padx=8, pady=1)

    def _set_prog(self, msg, pct):
        self.lbl_prog.config(text=msg)
        self.prog_var.set(pct)
        self.update_idletasks()

    # --------------------------------------------------------
    # NOTEBOOK
    # --------------------------------------------------------
    def _notebook(self):
        s = ttk.Style()
        s.configure("D.TNotebook", background=C["bg"], borderwidth=0, tabmargins=0)
        s.configure("D.TNotebook.Tab", background=C["panel"], foreground=C["gray"],
                    font=("Segoe UI", 10, "bold"), padding=[16, 7], focuscolor=C["panel"])
        s.map("D.TNotebook.Tab",
              background=[("selected", C["gold"]), ("active", C["panel"])],
              foreground=[("selected", C["white"]), ("active", C["gray"])],
              focuscolor=[("selected", C["gold"]), ("", C["panel"])],
              padding=[("selected", [16, 7])])

        self.nb = ttk.Notebook(self, style="D.TNotebook")
        self.nb.pack(fill="both", expand=True)

        self.tab_dash   = tk.Frame(self.nb, bg=C["bg"])
        tab_combined    = tk.Frame(self.nb, bg=C["bg"])

        self.nb.add(self.tab_dash,  text="  📊 Dashboard  ")
        self.nb.add(tab_combined,   text="  📈 Grafik & Evaluasi  ")

        # Scrollable container for combined tab
        _cv = tk.Canvas(tab_combined, bg=C["bg"], highlightthickness=0)
        _sb = ttk.Scrollbar(tab_combined, orient="vertical", command=_cv.yview)
        scroll_fr = tk.Frame(_cv, bg=C["bg"])
        scroll_fr.bind("<Configure>",
            lambda e: _cv.configure(scrollregion=_cv.bbox("all")))
        _win = _cv.create_window((0, 0), window=scroll_fr, anchor="nw")
        _cv.configure(yscrollcommand=_sb.set)
        _cv.bind("<Configure>", lambda e: _cv.itemconfig(_win, width=e.width))
        def _on_wheel(e): _cv.yview_scroll(int(-1*(e.delta/120)), "units")
        def _bind_wheel():   _cv.bind_all("<MouseWheel>", _on_wheel)
        def _unbind_wheel(): _cv.unbind_all("<MouseWheel>")
        self.nb.bind("<<NotebookTabChanged>>",
            lambda e: _bind_wheel() if self.nb.index("current") == 1 else _unbind_wheel())
        _sb.pack(side="right", fill="y")
        _cv.pack(side="left", fill="both", expand=True)

        # Grafik section — chart tab packs directly into scroll_fr
        self.tab_chart = scroll_fr
        self._build_dash()
        self._build_chart_tab()

        # Separator between Grafik and Evaluasi
        tk.Frame(scroll_fr, bg=C["border"], height=2).pack(fill="x", padx=10, pady=(10, 0))

        # Evaluasi section
        self.tab_eval = tk.Frame(scroll_fr, bg=C["bg"])
        self.tab_eval.pack(fill="x")
        self._build_eval_tab()

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------
    def _hdr(self, parent, text, color):
        lbl = tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                       bg=color, fg=C["white"], anchor="center", pady=5)
        lbl.pack(fill="x")
        return lbl

    def _build_dash(self):
        p = self.tab_dash

        # ---- SINGLE 3-COLUMN FRAME spanning full height ----
        main = tk.Frame(p, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=(10, 10))

        # ── LEFT COLUMN: Saat Ini (top) + Besok / Lusa (bottom) ──
        left_col = tk.Frame(main, bg=C["bg"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Parkiran Saat Ini
        saat_fr = tk.Frame(left_col, bg=C["bg"])
        saat_fr.pack(fill="both", expand=True, pady=(0, 4))
        self.hdr_saat_ini = self._hdr(saat_fr, "Parkiran Saat Ini", C["gold"])
        frm_k = tk.Frame(saat_fr, bg=C["white"])
        frm_k.pack(fill="both", expand=True)
        self.lbl_kondisi = tk.Label(frm_k, text="—",
                                    font=("Segoe UI", 64, "bold"),
                                    bg=C["white"], fg=C["text"])
        self.lbl_kondisi.pack(expand=True)

        # Parkiran Besok + Lusa side-by-side (each = Saat Ini / 2)
        bl_row = tk.Frame(left_col, bg=C["bg"])
        bl_row.pack(fill="both", expand=True)

        for attr, title, hdr_attr, px in [
                ("frm_besok", "Parkiran Besok", "hdr_besok", (0, 4)),
                ("frm_lusa",  "Parkiran Lusa",  "hdr_lusa",  (4, 0))]:
            col = tk.Frame(bl_row, bg=C["bg"])
            col.pack(side="left", fill="both", expand=True, padx=px)
            setattr(self, hdr_attr, self._hdr(col, title, C["gold"]))
            frm = tk.Frame(col, bg=C["white"])
            frm.pack(fill="both", expand=True)
            setattr(self, attr, frm)
            lbl_attr = "lbl_besok" if "besok" in attr else "lbl_lusa"
            lbl = tk.Label(frm, text="—", font=("Segoe UI", 48, "bold"),
                           bg=C["white"], fg=C["text"])
            lbl.pack(expand=True)
            setattr(self, lbl_attr, lbl)

        # ── RIGHT AREA: top row (Total Terisi + Mobil/Motor) + Statistik below ──
        right_area = tk.Frame(main, bg=C["bg"])
        right_area.pack(side="left", fill="both", expand=True)

        # Top row inside right_area
        top_row = tk.Frame(right_area, bg=C["bg"])
        top_row.pack(fill="both", expand=True, pady=(0, 4))

        # Total Terisi
        mid_top = tk.Frame(top_row, bg=C["bg"])
        mid_top.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self._hdr(mid_top, "Total Terisi", C["green"])
        mid_body = tk.Frame(mid_top, bg=C["white"])
        mid_body.pack(fill="both", expand=True)

        mid_body.grid_rowconfigure(0, weight=65)
        mid_body.grid_rowconfigure(1, weight=35)
        mid_body.grid_columnconfigure(0, weight=1)

        lbl_total_fr = tk.Frame(mid_body, bg=C["white"])
        lbl_total_fr.grid(row=0, column=0, sticky="nsew")
        self.lbl_total = tk.Label(lbl_total_fr, text="—",
                                   font=("Segoe UI", 52, "bold"),
                                   bg=C["white"], fg="#111111")
        self.lbl_total.pack(expand=True)

        bot = tk.Frame(mid_body, bg=C["white"])
        bot.grid(row=1, column=0, sticky="nsew", padx=4, pady=0)

        slot_c = tk.Frame(bot, bg=C["white"],
                          highlightbackground="#5c2020",
                          highlightcolor="#5c2020",
                          highlightthickness=2)
        slot_c.pack(side="left", fill="both", expand=True, padx=(0, 4))
        tk.Label(slot_c, text="Slot Kosong", font=("Segoe UI", 9, "bold"),
                 bg="#5c2020", fg=C["white"]).pack(fill="x")
        self.lbl_kosong = tk.Label(slot_c, text="—",
                                    font=("Segoe UI", 28, "bold"),
                                    bg=C["white"], fg="#111111")
        self.lbl_kosong.pack(expand=True)

        self.donut_holder = tk.Frame(bot, bg=C["white"])
        self.donut_holder.pack(side="left", fill="both", expand=True)

        # Mobil + Motor (fixed 185 px wide, 50/50 height split)
        col_right = tk.Frame(top_row, bg=C["bg"], width=185)
        col_right.pack(side="left", fill="y")
        col_right.pack_propagate(False)

        mobil_fr = tk.Frame(col_right, bg=C["bg"])
        mobil_fr.pack(fill="both", expand=True, pady=(0, 3))
        self._hdr(mobil_fr, "Mobil", C["blue"])
        mobil_body = tk.Frame(mobil_fr, bg=C["white"])
        mobil_body.pack(fill="both", expand=True)
        self.lbl_mobil = tk.Label(mobil_body, text="—",
                                   font=("Segoe UI", 34, "bold"),
                                   bg=C["white"], fg="#111111")
        self.lbl_mobil.pack(expand=True)

        motor_fr = tk.Frame(col_right, bg=C["bg"])
        motor_fr.pack(fill="both", expand=True)
        self._hdr(motor_fr, "Motor", C["teal"])
        motor_body = tk.Frame(motor_fr, bg=C["white"])
        motor_body.pack(fill="both", expand=True)
        self.lbl_motor = tk.Label(motor_body, text="—",
                                   font=("Segoe UI", 34, "bold"),
                                   bg=C["white"], fg="#111111")
        self.lbl_motor.pack(expand=True)

        # Statistik Minggu Ini
        stat_fr = tk.Frame(right_area, bg=C["bg"])
        stat_fr.pack(fill="both", expand=True)
        self._hdr(stat_fr, "Statistik Minggu Ini", C["steel"])
        self.stat_holder = tk.Frame(stat_fr, bg=C["white"])
        self.stat_holder.pack(fill="both", expand=True)

    def _update_dash(self):
        df = self.df_live if self.df_live is not None and len(self.df_live) > 0 else self.df
        if df is None: return
        today    = df[DATE_COL].max()
        today_df = df[df[DATE_COL] == today]

        total  = int(today_df[TARGET_COL].sum())
        mobil  = int(today_df[COL_MOBIL].sum()) if COL_MOBIL in df.columns else None
        motor  = int(today_df[COL_MOTOR].sum()) if COL_MOTOR in df.columns else None
        kap    = KAPASITAS_MAX
        kosong = max(0, kap - total)
        cond   = get_condition_label(total, kap)
        color  = STATUS_COLOR.get(cond, C["text"])

        self.hdr_saat_ini.config(text=f"Parkiran Saat Ini  ({today.strftime('%m/%d/%Y')})")
        self.lbl_kondisi.config(text=cond, fg=color)
        self.lbl_total.config(text=f"{total:,}")
        self.lbl_kosong.config(text=f"{kosong:,}")
        self.lbl_mobil.config(text=f"{mobil:,}" if mobil is not None else "—")
        self.lbl_motor.config(text=f"{motor:,}" if motor is not None else "—")

        self._draw_donut(today, total, mobil, motor, kosong)
        self._draw_weekly(today)

    def _draw_donut(self, today, total, mobil, motor, kosong):
        for w in self.donut_holder.winfo_children(): w.destroy()
        fig = Figure(figsize=(2.1, 2.0), facecolor=C["white"])
        ax  = fig.add_subplot(111)
        m_n  = mobil if mobil is not None else max(1, total // 5)
        mo_n = motor if motor is not None else max(1, total - m_n)
        sizes = [m_n, mo_n, max(0, kosong)]
        cols  = ["#1d4ed8", "#0e7490", "#5c2020"]
        nz = [(s, c) for s, c in zip(sizes, cols) if s > 0]
        if nz:
            sv, cv = zip(*nz)
            ax.pie(sv, colors=cv, startangle=90,
                   wedgeprops=dict(width=0.42, edgecolor="white"))
        ax.set_title(today.strftime("%b %d"), fontsize=7, pad=2)
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(["#1d4ed8","#0e7490","#5c2020"],
                                   ["Mobil","Motor","Kosong"])]
        ax.legend(handles=patches, fontsize=6, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        fig.tight_layout(pad=0.1)
        canvas = FigureCanvasTkAgg(fig, master=self.donut_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_weekly(self, today):
        for w in self.stat_holder.winfo_children(): w.destroy()
        src = self.df_live if self.df_live is not None and len(self.df_live) > 0 else self.df
        if src is None: return
        start = today - pd.Timedelta(days=6)
        wk = src[src[DATE_COL] >= start].groupby(DATE_COL)[TARGET_COL].sum().reset_index()
        fig = Figure(figsize=(5, 2.7), facecolor=C["white"])
        ax  = fig.add_subplot(111)
        ax.plot(wk[DATE_COL], wk[TARGET_COL], marker="o", color="#1d4ed8",
                linewidth=2.5, markersize=5)
        ax.fill_between(wk[DATE_COL], wk[TARGET_COL], alpha=0.08, color="#1d4ed8")
        ax.set_title(f"Total Kendaraan Per Hari  ({start.strftime('%b %d')} – {today.strftime('%b %d')})",
                     fontsize=8)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Total Vehicles", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout(pad=0.5, rect=[0, 0, 0.96, 1])
        canvas = FigureCanvasTkAgg(fig, master=self.stat_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------
    # CHART TAB
    # --------------------------------------------------------
    def _build_chart_tab(self):
        tk.Label(self.tab_chart, text="Grafik", font=("Segoe UI", 11, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(anchor="w", padx=12, pady=(10, 2))

        ctrl = tk.Frame(self.tab_chart, bg=C["bg"])
        ctrl.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(ctrl, text="Kondisi:", font=("Segoe UI",10,"bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left", padx=(12,4))
        self.combo_cond = ttk.Combobox(ctrl, values=[], width=18, state="readonly",
                                        font=("Segoe UI",10))
        self.combo_cond.pack(side="left", padx=4)
        self.combo_cond.bind("<<ComboboxSelected>>", lambda e: self._refresh_chart())

        split_fr = tk.Frame(self.tab_chart, bg=C["bg"])
        split_fr.pack(fill="x", padx=10, pady=(0, 10))

        # Left panel — Total (main, expands horizontally only)
        left_fr = tk.Frame(split_fr, bg=C["bg"])
        left_fr.pack(side="left", fill="both", expand=True, anchor="n")
        tk.Label(left_fr, text="Jumlah Total", font=("Segoe UI",9,"bold"),
                 bg=C["bg"], fg=C["gold"]).pack(anchor="w", padx=2)
        self.chart_holder = tk.Frame(left_fr, bg=C["bg"])
        self.chart_holder.pack(anchor="nw", fill="x")

        # Right panel — Mobil + Motor stacked
        right_fr = tk.Frame(split_fr, bg=C["bg"])
        right_fr.pack(side="right", fill="y", padx=(6, 0))

        tk.Label(right_fr, text="Jumlah Roda 4 (Mobil)", font=("Segoe UI",9,"bold"),
                 bg=C["bg"], fg=C["gold"]).pack(anchor="w", padx=2)
        self.chart_holder_mobil = tk.Frame(right_fr, bg=C["white"])
        self.chart_holder_mobil.pack(fill="x")

        tk.Label(right_fr, text="Jumlah Roda 2 (Motor)", font=("Segoe UI",9,"bold"),
                 bg=C["bg"], fg=C["gold"]).pack(anchor="w", padx=2, pady=(6,0))
        self.chart_holder_motor = tk.Frame(right_fr, bg=C["white"])
        self.chart_holder_motor.pack(fill="x")

    def _draw_one_chart(self, holder, rdf, cond, title, figsize):
        """Render a single Aktual-vs-XGBoost chart into the given holder frame."""
        for w in holder.winfo_children(): w.destroy()
        if rdf is None or (hasattr(rdf, "empty") and rdf.empty): return

        rdf = rdf.copy()
        rdf[DATE_COL] = pd.to_datetime(rdf[DATE_COL])
        rdf = rdf.sort_values(DATE_COL).reset_index(drop=True)

        fig = Figure(figsize=figsize, facecolor=C["white"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#f8fafc")

        segs, cur = [], [0]
        for i in range(1, len(rdf)):
            if (rdf[DATE_COL].iloc[i] - rdf[DATE_COL].iloc[i - 1]).days > 1:
                segs.append(cur); cur = [i]
            else:
                cur.append(i)
        segs.append(cur)

        has_gaps = cond != "Semua" and len(segs) > 1

        if has_gaps:
            GAP_SIZE = 3
            positions, seg_bounds, x = [0] * len(rdf), [], 0
            for s_idx, seg in enumerate(segs):
                if s_idx > 0:
                    seg_bounds.append((x - 1, x + GAP_SIZE))
                    x += GAP_SIZE
                for row_idx in seg:
                    positions[row_idx] = x; x += 1

            xv = [positions[i] for i in range(len(rdf))]
            ax.plot(xv, rdf["Aktual"].values, label="Aktual",
                    color=MODEL_COLOR["Aktual"], linewidth=2.0,
                    marker="o", markersize=4, zorder=5)
            for m in AVAILABLE_MODELS:
                if m in rdf.columns:
                    ax.plot(xv, rdf[m].values, label=m,
                            color=MODEL_COLOR.get(m, "gray"), linewidth=1.8,
                            marker="^", markersize=5, linestyle="--", alpha=0.9, zorder=6)

            for (prev_x, next_x), nxt in zip(seg_bounds, segs[1:]):
                gx = (prev_x + next_x) / 2
                ax.axvline(gx, color="#9ca3af", linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)
                ax.text(gx, 0.82, rdf[DATE_COL].iloc[nxt[0]].strftime("%Y-%m-%d"),
                        transform=ax.get_xaxis_transform(),
                        rotation=90, va="top", ha="center", fontsize=6.5, color="#6b7280")

            n = len(rdf); step = max(1, n // 8)
            tick_idx = list(range(0, n, step))
            if tick_idx[-1] != n - 1: tick_idx.append(n - 1)
            ax.set_xticks([positions[i] for i in tick_idx])
            ax.set_xticklabels(
                [rdf[DATE_COL].iloc[i].strftime("%Y-%m-%d") for i in tick_idx],
                rotation=30, ha="right", fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
        else:
            ax.plot(rdf[DATE_COL], rdf["Aktual"], label="Aktual",
                    color=MODEL_COLOR["Aktual"], linewidth=2.0,
                    marker="o", markersize=4, zorder=5)
            for m in AVAILABLE_MODELS:
                if m in rdf.columns:
                    ax.plot(rdf[DATE_COL], rdf[m], label=m,
                            color=MODEL_COLOR.get(m, "gray"), linewidth=1.8,
                            marker="^", markersize=5, linestyle="--", alpha=0.9, zorder=6)
            fig.autofmt_xdate(rotation=30)
            ax.tick_params(labelsize=6)

        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("Tanggal", fontsize=7)
        ax.set_ylabel("Jumlah Kendaraan", fontsize=7)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(width=int(figsize[0] * 100), height=int(figsize[1] * 100))
        widget.pack()

    def _refresh_chart(self):
        if not self.results: return
        cond = self.combo_cond.get()
        if cond not in self.results and cond != "Semua": return

        def _merge(res_dict):
            if not res_dict: return None
            return pd.concat(res_dict.values(), ignore_index=True) if cond == "Semua" \
                   else res_dict.get(cond)

        self._draw_one_chart(
            self.chart_holder, _merge(self.results), cond,
            title=f"Aktual vs Prediksi  ·  Kondisi: {cond}", figsize=(12, 5.975))
        self._draw_one_chart(
            self.chart_holder_mobil, _merge(self.results_mobil), cond,
            title=f"Roda 4 (Mobil)  ·  {cond}", figsize=(5, 2.85))
        self._draw_one_chart(
            self.chart_holder_motor, _merge(self.results_motor), cond,
            title=f"Roda 2 (Motor)  ·  {cond}", figsize=(5, 2.85))

    # --------------------------------------------------------
    # EVAL TAB
    # --------------------------------------------------------
    def _build_eval_tab(self):
        s = ttk.Style()
        s.configure("Eva.Treeview", background=C["white"], foreground=C["text"],
                    fieldbackground=C["white"], font=("Segoe UI", 10), rowheight=30)
        s.configure("Eva.Treeview.Heading", background=C["steel"], foreground=C["white"],
                    font=("Segoe UI", 9, "bold"), padding=[4, 6])
        s.map("Eva.Treeview", background=[], foreground=[])

        outer = tk.Frame(self.tab_eval, bg=C["bg"])
        outer.pack(fill="both", expand=True, padx=14, pady=10)

        # --- Hasil Evaluasi ---
        tk.Label(outer, text="Hasil Evaluasi",
                 font=("Segoe UI", 11, "bold"), bg=C["bg"], fg=C["text"]
                 ).pack(anchor="w", pady=(0, 6))

        cols = ("Kondisi Parkiran", "Data Test", "MAE", "RMSE", "MAPE (%)")
        col_w = [180, 80, 80, 80, 80]

        fr_tables = tk.Frame(outer, bg=C["bg"])
        fr_tables.pack(fill="x", pady=(0, 2))

        for attr, label in [("tree_total", "Jumlah Total"),
                             ("tree_mobil", "Jumlah Roda 4 (Mobil)"),
                             ("tree_motor", "Jumlah Roda 2 (Motor)")]:
            pnl = tk.Frame(fr_tables, bg=C["bg"])
            pnl.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tk.Label(pnl, text=label, font=("Segoe UI", 9, "bold"),
                     bg=C["bg"], fg=C["gold"]).pack(anchor="w", pady=(0, 3))
            tree = ttk.Treeview(pnl, columns=cols, show="headings",
                                height=4, style="Eva.Treeview", selectmode="none",
                                takefocus=False)
            for c, w in zip(cols, col_w):
                tree.heading(c, text=c)
                tree.column(c, width=w, anchor="center")
            tree.tag_configure("odd",  background=C["white"])
            tree.tag_configure("even", background=C["panel"])
            tree.bind("<Motion>",       lambda e: "break")
            tree.bind("<ButtonPress-1>",lambda e: "break")
            tree.pack(fill="x")
            setattr(self, attr, tree)

        self.tree = self.tree_total

        tk.Label(outer, text="★ Diurutkan berdasarkan RMSE terkecil per kondisi",
                 font=("Segoe UI", 8, "italic"), bg=C["bg"], fg=C["gray"]
                 ).pack(anchor="w", pady=(4, 0))

        # Separator
        tk.Frame(outer, bg=C["border"], height=1).pack(fill="x", pady=12)

        # --- Klasifikasi Decision Tree ---
        tk.Label(outer, text="Klasifikasi",
                 font=("Segoe UI", 11, "bold"), bg=C["bg"], fg=C["text"]
                 ).pack(anchor="w", pady=(0, 6))

        fr2 = tk.Frame(outer, bg=C["bg"])
        fr2.pack(fill="x", pady=(0, 4))

        dt_cols  = ("Kondisi", "Total Baris", "Min", "Maks", "Rata-rata")
        dt_col_w = [150, 100, 100, 100, 100]

        for attr, label in [("dt_tree_total", "Jumlah Total"),
                             ("dt_tree_mobil", "Jumlah Roda 4 (Mobil)"),
                             ("dt_tree_motor", "Jumlah Roda 2 (Motor)")]:
            pnl = tk.Frame(fr2, bg=C["bg"])
            pnl.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tk.Label(pnl, text=label, font=("Segoe UI", 9, "bold"),
                     bg=C["bg"], fg=C["gold"]).pack(anchor="w", pady=(0, 3))
            tree = ttk.Treeview(pnl, columns=dt_cols, show="headings",
                                height=3, style="Eva.Treeview", selectmode="none",
                                takefocus=False)
            for c, w in zip(dt_cols, dt_col_w):
                tree.heading(c, text=c)
                tree.column(c, width=w, anchor="center")
            tree.tag_configure("odd",  background=C["white"])
            tree.tag_configure("even", background=C["panel"])
            tree.bind("<Motion>",       lambda e: "break")
            tree.bind("<ButtonPress-1>",lambda e: "break")
            tree.pack(fill="x")
            setattr(self, attr, tree)

        tk.Label(outer,
                 text="↑ Rentang nilai yang diklasifikasikan oleh Decision Tree per kondisi",
                 font=("Segoe UI", 8, "italic"), bg=C["bg"], fg=C["gray"]
                 ).pack(anchor="w", pady=(4, 0))

        # Separator
        tk.Frame(outer, bg=C["border"], height=1).pack(fill="x", pady=12)

        # --- Prediksi 7 Hari ke Depan ---
        tk.Label(outer, text="Prediksi 7 Hari ke Depan",
                 font=("Segoe UI", 11, "bold"), bg=C["bg"], fg=C["text"]
                 ).pack(anchor="w", pady=(0, 6))

        pred7_cols  = ("Tanggal", "Hari", "Jumlah Total", "Roda 4 (Mobil)", "Roda 2 (Motor)", "Kondisi", "Sumber")
        pred7_col_w = [110, 80, 110, 110, 110, 110, 90]

        fr_pred7 = tk.Frame(outer, bg=C["bg"])
        fr_pred7.pack(fill="x", pady=(0, 4))

        self.tree_pred7 = ttk.Treeview(fr_pred7, columns=pred7_cols, show="headings",
                                        height=7, style="Eva.Treeview",
                                        selectmode="none", takefocus=False)
        for c, w in zip(pred7_cols, pred7_col_w):
            self.tree_pred7.heading(c, text=c)
            self.tree_pred7.column(c, width=w, anchor="center")
        self.tree_pred7.tag_configure("odd",    background=C["white"])
        self.tree_pred7.tag_configure("even",   background=C["panel"])
        self.tree_pred7.tag_configure("actual", background="#d4edda", foreground="#155724")
        self.tree_pred7.bind("<Motion>",        lambda e: "break")
        self.tree_pred7.bind("<ButtonPress-1>", lambda e: "break")

        sb7 = ttk.Scrollbar(fr_pred7, orient="vertical", command=self.tree_pred7.yview)
        self.tree_pred7.configure(yscrollcommand=sb7.set)
        self.tree_pred7.pack(side="left", fill="x", expand=True)
        sb7.pack(side="left", fill="y")

        tk.Label(outer,
                 text="★ Hijau = data aktual dari DB  |  Putih/Abu = hasil prediksi XGBoost",
                 font=("Segoe UI", 8, "italic"), bg=C["bg"], fg=C["gray"]
                 ).pack(anchor="w", pady=(4, 0))

    def _refresh_pred7_table(self):
        if not hasattr(self, "tree_pred7") or not self._pred_7days:
            return
        t = self.tree_pred7
        for r in t.get_children():
            t.delete(r)
        for i, (tgt_date, day_name, total, mobil, motor, cond, is_actual) in enumerate(self._pred_7days):
            tag = "actual" if is_actual else ("even" if i % 2 else "odd")
            src = "Aktual" if is_actual else "Prediksi"
            m_str  = f"{mobil:,}" if mobil is not None else "-"
            mo_str = f"{motor:,}" if motor is not None else "-"
            vals = (tgt_date.strftime("%d/%m/%Y"), day_name,
                    f"{total:,}", m_str, mo_str, cond, src)
            t.insert("", "end", values=vals, tags=(tag,))

    def _refresh_file_label(self):
        csv_part = (f"CSV: {os.path.basename(str(TRAINING_CSV))}  ({len(self.df)} baris)"
                    if self.df is not None else "CSV: belum dimuat")
        db_part  = (f"DB: {len(self.df_live)} baris  ({self.df_live[DATE_COL].max().strftime('%d/%m/%Y')})"
                    if self.df_live is not None and len(self.df_live) > 0
                    else ("DB: " + os.path.basename(self.db_path) + " (kosong)"
                          if self.db_path else "DB: belum terhubung"))
        self.lbl_file.config(text=f"{csv_part}   |   {db_part}")

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------
    def _open_database(self):
        path = filedialog.askopenfilename(
            title="Buka Database Parkiran (30-day window)",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            initialdir=str(DATA_DIR) if DATA_DIR.is_dir() else ".")
        if not path:
            return
        self._load_db_path(path, auto_run=False)

    def _load_db_path(self, path, auto_run=False):
        try:
            df = db_load(path)
            self.db_path = path
            n_rows = len(df)
            self.btn_add.config(state="normal")
            if n_rows == 0:
                self._refresh_file_label()
                self._set_prog(
                    "Database kosong. Tambah data dengan tombol + Tambah Data.", 0)
                return
            self.df_live = df.sort_values(DATE_COL).reset_index(drop=True)
            self._refresh_file_label()
            self._update_dash()
            if auto_run and self.df is not None:
                self._set_prog("Data dimuat. Memulai forecasting otomatis...", 0)
                self.after(200, self._run_thread)
            else:
                self._set_prog("Database terhubung.", 0)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuka database:\n{e}", parent=self)

    def _sync_from_ref_db(self):
        # Auto-detect Data Parkiran.db di folder data/, atau buka file dialog
        ref_default = DATA_DIR / "Data Parkiran.db"
        if ref_default.exists():
            src_path = str(ref_default)
        else:
            src_path = filedialog.askopenfilename(
                title="Pilih Database Referensi (sumber 56 data terbaru)",
                filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
                initialdir=str(DATA_DIR) if DATA_DIR.is_dir() else ".")
            if not src_path:
                return

        try:
            conn_src = sqlite3.connect(src_path)
            df_src = pd.read_sql(
                f'SELECT * FROM parkiran ORDER BY "{DATE_COL}" DESC LIMIT 56',
                conn_src)
            conn_src.close()

            if df_src.empty:
                messagebox.showwarning("Sync DB", "Database referensi kosong.", parent=self)
                return

            # Urutkan ascending lagi setelah ambil 56 terbaru
            if "id" in df_src.columns:
                df_src = df_src.drop(columns=["id"])
            df_src[DATE_COL] = pd.to_datetime(df_src[DATE_COL], errors="coerce")
            df_src[TARGET_COL] = pd.to_numeric(df_src[TARGET_COL], errors="coerce")
            df_src = df_src.dropna(subset=[DATE_COL, TARGET_COL])
            df_src = df_src.sort_values(DATE_COL).reset_index(drop=True)

            d_min = df_src[DATE_COL].min().strftime("%d/%m/%Y")
            d_max = df_src[DATE_COL].max().strftime("%d/%m/%Y")
            n = len(df_src)

            ok = messagebox.askyesno(
                "Konfirmasi Sync DB",
                f"Sumber  : {Path(src_path).name}\n"
                f"Data    : {n} baris ({d_min} s/d {d_max})\n\n"
                f"Ini akan MENGGANTI seluruh isi parkiran.db.\n"
                f"Lanjutkan?",
                parent=self)
            if not ok:
                return

            # Tulis ke parkiran.db (live window)
            target_path = self.db_path if self.db_path else str(DB_FILE)
            db_import_df(target_path, df_src)

            # Reload
            self.db_path = target_path
            self._load_db_path(target_path, auto_run=self.df is not None)
            self._set_prog(f"Sync selesai: {n} baris ({d_min} s/d {d_max}) dimuat ke window.", 0)

        except Exception as e:
            messagebox.showerror("Sync DB Error", f"Gagal sync:\n{e}", parent=self)

    def _show_add_data_dialog(self):
        if not self.db_path:
            messagebox.showwarning("Peringatan",
                                   "Buka database terlebih dahulu menggunakan tombol 🗄 Database.",
                                   parent=self)
            return

        dlg = tk.Toplevel(self)
        dlg.title("Tambah Data Parkiran")
        dlg.configure(bg=C["bg"])
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("400x270")

        fields = {}

        def _row(label, key, default="", options=None):
            fr = tk.Frame(dlg, bg=C["bg"])
            fr.pack(fill="x", padx=24, pady=5)
            tk.Label(fr, text=label, width=20, anchor="w",
                     font=("Segoe UI", 10), bg=C["bg"], fg=C["text"]).pack(side="left")
            var = tk.StringVar(value=default)
            if options:
                cb = ttk.Combobox(fr, textvariable=var, values=options,
                                  width=16, state="readonly", font=("Segoe UI", 10))
                cb.pack(side="left")
            else:
                ttk.Entry(fr, textvariable=var, width=18,
                          font=("Segoe UI", 10)).pack(side="left")
            fields[key] = var

        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        _row("Tanggal (YYYY-MM-DD)", DATE_COL, today_str)
        _row("Jumlah Roda 4 (Mobil)", COL_MOBIL)
        _row("Jumlah Roda 2 (Motor)", COL_MOTOR)
        _row("Status Minggu", STATUS_COL, "Perkuliahan", ["Libur", "Libur Akademik", "Perkuliahan", "Ujian"])

        def on_ok():
            try:
                tgl = pd.to_datetime(fields[DATE_COL].get())
            except Exception:
                messagebox.showerror("Input Error", "Format tanggal tidak valid.", parent=dlg)
                return

            mobil_str = fields[COL_MOBIL].get().strip().replace(",", "").replace(".", "")
            motor_str = fields[COL_MOTOR].get().strip().replace(",", "").replace(".", "")
            if not mobil_str.isdigit() or not motor_str.isdigit():
                messagebox.showerror("Input Error",
                                     "Jumlah Roda 4 dan Roda 2 harus diisi dengan angka.",
                                     parent=dlg)
                return

            mobil = int(mobil_str)
            motor = int(motor_str)
            total = mobil + motor

            row_dict = {
                DATE_COL:   tgl.strftime("%Y-%m-%d"),
                DAY_COL:    HARI_ID[tgl.dayofweek],
                TARGET_COL: total,
                COL_MOBIL:  mobil,
                COL_MOTOR:  motor,
                STATUS_COL: fields[STATUS_COL].get(),
            }

            db_insert_row(self.db_path, row_dict)
            # Sliding window: keep exactly 56 rows (8 minggu)
            conn = sqlite3.connect(self.db_path)
            conn.execute(f'''
                DELETE FROM parkiran WHERE "{DATE_COL}" NOT IN (
                    SELECT "{DATE_COL}" FROM parkiran
                    ORDER BY "{DATE_COL}" DESC LIMIT 56
                )
            ''')
            conn.commit()
            conn.close()
            dlg.destroy()
            self._load_db_path(self.db_path, auto_run=True)
            messagebox.showinfo("Sukses",
                                f"Data {row_dict[DATE_COL]} berhasil ditambahkan.\n"
                                f"Jumlah Total: {total:,} (Mobil: {mobil:,} + Motor: {motor:,})\n"
                                "Forecasting diperbarui otomatis.",
                                parent=self)

        tk.Frame(dlg, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)
        btn_fr = tk.Frame(dlg, bg=C["bg"])
        btn_fr.pack(pady=(0, 14))
        tk.Button(btn_fr, text="  Simpan  ", bg=C["green"], fg=C["white"],
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  cursor="hand2", padx=10, pady=5,
                  command=on_ok).pack(side="left", padx=8)
        tk.Button(btn_fr, text="  Batal  ", bg=C["gray"], fg=C["white"],
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  cursor="hand2", padx=10, pady=5,
                  command=dlg.destroy).pack(side="left", padx=8)

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------
    def _auto_load_csv(self):
        """On startup: load training CSV, then connect live DB."""
        if not DATA_DIR.is_dir():
            return
        # 1. Load historical CSV for model training
        if TRAINING_CSV.exists():
            self._load_csv_path(str(TRAINING_CSV), auto_run=False)
        else:
            csvs = sorted(
                [p for p in DATA_DIR.glob("*.csv")],
                key=lambda p: p.stat().st_size, reverse=True)
            if csvs:
                self._load_csv_path(str(csvs[0]), auto_run=False)
        # 2. Connect live DB window (triggers auto-run if CSV also loaded)
        if DB_FILE.exists():
            self._load_db_path(str(DB_FILE), auto_run=self.df is not None)

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Pilih File CSV Dataset Parkiran",
            filetypes=[("CSV Files","*.csv"),("All","*.*")])
        if not path: return
        self._load_csv_path(path)

    def _load_csv_path(self, path, auto_run=False):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df[DATE_COL]   = pd.to_datetime(df[DATE_COL], errors="coerce")
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
            df = df.dropna(subset=[DATE_COL, TARGET_COL])
            df = df.sort_values(DATE_COL).reset_index(drop=True)
            # Auto-classify each row into Sepi / Ramai / Sangat Ramai via Decision Tree
            conditions, clf, _ = classify_parking_conditions(df)
            df[CONDITION_COL] = conditions
            self.df = df
            self.condition_classifier = clf
            self._refresh_file_label()
            if self.df_live is None:
                self._update_dash()
            if auto_run:
                self._set_prog("File dimuat. Memulai forecasting otomatis...", 0)
                self.after(200, self._run_thread)
            else:
                self._set_prog("File dimuat.", 0)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca CSV:\n{e}")

    # --------------------------------------------------------
    # RUN FORECASTING
    # --------------------------------------------------------
    def _predict_next_from_live(self):
        """
        Predict today, besok, lusa using XGBoost (trained on CSV) + lag from DB.
        Returns list of 3 tuples:
            (date, day_name, total, mobil, motor, cond, is_actual)
        """
        df_hist    = self.df
        df_live    = self.df_live
        src        = df_live if df_live is not None and len(df_live) > 0 else df_hist
        today_real = pd.Timestamp.today().normalize()

        lo = float(df_hist[TARGET_COL].min())
        hi = float(df_hist[TARGET_COL].max())

        # Train XGBoost on full historical CSV
        full_feat = xgb_features(df_hist.copy())
        valid     = full_feat.dropna(subset=XGB_FEAT + [TARGET_COL])
        model     = None
        if len(valid) >= 20 and XGBRegressor is not None:
            model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=42, objective="reg:squarederror", verbosity=0)
            model.fit(valid[XGB_FEAT], valid[TARGET_COL])

        mobil_ratio = (df_hist[COL_MOBIL].sum() / df_hist[TARGET_COL].sum()) \
                      if COL_MOBIL in df_hist.columns and df_hist[TARGET_COL].sum() > 0 else None
        motor_ratio = (df_hist[COL_MOTOR].sum() / df_hist[TARGET_COL].sum()) \
                      if COL_MOTOR in df_hist.columns and df_hist[TARGET_COL].sum() > 0 else None

        hari_cats = sorted(df_hist[DAY_COL].unique().tolist()) if DAY_COL in df_hist.columns else []
        all_stats = set()
        for d in [df_hist, df_live]:
            if d is not None and STATUS_COL in d.columns:
                all_stats.update(d[STATUS_COL].dropna().unique().tolist())
        stat_cats   = sorted(all_stats)
        last_status = src[STATUS_COL].iloc[-1] if STATUS_COL in src.columns else None

        results = []
        running = src.copy()

        for offset in range(3):  # 0=today, 1=besok, 2=lusa
            tgt_date = today_real + pd.Timedelta(days=offset)
            day_name = HARI_ID[tgt_date.dayofweek]

            # Use actual DB data if it exists for this date
            if df_live is not None:
                existing = df_live[df_live[DATE_COL] == tgt_date]
                if not existing.empty:
                    row   = existing.iloc[0]
                    total = int(row[TARGET_COL])
                    mobil = int(row[COL_MOBIL]) if COL_MOBIL in df_live.columns else None
                    motor = int(row[COL_MOTOR]) if COL_MOTOR in df_live.columns else None
                    cond  = get_condition_label(total, KAPASITAS_MAX)
                    results.append((tgt_date, day_name, total, mobil, motor, cond, True))
                    continue

            # Build lag features from running tail (DB + any prior predictions)
            prev_vals = running[TARGET_COL].values
            lag1  = float(prev_vals[-1])
            roll3 = float(np.mean(prev_vals[-3:])) if len(prev_vals) >= 3 else lag1

            h_code = hari_cats.index(day_name) if day_name in hari_cats else 0
            s_code = stat_cats.index(last_status) if last_status in stat_cats else 0

            same_dow = running[DATE_COL].dt.dayofweek == int(tgt_date.dayofweek)
            lds = float(running[same_dow][TARGET_COL].iloc[-1]) if same_dow.sum() > 0 else lag1

            feat_row = pd.DataFrame([{
                "dayofweek": int(tgt_date.dayofweek),
                "hari_code": h_code, "status_code": s_code,
                "lag_1": lag1, "roll3": roll3, "last_ds": lds, "last_s": lag1,
            }])

            xgb_pred = float(np.clip(model.predict(feat_row[XGB_FEAT])[0], lo, hi)) \
                       if model is not None else lag1

            # 50/50 blend with historical day-of-week median
            same_dow_hist = df_hist[DATE_COL].dt.dayofweek == int(tgt_date.dayofweek)
            day_med = float(df_hist[same_dow_hist][TARGET_COL].median()) \
                      if same_dow_hist.sum() >= 3 else float(df_hist[TARGET_COL].median())
            pred_total = float(np.clip(0.5 * xgb_pred + 0.5 * day_med, lo, hi))

            pred_mobil = int(round(pred_total * mobil_ratio)) if mobil_ratio is not None else None
            pred_motor = int(round(pred_total * motor_ratio)) if motor_ratio is not None else None
            cond       = get_condition_label(int(pred_total), KAPASITAS_MAX)

            results.append((tgt_date, day_name, int(pred_total), pred_mobil, pred_motor, cond, False))

            new_row = {DATE_COL: tgt_date, TARGET_COL: pred_total}
            if DAY_COL    in running.columns: new_row[DAY_COL]    = day_name
            if STATUS_COL in running.columns: new_row[STATUS_COL] = last_status
            running = pd.concat([running, pd.DataFrame([new_row])], ignore_index=True)

        return results

    def _predict_7days(self):
        """Predict 7 hari ke depan (for bottom table). Same logic as _predict_next_from_live but 7 steps."""
        df_hist    = self.df
        df_live    = self.df_live
        src        = df_live if df_live is not None and len(df_live) > 0 else df_hist
        today_real = pd.Timestamp.today().normalize()

        lo = float(df_hist[TARGET_COL].min())
        hi = float(df_hist[TARGET_COL].max())

        full_feat = xgb_features(df_hist.copy())
        valid     = full_feat.dropna(subset=XGB_FEAT + [TARGET_COL])
        model     = None
        if len(valid) >= 20 and XGBRegressor is not None:
            model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=42, objective="reg:squarederror", verbosity=0)
            model.fit(valid[XGB_FEAT], valid[TARGET_COL])

        mobil_ratio = (df_hist[COL_MOBIL].sum() / df_hist[TARGET_COL].sum()) \
                      if COL_MOBIL in df_hist.columns and df_hist[TARGET_COL].sum() > 0 else None
        motor_ratio = (df_hist[COL_MOTOR].sum() / df_hist[TARGET_COL].sum()) \
                      if COL_MOTOR in df_hist.columns and df_hist[TARGET_COL].sum() > 0 else None

        hari_cats = sorted(df_hist[DAY_COL].unique().tolist()) if DAY_COL in df_hist.columns else []
        all_stats = set()
        for d in [df_hist, df_live]:
            if d is not None and STATUS_COL in d.columns:
                all_stats.update(d[STATUS_COL].dropna().unique().tolist())
        stat_cats   = sorted(all_stats)
        last_status = src[STATUS_COL].iloc[-1] if STATUS_COL in src.columns else None

        results = []
        running = src.copy()

        for offset in range(7):
            tgt_date = today_real + pd.Timedelta(days=offset)
            day_name = HARI_ID[tgt_date.dayofweek]

            if df_live is not None:
                existing = df_live[df_live[DATE_COL] == tgt_date]
                if not existing.empty:
                    row   = existing.iloc[0]
                    total = int(row[TARGET_COL])
                    mobil = int(row[COL_MOBIL]) if COL_MOBIL in df_live.columns else None
                    motor = int(row[COL_MOTOR]) if COL_MOTOR in df_live.columns else None
                    cond  = get_condition_label(total, KAPASITAS_MAX)
                    results.append((tgt_date, day_name, total, mobil, motor, cond, True))
                    continue

            prev_vals = running[TARGET_COL].values
            lag1  = float(prev_vals[-1])
            roll3 = float(np.mean(prev_vals[-3:])) if len(prev_vals) >= 3 else lag1
            h_code = hari_cats.index(day_name) if day_name in hari_cats else 0
            s_code = stat_cats.index(last_status) if last_status in stat_cats else 0
            same_dow = running[DATE_COL].dt.dayofweek == int(tgt_date.dayofweek)
            lds = float(running[same_dow][TARGET_COL].iloc[-1]) if same_dow.sum() > 0 else lag1

            feat_row = pd.DataFrame([{
                "dayofweek": int(tgt_date.dayofweek),
                "hari_code": h_code, "status_code": s_code,
                "lag_1": lag1, "roll3": roll3, "last_ds": lds, "last_s": lag1,
            }])
            xgb_pred = float(np.clip(model.predict(feat_row[XGB_FEAT])[0], lo, hi)) \
                       if model is not None else lag1
            same_dow_hist = df_hist[DATE_COL].dt.dayofweek == int(tgt_date.dayofweek)
            day_med = float(df_hist[same_dow_hist][TARGET_COL].median()) \
                      if same_dow_hist.sum() >= 3 else float(df_hist[TARGET_COL].median())
            pred_total = float(np.clip(0.5 * xgb_pred + 0.5 * day_med, lo, hi))

            pred_mobil = int(round(pred_total * mobil_ratio)) if mobil_ratio is not None else None
            pred_motor = int(round(pred_total * motor_ratio)) if motor_ratio is not None else None
            cond       = get_condition_label(int(pred_total), KAPASITAS_MAX)

            results.append((tgt_date, day_name, int(pred_total), pred_mobil, pred_motor, cond, False))

            new_row = {DATE_COL: tgt_date, TARGET_COL: pred_total}
            if DAY_COL    in running.columns: new_row[DAY_COL]    = day_name
            if STATUS_COL in running.columns: new_row[STATUS_COL] = last_status
            running = pd.concat([running, pd.DataFrame([new_row])], ignore_index=True)

        return results

    def _update_future_dashboard(self):
        if not self._pred_future or len(self._pred_future) < 3:
            return
        today_p, besok_p, lusa_p = self._pred_future[0], self._pred_future[1], self._pred_future[2]

        tgt_date, _, total, mobil, motor, cond, is_actual = today_p
        suffix = "" if is_actual else "  (Prediksi)"
        self.hdr_saat_ini.config(
            text=f"Parkiran Saat Ini  ({tgt_date.strftime('%d/%m/%Y')}){suffix}")
        color = STATUS_COLOR.get(cond.upper(), C["text"])
        self.lbl_kondisi.config(text=cond, fg=color)
        kosong = max(0, KAPASITAS_MAX - total)
        self.lbl_total.config(text=f"{total:,}")
        self.lbl_kosong.config(text=f"{kosong:,}")
        if mobil is not None: self.lbl_mobil.config(text=f"{mobil:,}")
        if motor is not None: self.lbl_motor.config(text=f"{motor:,}")
        self._draw_donut(tgt_date, total, mobil, motor, kosong)

        d1, _, _, _, _, c1, _ = besok_p
        self.hdr_besok.config(text=f"Parkiran Besok  ({d1.strftime('%d/%m/%Y')})")
        self.lbl_besok.config(text=c1, fg=STATUS_COLOR.get(c1.upper(), C["text"]))

        d2, _, _, _, _, c2, _ = lusa_p
        self.hdr_lusa.config(text=f"Parkiran Lusa  ({d2.strftime('%d/%m/%Y')})")
        self.lbl_lusa.config(text=c2, fg=STATUS_COLOR.get(c2.upper(), C["text"]))

    def _run_thread(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            df = self.df.copy()

            for c in [DAY_COL, CONDITION_COL]:
                if c not in df.columns:
                    self.after(0, lambda c=c: messagebox.showwarning(
                        "Kolom Tidak Ditemukan",
                        f"Kolom '{c}' tidak ditemukan di CSV.\n"
                        "Cek konfigurasi kolom di bagian atas kode."))
                    return

            cond_ds = {c: df[df[CONDITION_COL] == c].sort_values(DATE_COL).reset_index(drop=True)
                       for c in df[CONDITION_COL].unique()}
            split   = {c: split_tt(sub) for c, sub in cond_ds.items()}

            xgb_r = {}
            conds = list(cond_ds.keys())
            total_steps = len(conds)
            step = 0

            def prog(s, t, c, method):
                self.after(0, lambda: self._set_prog(
                    f"[{s}/{t}] {method} — {c}", s/t*100))

            for cond in conds:
                tr, te = split[cond]

                step += 1; prog(step, total_steps, cond, "XGBoost")
                try:
                    p = xgb_forecast_fn(tr, te)
                except Exception as ex:
                    print(f"[XGBoost] {cond}: {ex}")
                    p = [float(tr[TARGET_COL].mean())] * len(te)
                t2 = te.copy(); t2["XGBoost"] = p
                keep = [c for c in [DATE_COL, DAY_COL, CONDITION_COL, TARGET_COL, "XGBoost"] if c in t2.columns]
                xgb_r[cond] = t2[keep].copy()

            self.after(0, lambda: self._set_prog("Menggabungkan hasil...", 97))
            combined = {}
            for cond in conds:
                keep_cols = [c for c in [DATE_COL, DAY_COL, CONDITION_COL, TARGET_COL]
                             if c in split[cond][1].columns]
                rdf = split[cond][1][keep_cols].copy()
                rdf = rdf.rename(columns={TARGET_COL: "Aktual"})
                merge_sources = [(xgb_r, "XGBoost")]
                for res, col in merge_sources:
                    if cond in res and col in res[cond].columns:
                        rdf = rdf.merge(res[cond][[DATE_COL, col]], on=DATE_COL, how="left")
                combined[cond] = rdf

            self.after(0, lambda: self._set_prog("Menghitung evaluasi Total...", 97))
            erows, all_actual, all_pred = [], [], []
            for cond, rdf in xgb_r.items():
                d = rdf.dropna(subset=[TARGET_COL, "XGBoost"])
                if len(d) == 0: continue
                mae_v, rmse_v, mape_v = evaluate(d[TARGET_COL], d["XGBoost"])
                erows.append({"Kondisi Parkiran": cond, "Data Test": len(d),
                              "MAE": round(mae_v, 2), "RMSE": round(rmse_v, 2), "MAPE (%)": round(mape_v, 2)})
                all_actual.extend(d[TARGET_COL].tolist()); all_pred.extend(d["XGBoost"].tolist())
            eval_total = pd.DataFrame(erows).sort_values("RMSE").reset_index(drop=True)
            if all_actual:
                mae_all, rmse_all, mape_all = evaluate(all_actual, all_pred)
                eval_total = pd.concat([eval_total, pd.DataFrame([{"Kondisi Parkiran": "Semua",
                                                                    "Data Test": len(all_actual),
                                                                    "MAE": round(mae_all, 2),
                                                                    "RMSE": round(rmse_all, 2),
                                                                    "MAPE (%)": round(mape_all, 2)}])],
                                       ignore_index=True)

            self.after(0, lambda: self._set_prog("Evaluasi & Forecasting Mobil...", 98))
            eval_mobil, pred_mobil = _xgb_run_col(split, COL_MOBIL)
            self.after(0, lambda: self._set_prog("Evaluasi & Forecasting Motor...", 99))
            eval_motor, pred_motor = _xgb_run_col(split, COL_MOTOR)
            eval_results = {"Total": eval_total, "Mobil": eval_mobil, "Motor": eval_motor}
            eval_df = eval_total

            # DT classification range stats (from full dataset)
            _cond_order = {"Sepi": 0, "Ramai": 1, "Sangat Ramai": 2}
            dt_rows = []
            for cond, grp in df.groupby(CONDITION_COL):
                dt_rows.append({
                    "Kondisi":     cond,
                    "Total Baris": len(grp),
                    "Min Total":   int(grp[TARGET_COL].min()),
                    "Maks Total":  int(grp[TARGET_COL].max()),
                    "Rata Total":  round(grp[TARGET_COL].mean(), 1),
                    "Min Mobil":   int(grp[COL_MOBIL].min()) if COL_MOBIL in grp.columns else "-",
                    "Maks Mobil":  int(grp[COL_MOBIL].max()) if COL_MOBIL in grp.columns else "-",
                    "Rata Mobil":  round(grp[COL_MOBIL].mean(), 1) if COL_MOBIL in grp.columns else "-",
                    "Min Motor":   int(grp[COL_MOTOR].min()) if COL_MOTOR in grp.columns else "-",
                    "Maks Motor":  int(grp[COL_MOTOR].max()) if COL_MOTOR in grp.columns else "-",
                    "Rata Motor":  round(grp[COL_MOTOR].mean(), 1) if COL_MOTOR in grp.columns else "-",
                })
            dt_rows.sort(key=lambda r: _cond_order.get(r["Kondisi"], 99))

            self.results       = combined
            self.results_mobil = pred_mobil
            self.results_motor = pred_motor
            self.evaluation_df = eval_df
            self.eval_results  = eval_results
            self.dt_stats      = dt_rows
            self._pred_future  = self._predict_next_from_live()
            self._pred_7days   = self._predict_7days()

            self.after(0, lambda: self._set_prog("✅ Forecasting selesai!", 100))
            self.after(0, self._on_done)

        except Exception:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Error Forecasting", tb))

    def _on_done(self):
        self.btn_save.config(state="normal")

        # Update Saat Ini / Besok / Lusa
        self._update_future_dashboard()
        self._refresh_file_label()

        # Tabel evaluasi (3 tables)
        for attr, key in [("tree_total","Total"), ("tree_mobil","Mobil"), ("tree_motor","Motor")]:
            t = getattr(self, attr)
            for r in t.get_children(): t.delete(r)
            df_e = self.eval_results.get(key, None)
            if df_e is not None and not df_e.empty:
                for i, (_, row) in enumerate(df_e.iterrows()):
                    t.insert("", "end", values=list(row), tags=("even" if i % 2 else "odd",))

        # Tabel klasifikasi DT (3 tables: Total, Mobil, Motor)
        _dt_key_map = [
            ("dt_tree_total", "Min Total",  "Maks Total",  "Rata Total"),
            ("dt_tree_mobil", "Min Mobil",  "Maks Mobil",  "Rata Mobil"),
            ("dt_tree_motor", "Min Motor",  "Maks Motor",  "Rata Motor"),
        ]
        for tree_attr, k_min, k_maks, k_rata in _dt_key_map:
            t = getattr(self, tree_attr)
            for r in t.get_children(): t.delete(r)
            for i, row in enumerate(self.dt_stats):
                vals = (row["Kondisi"], row["Total Baris"], row[k_min], row[k_maks], row[k_rata])
                t.insert("", "end", values=vals, tags=("even" if i % 2 else "odd",))

        # Tabel prediksi 7 hari
        self._refresh_pred7_table()

        # Combobox kondisi
        conds = list(self.results.keys())
        self.combo_cond["values"] = ["Semua"] + conds
        if conds:
            self.combo_cond.current(0)
            self._refresh_chart()

        self.nb.select(0)

    # --------------------------------------------------------
    # SIMPAN HASIL
    # --------------------------------------------------------
    def _save(self):
        if not self.results:
            messagebox.showwarning("Peringatan", "Belum ada hasil untuk disimpan.")
            return
        out_dir = filedialog.askdirectory(title="Pilih Folder Tujuan")
        if not out_dir: return

        folder = os.path.join(out_dir, "hasil_forecasting_parkiran")
        os.makedirs(folder, exist_ok=True)

        try:
            all_df = pd.concat(
                [rdf.assign(**{"Kondisi Dataset": c}) for c, rdf in self.results.items()],
                ignore_index=True)
            all_df.to_csv(os.path.join(folder, "prediksi_semua.csv"), index=False)

            with pd.ExcelWriter(os.path.join(folder, "prediksi_semua.xlsx")) as w:
                all_df.to_excel(w, sheet_name="Semua", index=False)
                for c, rdf in self.results.items():
                    rdf.to_excel(w, sheet_name=c.replace(" ","_")[:31], index=False)

            if self.eval_results:
                with pd.ExcelWriter(os.path.join(folder, "evaluasi_model.xlsx")) as w:
                    for key, df_e in self.eval_results.items():
                        if df_e is not None and not df_e.empty:
                            df_e.to_excel(w, sheet_name=key, index=False)
                total_e = self.eval_results.get("Total")
                if total_e is not None and not total_e.empty:
                    total_e.to_csv(os.path.join(folder, "evaluasi_model.csv"), index=False)
            elif self.evaluation_df is not None:
                self.evaluation_df.to_excel(
                    os.path.join(folder, "evaluasi_model.xlsx"), index=False)
                self.evaluation_df.to_csv(
                    os.path.join(folder, "evaluasi_model.csv"), index=False)

            # Grafik
            for cond, rdf in self.results.items():
                fig, ax = plt.subplots(figsize=(14,6))
                ax.plot(rdf[DATE_COL], rdf["Aktual"], label="Aktual",
                        color=MODEL_COLOR["Aktual"], linewidth=2.5)
                for m in ["XGBoost"]:
                    if m in rdf.columns:
                        ax.plot(rdf[DATE_COL], rdf[m], label=m,
                                color=MODEL_COLOR[m], linestyle="--", linewidth=1.8)
                ax.set_title(f"Kondisi {cond} — Aktual vs Semua Model")
                ax.legend(); ax.grid(True, alpha=0.2)
                fig.autofmt_xdate(); fig.tight_layout()
                fig.savefig(os.path.join(folder, f"grafik_{cond.replace(' ','_')}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()

            shutil.make_archive(folder, "zip", folder)
            messagebox.showinfo("Tersimpan",
                                f"Hasil disimpan di:\n{folder}\n\nZIP: {folder}.zip")
        except Exception as e:
            messagebox.showerror("Error Simpan", str(e))


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()

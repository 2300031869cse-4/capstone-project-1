"""IntelliBudget: calibrated multi-label budget-line classifier with a human-review path (DSCI-29)."""
import json, re, hashlib
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st

HERE = Path(__file__).parent
M = json.loads((HERE / "model.json").read_text())
R = json.loads((HERE / "results.json").read_text())
GROUPS, AUTO = M["groups"], R["item"]["auto_groups"]
VI = {v: i for i, v in enumerate(M["vocab"])}
W, B = np.array(M["W"]), np.array(M["b"])
PA, PB = np.array(M["pa"]), np.array(M["pb"])
IDX = {g: [j for j, l in enumerate(M["labels"]) if l[0] == g] for g in GROUPS}
CATMAP = {"Administration": "Administration", "Facilities": "Facilities", "Information Technology": "Technology",
          "Instruction": "Academic", "Library": "Library", "Public Outreach": "Community Engagement",
          "Research": "Research", "Student Services": "Student Support"}
COV_MIN = 0.75

def norm(t):
    return [M["abbr"].get(x, x) for x in re.findall(r"[a-z0-9]+", str(t).lower())]

def classify(text):
    w = norm(text); toks = w + [a + " " + b for a, b in zip(w, w[1:])]
    x = np.zeros(len(VI))
    for k, n in Counter(toks).items():
        if k in VI: x[VI[k]] = (1 + np.log(n)) * M["idf"][VI[k]]
    nr = np.linalg.norm(x); x = x / nr if nr else x
    raw = 1 / (1 + np.exp(-(W @ x + B)))
    q = np.clip(raw, 1e-6, 1 - 1e-6)
    cal = 1 / (1 + np.exp(-(PA * np.log(q / (1 - q)) + PB)))
    cov = np.mean([t in VI for t in w]) if w else 0.0
    out, low = {}, []
    for g in GROUPS:
        b = max(IDX[g], key=lambda j: cal[j]); auto = g in AUTO
        ok = auto and cal[b] >= M["tau"][g] and cov >= COV_MIN
        out[g] = dict(label=M["labels"][b][1], p=float(cal[b]), raw=float(raw[b]), auto=auto, ok=bool(ok))
        if auto and not ok: low.append(g)
    status = "review" if (low or cov < COV_MIN) else "auto"
    return dict(g=out, cov=float(cov), status=status, low=low, minp=min(out[g]["p"] for g in AUTO))

def rec_key(i, ver, text):
    return hashlib.sha256(f"{i}|{ver}|{M['version']}|{text}".encode()).hexdigest()[:16]

def ingest(df, default_ver, store, quar):
    cols = {c.lower().strip(): c for c in df.columns}
    tc = cols.get("description") or cols.get("text") or df.columns[0]
    ic, vc = cols.get("line_item_id"), cols.get("source_version") or cols.get("version")
    new = same = 0
    for n, r in enumerate(df.to_dict("records")):
        rid = str(r[ic]) if ic else f"ROW{n+1}"; text = str(r[tc]) if pd.notna(r[tc]) else ""
        ver = str(r[vc]).strip() if vc and pd.notna(r[vc]) else default_ver.strip()
        if not ver or ver == "nan": quar.append((rid, "no source version (lineage)")); continue
        if not text.strip(): quar.append((rid, "empty description")); continue
        k = rec_key(rid, ver, text)
        if k in store: same += 1; continue
        prev = store.get("id:" + rid)
        if prev and prev["ver"] == ver and prev["k"] != k: quar.append((rid, "conflicting content for existing ID")); continue
        rec = dict(k=k, id=rid, ver=ver, text=text, c=classify(text)); store[k] = rec; store["id:" + rid] = rec; new += 1
    return new, same

def records(store): return [v for k, v in store.items() if not k.startswith("id:")]
def state_hash(store):
    s = "|".join(sorted(r["k"] + json.dumps([o["label"] for o in r["c"]["g"].values()]) for r in records(store)))
    return hashlib.sha256(s.encode()).hexdigest()[:12]

st.set_page_config(page_title="IntelliBudget", page_icon="📊", layout="wide")
ss = st.session_state
ss.setdefault("store", {}); ss.setdefault("quar", []); ss.setdefault("dec", {}); ss.setdefault("last", None)
st.title("IntelliBudget: calibrated budget-line classifier")
st.caption("Multi-label classification with per-label calibrated probabilities and a human-review path for uncertain items.")
t1, t2, t3, t4 = st.tabs(["Classify", "Batch & review queue", "Evidence & KPIs", "About"])

with t1:
    ex = ["lab Rsch equip. cloud service - annual allocation", "SaaS subscription: software subscriptions",
          "career services, maintenance service", "zq417 xvprj kfund campus bridging"]
    pick = st.selectbox("Example", [""] + ex)
    text = st.text_input("Budget line item", value=pick, placeholder="e.g. digital archive svcs professional services")
    if text.strip():
        r = classify(text)
        (st.success if r["status"] == "auto" else st.warning)(
            "Auto-labelled" if r["status"] == "auto" else
            "Routed to human review" + (" (unfamiliar vocabulary)" if r["cov"] < COV_MIN else "") +
            (f" (low confidence: {', '.join(r['low'])})" if r["low"] else ""))
        st.caption(f"Vocabulary coverage {r['cov']:.0%}")
        cols = st.columns(len(GROUPS))
        for c, g in zip(cols, GROUPS):
            o = r["g"][g]; c.metric(g.replace("_", " "), o["label"], f"{o['p']:.1%} calibrated")
            c.progress(min(max(o["p"], 0.0), 1.0))
            c.caption(f"raw {o['raw']:.1%} · " + ("confident" if o["ok"] else "review" if o["auto"] else "unverified suggestion"))

with t2:
    st.write("Upload a CSV with `description` (or `text`) and optionally `line_item_id` and `source_version`. "
             "Rows without a source version are quarantined, never silently committed.")
    ver = st.text_input("Default source version", f"intellibudget_100k@{R['source_sha256_16']}")
    up = st.file_uploader("CSV file", type=["csv", "txt"])
    c1, c2, c3 = st.columns(3)
    if c1.button("Ingest", type="primary") and up is not None:
        df = pd.read_csv(up); ss.last = df
        n, s = ingest(df, ver, ss.store, ss.quar); st.info(f"{n} new, {s} unchanged.")
    if c2.button("Replay same batch (idempotency check)") and ss.last is not None:
        before, cnt = state_hash(ss.store), len(records(ss.store))
        n, s = ingest(ss.last, ver, ss.store, ss.quar)
        same = before == state_hash(ss.store) and cnt == len(records(ss.store))
        (st.success if same else st.error)(f"Replay: {n} new, {s} unchanged. State hash {'identical' if same else 'CHANGED'}.")
    if c3.button("Reset"): ss.store, ss.quar, ss.dec, ss.last = {}, [], {}, None; st.rerun()
    rs = records(ss.store); rv = [r for r in rs if r["c"]["status"] == "review"]
    m = st.columns(4); m[0].metric("Committed", len(rs)); m[1].metric("Auto-labelled", f"{1 - len(rv)/len(rs):.1%}" if rs else "-")
    m[2].metric("In review queue", len(rv)); m[3].metric("Quarantined", len(ss.quar))
    if rs: st.caption(f"State hash: {state_hash(ss.store)}")
    st.subheader("Review queue")
    if rv:
        rows = []
        for r in rv[:300]:
            d = ss.dec.get(r["k"], {}); row = dict(id=r["id"], description=r["text"], confirmed=r["k"] in ss.dec, key=r["k"])
            for g in ("function", "object_type", "operating_status", "position_type"): row[g] = d.get(g, r["c"]["g"][g]["label"])
            rows.append(row)
        cfg = {g: st.column_config.SelectboxColumn(g, options=M["classes"][g]) for g in ("function", "object_type", "operating_status", "position_type")}
        ed = st.data_editor(pd.DataFrame(rows), column_config=cfg, hide_index=True, use_container_width=True,
                            disabled=["id", "description", "key"], column_order=["id", "description", "function", "object_type", "operating_status", "position_type", "confirmed"])
        if st.button("Save confirmed rows"):
            for _, row in ed[ed["confirmed"]].iterrows():
                ss.dec[row["key"]] = {g: row[g] for g in ("function", "object_type", "operating_status", "position_type")}
            st.success("Saved."); st.rerun()
        if len(rv) > 300: st.caption(f"Showing 300 of {len(rv)}.")
    else: st.caption("Nothing waiting for review.")
    if ss.quar: st.subheader("Quarantine"); st.dataframe(pd.DataFrame(ss.quar, columns=["id", "reason"]), hide_index=True)
    if rs:
        out = []
        for r in rs:
            d = ss.dec.get(r["k"]); v = lambda g: d[g] if d else r["c"]["g"][g]["label"]; fn = v("function")
            out.append(dict(line_item_id=r["id"], description=r["text"], function=fn, object_type=v("object_type"),
                            operating_status=v("operating_status"), reporting_category=CATMAP.get(fn, r["c"]["g"]["reporting_category"]["label"]),
                            position_type=v("position_type"), route="human-confirmed" if d else r["c"]["status"],
                            min_confidence=round(r["c"]["minp"], 4), source_version=r["ver"], record_hash=r["k"]))
        st.download_button("Export labelled CSV", pd.DataFrame(out).to_csv(index=False), "intellibudget_labels.csv", "text/csv")

with t3:
    o, c, it, L, N = R["O2_reference"], R["O3_platt"], R["item"], R["latency"], R["nt"]
    lr, lp, rp = c["log_loss"] / o["log_loss"], L["final"]["p95"] / L["ref"]["p95"], 100 * (it["review_precision"] - it["ref_review_precision"])
    K = [("KPI-1 log loss", f"{c['log_loss']:.4f} vs ref {o['log_loss']:.4f} (ratio {lr:.3f}; target ≤ 0.80)", lr <= .8),
         ("KPI-2 ECE", f"{c['ece']:.4f} vs ref {o['ece']:.4f} (target ≤ 0.05, no worse than ref)", c["ece"] <= .05 and c["ece"] <= o["ece"]),
         ("KPI-3 review precision", f"{it['review_precision']:.1%} vs ref {it['ref_review_precision']:.1%} ({rp:+.1f} pts; target +3)", rp >= 3),
         ("KPI-4 lineage coverage", f"{R['lineage']:.0%} (target ≥ 95%)", R["lineage"] >= .95),
         ("KPI-5 reconciliation errors", "0 across replay, retry and partial-failure fixtures", N["NT5"]["identical_hash"]),
         ("KPI-6 latency p95", f"{L['final']['p95']:.1f} ms vs ref {L['ref']['p95']:.1f} ms (ratio {lp:.2f}; target ≤ 0.80)", lp <= .8)]
    st.subheader("KPI scorecard (held-out acceptance institutions)")
    st.table(pd.DataFrame([dict(KPI=k, Measured=m_, Result="Pass" if p else "Not met") for k, m_, p in K]))
    st.caption("Two label groups (operating_status, position_type) carry almost no signal in the text, which sets a floor on log loss, so the 0.80× target is not reachable on this data.")
    a, b = st.columns(2)
    rel = pd.DataFrame([x for x in R["reliability"] if x["conf"] is not None])
    a.subheader("Reliability"); a.line_chart(pd.DataFrame({"perfect": rel["conf"], "calibrated": rel["acc"], "reference": rel["ref_acc"]}, index=rel["conf"].round(3)))
    b.subheader("Per-group results")
    b.dataframe(pd.DataFrame([dict(group=g, accuracy=R["group"][g]["accuracy"], majority=R["majority"][g], tau="review-only" if R["group"][g]["tau"] >= .99 else R["group"][g]["tau"], auto_rate=R["group"][g]["auto_rate"]) for g in GROUPS]), hide_index=True)
    st.subheader("Negative-test campaign")
    st.table(pd.DataFrame([("NT-1 confident wrong labels", f"{N['NT1']['flagged']:.0%} of {N['NT1']['n']} caught and quarantined"),
        ("NT-2 institution-specific vocabulary", f"{N['NT2']['flagged']:.0%} flagged to review (clean-text false flags {N['NT2']['baseline_flag']:.0%})"),
        ("NT-3 rare categories", f"{', '.join(N['NT3']['labels'])}: ECE ≤ {max(N['NT3']['ece']):.4f}"),
        ("NT-4 lineage loss", f"{N['NT4']['rejected']}/{N['NT4']['of']} rejected, {N['NT4']['committed_without_lineage']} committed"),
        ("NT-5 non-idempotent replay", f"state hash identical after partial failure and retries: {N['NT5']['identical_hash']}")], columns=["Test", "Outcome"]))

with t4:
    st.markdown(f"""**Method.** O2 reference: abbreviation normalisation + one-vs-rest logistic regression on word/bigram TF-IDF
(chosen over Naïve Bayes). O3 contribution: per-label Platt calibration fitted on separate institutions. O4: low-confidence items or
unfamiliar vocabulary go to the review queue. Splits are by institution: {R['split']['train']} train, {R['split']['calibration']} calibration,
{R['split']['acceptance']} acceptance rows (reserved before tuning). Model `{M['version']}`; source hash `{R['source_sha256_16']}`.""")

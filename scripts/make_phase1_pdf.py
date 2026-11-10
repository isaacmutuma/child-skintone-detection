"""Build docs/PHASE1_REPORT.pdf: a short, shareable Phase 1 report.

Every number is read from results/phase1/*.json (nothing typed by hand), so the PDF
always matches the committed results. The chart is drawn from the same JSON.
Rendering uses headless Google Chrome (print to PDF).

  .venv/bin/python scripts/make_phase1_pdf.py [--author "Isaac Mutuma"] [--affiliation "..."]
"""

import argparse
import base64
import html
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase1"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BLACK = "C(group, Treatment(reference='White'))[T.Black]"
NAMES = {"mivolo_face": "MiVOLO v2 (face age)", "fairface_clf": "FairFace age classifier",
         "clip_vitb32": "CLIP ViT-B/32 zero-shot", "clip_vitl14": "CLIP ViT-L/14 zero-shot",
         "vlm": "Qwen2-VL-2B yes/no"}
DS = {"utkface": "UTKFace", "fairface_val": "FairFace"}
SHORT = {"mivolo_face": "MiVOLO v2", "fairface_clf": "FairFace classifier",
         "clip_vitb32": "CLIP ViT-B/32", "clip_vitl14": "CLIP ViT-L/14", "vlm": "Qwen2-VL-2B"}


def load(name):
    return json.loads((RES / name).read_text())


def pts(x, d=1):
    """Proportion -> percentage points string."""
    return f"{x * 100:.{d}f}".replace("-", "\u2212")   # typographic minus


def ci_pts(ci, d=1):
    return f"[{pts(ci[0], d)}, {pts(ci[1], d)}]"


def figure(pe):
    """Dot-and-whisker plot of the primary endpoint, returned as a base64 PNG."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8.5})
    rows = []
    for ds in ["utkface", "fairface_val"]:
        for det, r in pe["datasets"][ds]["detectors"].items():
            rows.append((ds, SHORT[det], r["tpr_gap"], r["ci"]))
    fig, ax = plt.subplots(figsize=(5.6, 2.9), dpi=220)
    y, ticks, labels = 0, [], []
    for ds in ["utkface", "fairface_val"]:
        for _, name, g, ci in [r for r in rows if r[0] == ds]:
            filled = ds == "utkface"
            ax.plot([ci[0] * 100, ci[1] * 100], [y, y], color="black", lw=0.9)
            ax.plot([g * 100], [y], "o", ms=4.2, color="black", mfc="black" if filled else "white", mew=0.9)
            ticks.append(y)
            labels.append(f"{name} ({DS[ds]})")
            y -= 1
        y -= 0.7
    ax.axvline(0, color="0.45", lw=0.7, ls="--")
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("TPR gap, White $-$ Black (percentage points)")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build(author, affiliation, date):
    pe = load("primary_endpoint.json")
    m = {(det, ds): load(f"metrics_{det}_{ds}.json")
         for ds in ["utkface", "fairface_val"] for det in pe["datasets"][ds]["detectors"]}
    utk, ff = pe["datasets"]["utkface"], pe["datasets"]["fairface_val"]
    gaps_utk = [r["tpr_gap"] for r in utk["detectors"].values()]
    holm_max = max(r["p_holm"] for r in utk["detectors"].values())
    mv_u, mv_f = m[("mivolo_face", "utkface")], m[("mivolo_face", "fairface_val")]
    teen = lambda det, g: m[(det, "utkface")]["groups"][g]["tpr_by_stratum"]["10-17"]  # noqa: E731
    young_gaps = [abs(m[(d, "utkface")]["groups"]["White"]["tpr_by_stratum"]["0-9"]
                      - m[(d, "utkface")]["groups"]["Black"]["tpr_by_stratum"]["0-9"])
                  for d in utk["detectors"]]
    age_u = mv_u["primary_contrast"]["age_err_gap_b_minus_a"]
    age_f = mv_f["primary_contrast"]["age_err_gap_b_minus_a"]
    age_d3 = m[("fairface_clf", "utkface")]["primary_contrast"]["age_err_gap_b_minus_a"]
    ff_u = mv_u["primary_contrast"]["face_fail_gap_b_minus_a"]
    ff_f = mv_f["primary_contrast"]["face_fail_gap_b_minus_a"]
    orr = lambda det: m[(det, "utkface")]["confounder_logit"]["terms"][BLACK]  # noqa: E731
    eq = lambda det: m[(det, "utkface")]["equalised_thresholds"]["groups"]["Black"]  # noqa: E731
    n_utk = mv_u["groups"]
    n_ff = mv_f["groups"]

    table_rows = []
    for ds_key, ds in [("utkface", utk), ("fairface_val", ff)]:
        for i, (det, r) in enumerate(ds["detectors"].items()):
            g = m[(det, ds_key)]["groups"]
            first = ' class="grp"' if i == 0 and ds_key == "fairface_val" else ""
            table_rows.append(
                f"<tr{first}><td>{DS[ds_key] if i == 0 else ''}</td><td>{SHORT[det]}{'*' if r['tag'] == 'full_sub' else ''}</td>"
                f"<td>{pts(g['White']['tpr_strat'])}</td><td>{pts(g['Black']['tpr_strat'])}</td>"
                f"<td>{pts(r['tpr_gap'])}</td><td>{pts(r['ci'][0])} to {pts(r['ci'][1])}</td>"
                f"<td>{r['p_holm']:.3f}</td></tr>")
    tw = lambda d: pts(teen(d, "White"), 0)  # noqa: E731
    tb = lambda d: pts(teen(d, "Black"), 0)  # noqa: E731
    nb = n_utk["Black"]["n_minor"]
    missed_b = nb - n_utk["Black"]["n_minor_age_est"]
    missed_w = n_utk["White"]["n_minor"] - n_utk["White"]["n_minor_age_est"]
    img = figure(pe)
    css = """
    @page { size: A4; margin: 19mm 22mm 18mm; }
    body { font-family: 'STIXGeneral', 'Times', 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.32;
           color: #000; margin: 0; }
    .title { text-align: center; font-size: 15.5pt; line-height: 1.25; margin: 0 0 8px; font-weight: normal; }
    .author { text-align: center; font-size: 10.5pt; margin: 0; }
    .date { text-align: center; font-size: 10pt; margin: 2px 0 14px; }
    .abs-h { text-align: center; font-weight: bold; font-size: 9.5pt; margin: 0 0 3px; }
    .abstract { margin: 0 9mm 10px; font-size: 9.5pt; text-align: justify; }
    h2 { font-size: 11.5pt; margin: 13px 0 4px; }
    p { margin: 0 0 6px; text-align: justify; hyphens: auto; }
    p + p { text-indent: 1.4em; }
    table { border-collapse: collapse; margin: 6px auto 2px; font-size: 9.3pt; }
    th, td { padding: 1.5px 8px; text-align: left; vertical-align: top; }
    thead th { border-top: 1.2px solid #000; border-bottom: 0.7px solid #000; font-weight: normal; }
    tbody tr:last-child td { border-bottom: 1.2px solid #000; }
    tr.grp td { border-top: 0.5px solid #000; }
    td.num, .res td:nth-child(n+3) { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .res th:nth-child(n+3) { text-align: right; }
    .cap { font-size: 9pt; margin: 3px 0 12px; text-align: center; }
    .cap.left { text-align: justify; }
    figure { margin: 8px 0 0; text-align: center; } figure img { width: 80%; }
    .refs { font-size: 8.8pt; padding-left: 18px; margin: 4px 0 0; } .refs li { margin: 0 0 2px; }
    .det { width: 100%; } .det td:first-child { white-space: nowrap; }
    """
    a = html.escape
    body = f"""
<div class="title">Do Automated Child Detectors Miss Darker-Skinned Children?<br>A Preregistered Pilot Audit</div>
<p class="author">{a(author)}{'<br>' + a(affiliation) if affiliation else ''}</p>
<p class="date">{a(date)}</p>

<div class="abs-h">Abstract</div>
<div class="abstract">Text-to-image training data is often filtered with automated child detectors, but it is not
known whether these detectors work equally well for children of different skin tones. In this pilot I evaluate five
detectors on two face datasets with perceived-race labels and compare true-positive rates (TPR) for White and Black
minors after stratifying by age. The analysis plan was preregistered before any detector was run on the full data.
On UTKFace, every detector flags Black minors less often than White minors, with gaps of {pts(min(gaps_utk))} to
{pts(max(gaps_utk))} percentage points that remain significant after Holm correction. Most of the difference comes
from 10&ndash;17-year-olds, and the age-estimation models tend to estimate Black minors as older than they are. On
FairFace, where only children under ten can be evaluated, I find no difference. Since these are cropped face images,
the next phase repeats the analysis on in-the-wild images from ICCWD with skin-tone annotations.</div>

<h2>1&nbsp;&nbsp;Introduction</h2>
<p>Cre&#539;u et al. [1] evaluated a range of child detectors as a defence against the generation of child sexual
abuse material, and found that even the best method misses more than 6% of child images. They also point out that
when sub-groups are filtered at different rates, the data that remains is biased. As far as I know, skin tone has not
been examined as one of these sub-groups. If detectors miss darker-skinned children more often, a filtered dataset
would keep proportionally more images of them, which is the opposite of what the filter is meant to do. This report
describes a first, controlled test of that question.</p>

<h2>2&nbsp;&nbsp;Method</h2>
<p>I used UTKFace [3], which has {mv_u['n_images']:,} aligned face images with exact ages, and the validation split
of FairFace [4], with {mv_f['n_images']:,} faces labelled in age bins. Both datasets provide perceived-race labels,
which I used as given; the main comparison is between White and Black. Minors (under 18) are used for the TPR and
adults aged 20&ndash;59 for the false-positive rate (FPR). The FairFace 10&ndash;19 bin crosses the age of 18, so I
left it out. This gives {n_utk['White']['n_minor']:,} White and {nb} Black minors in UTKFace, and
{n_ff['White']['n_minor']} and {n_ff['Black']['n_minor']} in FairFace.</p>
<p>The five detectors are listed in Table 1. Two estimate age from the face, two are zero-shot CLIP classifiers, and
one is a small vision-language model answering a yes/no question. If no face is found, the image counts as not
flagged, since a filter would keep it.</p>

<table class="det"><thead><tr><th>Detector</th><th>An image is flagged as containing a child if</th></tr></thead><tbody>
<tr><td>MiVOLO v2, face only [5]</td><td>the youngest detected face is estimated to be under 18, as in [1]</td></tr>
<tr><td>FairFace classifier</td><td>the expected age over the nine FairFace age bins is under 18 (UTKFace only)</td></tr>
<tr><td>CLIP ViT-B/32, ViT-L/14 [6]</td><td>the best of four child prompts scores higher than the best of three adult prompts</td></tr>
<tr><td>Qwen2-VL-2B [7]</td><td>&ldquo;yes&rdquo; is more likely than &ldquo;no&rdquo; as the answer to &ldquo;Is there a child (a person under 18 years old) in this image?&rdquo;</td></tr>
</tbody></table>
<div class="cap">Table 1: Detectors and decision rules.</div>

<p>Older children are harder to detect for every model, so comparing raw rates would penalise whichever group has
more teenagers. I therefore computed the TPR within age strata (0&ndash;9 and 10&ndash;17 for UTKFace, 0&ndash;2 and
3&ndash;9 for FairFace) and averaged the strata with the same weights for both groups. Confidence intervals come from
2,000 bootstrap resamples within each group, p-values from a permutation test that shuffles the group labels within
strata, and Holm's method corrects for testing several detectors on the same dataset. I also looked at
face-detection failures, errors in the estimated age, false positives on adults, a logistic regression adjusting for
age and image quality, and per-group decision thresholds.</p>
<p>The hypotheses and analysis were written down and frozen before the full runs. Three things changed afterwards,
and all are recorded in the preregistration. The original FairFace model weights were no longer available, so I used
a FairFace-trained ViT with the same age bins. After seeing the results, I replaced the bootstrap intervals for
face-detection failure with Wilson intervals, because the bootstrap gives a zero-width interval when a group has no
failures. I also found that the CLIP models had first been run with the wrong activation function (GELU instead of
the QuickGELU they were trained with); I fixed this and reran them. All results below use the corrected models, and
no conclusion changed.</p>

<h2>3&nbsp;&nbsp;Results</h2>
<table class="res"><thead><tr><th>Dataset</th><th>Detector</th><th>TPR White</th><th>TPR Black</th><th>Gap</th><th>95% CI</th><th>Holm p</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<div class="cap left">Table 2: Age-stratified TPR (%) and the White &minus; Black gap in percentage points.
*Qwen2-VL was run on a preregistered stratified subsample ({m[('vlm', 'utkface')]['n_images']:,} UTKFace and
{m[('vlm', 'fairface_val')]['n_images']:,} FairFace images).</div>

<p>On UTKFace, all five detectors show a positive gap whose confidence interval excludes zero, so the preregistered
hypothesis holds there (Table 2, Figure 1). On FairFace, none of the four detectors shows a gap.</p>
<p>On UTKFace the difference comes almost entirely from teenagers. Among 10&ndash;17-year-olds, MiVOLO flags
{tw('mivolo_face')}% of White minors and {tb('mivolo_face')}% of Black minors; for the two CLIP models and Qwen2-VL
the figures are {tw('clip_vitb32')}% and {tb('clip_vitb32')}%, {tw('clip_vitl14')}% and {tb('clip_vitl14')}%, and
{tw('vlm')}% and {tb('vlm')}%. Under ten, the gap is at most {pts(max(young_gaps))} points for any detector. FairFace
only allows children under ten to be tested, which is probably why it shows no gap, although this explanation was
not part of the original plan.</p>

<figure><img src="data:image/png;base64,{img}" alt="TPR gap with 95% confidence interval for each detector"></figure>
<div class="cap">Figure 1: White &minus; Black TPR gap with 95% bootstrap confidence intervals.
Filled markers: UTKFace; open markers: FairFace.</div>

<p>The age-estimation models suggest one reason. Among minors whose face was found, MiVOLO's age estimates for Black
minors are on average {age_u['gap']:.2f} years higher, relative to their true age, than for White minors on UTKFace
(95% CI {age_u['ci'][0]:.2f} to {age_u['ci'][1]:.2f}), and {age_f['gap']:.2f} years higher on FairFace
({age_f['ci'][0]:.2f} to {age_f['ci'][1]:.2f}, using bin midpoints). The FairFace classifier shows a smaller
difference of {age_d3['gap']:.2f} years. Face detection does not explain the gap: on UTKFace, MiVOLO missed the face
in {missed_b} of the {nb} Black minors' images and in {missed_w} of the {n_utk['White']['n_minor']:,} White minors'
images. The gap also remains in a logistic regression that adjusts for age, brightness, blur and face size, with odds
ratios for Black minors of {orr('mivolo_face')['or']:.2f} for MiVOLO, {orr('clip_vitb32')['or']:.2f} and
{orr('clip_vitl14')['or']:.2f} for the CLIP models, and {orr('vlm')['or']:.2f} for Qwen2-VL. For the FairFace
classifier the odds ratio is {orr('fairface_clf')['or']:.2f} and not significant.</p>
<p>Lowering the decision threshold for Black minors until their TPR matches the overall rate is possible, but costly.
The share of Black adults flagged as children rises from {pts(eq('mivolo_face')['fpr_default'])}% to
{pts(eq('mivolo_face')['fpr_equalised'])}% for MiVOLO, and from {pts(eq('clip_vitb32')['fpr_default'])}% to
{pts(eq('clip_vitb32')['fpr_equalised'])}% for CLIP ViT-B/32.</p>

<h2>4&nbsp;&nbsp;Limitations</h2>
<p>This is a pilot and has clear limits. The images are cropped and aligned faces, which are much easier than real
training data; [1] reports that MiVOLO detects only 46&ndash;59% of children in the wild. The groups are perceived-race
labels rather than measured skin tone. UTKFace contains only {nb} Black minors, {n_utk['Black']['n_minor_by_stratum']['10-17']}
of them teenagers, and its age and race labels are known to be noisy. Finally, my detectors are not identical to those
in [1]: the MiVOLO version used there is not stated, and my FairFace model is a substitute with a different rule for
the 10&ndash;19 age bin.</p>

<h2>5&nbsp;&nbsp;Next steps</h2>
<p>The next phase uses ICCWD [2], which contains 10,000 human-labelled image&ndash;caption pairs from CC3M, 1,675 of
them with children, and is the benchmark used in [1]. I plan to annotate the child images and a sample of adult
images with the Monk Skin Tone scale, using two annotators, and then rerun the same detectors together with
caption-based and combined image-and-caption detectors. This will show whether the gap holds on realistic images and
whether combining the two modalities reduces it. As in this phase, the analysis will be preregistered before any
detector is run.</p>

<h2>References</h2>
<ol class="refs">
<li>A.-M. Cre&#539;u, K. Kireev, A. Abdalla, W. Obinna, R. Meier, S. A. Bargal, E. M. Redmiles, and C. Troncoso. Evaluating concept filtering defenses against child sexual abuse material generation by text-to-image models. In <i>IEEE Symposium on Security and Privacy</i>, 2026. arXiv:2512.05707.</li>
<li>K. Kireev, A.-M. Cre&#539;u, R. Meier, S. A. Bargal, E. Redmiles, and C. Troncoso. A manually annotated image-caption dataset for detecting children in the wild. arXiv:2506.10117, 2025.</li>
<li>Z. Zhang, Y. Song, and H. Qi. Age progression/regression by conditional adversarial autoencoder. In <i>CVPR</i>, 2017.</li>
<li>K. K&auml;rkk&auml;inen and J. Joo. FairFace: Face attribute dataset for balanced race, gender, and age. In <i>WACV</i>, 2021.</li>
<li>M. Kuprashevich and I. Tolstykh. MiVOLO: Multi-input transformer for age and gender estimation. arXiv:2307.04616, 2023.</li>
<li>A. Radford et al. Learning transferable visual models from natural language supervision. In <i>ICML</i>, 2021.</li>
<li>P. Wang et al. Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution. arXiv:2409.12191, 2024.</li>
</ol>"""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Phase 1 Report</title>"
            f"<style>{css}</style></head><body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Isaac Mutuma")
    ap.add_argument("--affiliation", default="")
    ap.add_argument("--date", default="November 2026")
    ap.add_argument("--out", default=str(ROOT / "docs" / "PHASE1_REPORT.pdf"))
    a = ap.parse_args()
    page = build(a.author, a.affiliation, a.date)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(page)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                        f"--user-data-dir={tmp}/chrome", "--no-pdf-header-footer",
                        f"--print-to-pdf={a.out}", src.as_uri()],
                       check=True, capture_output=True)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())

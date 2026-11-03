# Five-minute demo script

Setup (before the judges arrive):
1. `bash scripts/reproduce.sh` has been run, or at least `train`, `evidence` and `pack` from it.
2. Start the server: `CLINICTUNE_BUNDLE_DIR=models/clinictune-v1 uv run clinictune serve`, and open
   http://localhost:8000.
3. Keep `data/onboarding/montgomery-tb.zip` and the `data/onboarding/montgomery-tb_unseen/` folder at hand.
   Montgomery County (USA) was held out of training: the model has never seen this hospital.

## 1. The problem, in one image (45 seconds)

**Screen** tab, Chest X-ray, clinic **Not onboarded**. Drop
`montgomery-tb_unseen/label0_MCUCXR_0015_0.png`, a healthy patient.

The aid either escalates or refers this healthy patient. Say: *"A model that scores 0.98 ROC-AUC at the
hospitals it was trained on refers 85% of healthy patients at this new hospital. This is why AI screening
tools disappoint in the field."* Open **Validation**: 6 of 6 unseen hospitals broke the 90% guarantee.

## 2. Onboarding a clinic (90 seconds)

**Clinics** tab. Register "Montgomery County TB clinic", then onboard it with `montgomery-tb.zip`: 100
X-rays with locally confirmed diagnoses. The five numbered steps run in about a minute.

Walk through the report card:
- **Shift:** "severe". A model can tell this clinic's images apart from the training hospitals' perfectly.
- **Before:** specificity 15%; the 90% guarantee held for only 5% of healthy patients.
- **Two strategies compared, cross-fitted:** recalibration alone stays at ROC-AUC 0.66. The local
  adaptation layer reaches 0.93, so it was deployed.
- **After:** sensitivity 90% (95% CI 76 to 97%), specificity 77%, calibration error 0.55 → 0.07.
- **Verdict:** "use with caution", with the exact reason. *"It tells the clinic the truth, including when
  it is not good enough."*

## 3. Screening, the clinician's view (60 seconds)

Back to **Screen**, clinic set to Montgomery. Drop `label1_MCUCXR_0166_1.png`:
- The scan line sweeps the lightbox, then the **Refer** stamp lands on the slip.
- The heatmap concentrates on the upper lobes (where TB lives), not on the "LT ERECT" marker.
- **Most similar reference cases:** six confirmed cases from the training hospitals.
- Switch the note language to हिन्दी or ಕನ್ನಡ.
- **Print slip** gives a one-page referral for the patient's OPD card.

Drop a blurry or wrong-modality image, for example a fundus photo with Chest X-ray selected, to show the
quality gate answering *"retake"* instead of guessing.

## 4. A morning's queue and monitoring (45 seconds)

**Queue**: drop the whole `montgomery-tb_unseen` folder. It sorts itself with referrals at the top, and
**Export CSV** gives the day's register. In **Monitoring**: results per day, rates, and alerts. Enter a
confirmatory result on a slip ("Confirmatory test came back: Positive") and it feeds observed sensitivity.
A confirmed positive that was called low risk raises a critical alert.

## 5. Why trust any of it (30 seconds)

**Validation** tab:
- The charts show recalibration flat and local adaptation rising, on every hospital.
- The 90% guarantee is red before onboarding and back after it, at all 6 hospitals.
- Shortcut tests use border-only images and lungs blanked with manual masks.
- **The subgroup table shows our own negative result:** at Shenzhen, coverage for TB-positive women was 82%
  against 94% for men. *"We report it because per-subgroup calibration is the next thing to build."*

Close with: it runs on a laptop CPU (about 0.5 s of model time per X-ray and 0.1 s per fundus photo), works offline, and stores no
images.

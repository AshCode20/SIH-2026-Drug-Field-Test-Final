# SIH 2026 — Final Jury-Friendly Streamlit Prototype

A clean, professional Streamlit frontend for the SIH 2026 smart field-test analyzer prototype.

## What changed
- SHA-256 is **not shown anywhere in the frontend**. It remains inside the stored record and the generated PDF report.
- Added email/password **Sign in** and **Create account**.
- Added **Continue without signing in** for quick demo/jury access.
- Added attractive neutral, professional navigation with icons.
- Added Dashboard, New Test, History, Verify Image, How It Works and System Status.
- Location is requested only after a photo is available.
- Verification requires only **Case ID + original image upload**.
- Verification never displays the stored or calculated SHA-256 value.
- Added a clean result card, case-information card, location card and downloadable PDF report.
- Preserved the supplied dataset-trained colour classifier models.

# SIH 2026 Digital Companion for Field Drug Testing

## 🔗 Live Demo

[Open the Live SIH Application](https://sih-2026-drug-field-test-final-495d3fmjovcsqfh53aeecg.streamlit.app)
## Run in VS Code / PowerShell
Use the Python 3.11 installation that has the project dependencies:

```powershell
& "C:\Users\Asus\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
& "C:\Users\Asus\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

## Important prototype note
The supplied training data is synthetic/demo data. The result is presumptive only. Real deployment requires authorised kit-specific reference data, validation, security review and laboratory correlation.

Browser GPS also depends on browser permission and secure-context rules. On localhost it generally works; remote deployment normally needs HTTPS.

# Transformation Strategy Lab — Hospital at Home Prototype

This Streamlit prototype contains:

- a four-scenario landing page,
- one active scenario: Hospital at Home,
- role-specific starting assumptions,
- a fixed 100-point transformation strategy budget,
- an animated coordination network,
- staffing-shortage and sponsor-departure shocks,
- adoption and persistence outcomes,
- structured debrief questions.

The other three scenario cards are placeholders for later development.

## Run locally

### 1. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

If needed:

```bash
python -m streamlit run app.py
```

## Prototype flow

1. Choose **Hospital at Home** on the landing page.
2. In **Assumptions**, set the average starting conditions for each role and justify the choices.
3. In **Strategy budget**, allocate exactly 100 points across the four transformation strategies.
4. In **Simulation**, select **Initialize**.
5. Use **Step**, **Run 10**, or **Animate to 100**.
6. Apply a staffing shortage or sponsor departure before continuing.
7. Use the **Debrief** tab to discuss the outcome.

## About the smoother animation

The earlier prototype reran the whole Streamlit script after every tick, which can create a visible page blink. This version keeps the animation inside one run and updates persistent placeholders in place. The network positions are also fixed after initialization. That should make the visual evolution noticeably smoother.

Streamlit is still a browser dashboard framework rather than a game engine, so extremely high-frame-rate animation is not the goal. The default speed is intentionally slow enough for a classroom audience to watch state changes.

## Model caveat

The numerical transition rules, role conditions, and strategy effects are pedagogical assumptions. They are intended to support discussion about transformation, not to estimate real-world implementation effects.

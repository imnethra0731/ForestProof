# ForestProof — 4-minute demo

## Before the timer

Open the public Streamlit URL and leave the page loaded. Wake the Render API shortly before the demo by opening `https://<your-render-service>.onrender.com/health` in another tab and confirming `{"status":"ok"}`. The first request to a free Render service can take about a minute after it has spun down. Keep the demo focused on the evidence-review workflow; do not train the optional model live.

## 0:00–0:35 — Problem and scope

“ForestProof is a satellite-based forest disturbance review tool for Floresta Nacional do Jamanxim in Pará, Brazil. It combines Sentinel-2 imagery, NDVI comparisons, multi-date trends, and independent public context so a reviewer can prioritize places to inspect. It produces candidates for human review; it does not prove illegal activity or identify a cause by itself.”

Point out the scope warning at the top of the dashboard.

## 0:35–1:15 — Public data and geography

Show the official ICMBio boundary and the selected analysis box. Explain that the backend retrieves the protected-area boundary from ICMBio's public WFS and searches Sentinel-2 L2A scenes through the Earth Search STAC catalog. Briefly identify the before and after date windows and the cloud metadata filter.

## 1:15–2:15 — Satellite comparison

Use **Load verified Jamanxim historical demo case** if the public INPE PRODES reference endpoint is responding. Otherwise use the preselected small analysis box and date windows already visible. Start the Sentinel-2 analysis and explain the NDVI rule: baseline NDVI at least 0.50, comparison NDVI below 0.50, and a decrease of at least 0.20, after cloud/shadow screening. Describe any result as a candidate vegetation decrease, not a confirmed event. If the live public imagery request is slow, use the already loaded demo case and explain the live catalog dependency rather than claiming a result that did not return.

## 2:15–3:05 — Prioritize and review

Show the severity and inspection-priority indicators, usable-pixel coverage, persistence context, scene dates, and the map/evidence preview. Explain that the priority score helps order field review; it is not a probability of illegality. Open the **Label Review** tab and import the latest real analysis. Show the sample metadata and choose **uncertain** if you have not independently reviewed the imagery. Do not mark a label accepted merely for the demo.

## 3:05–3:35 — Optional ML and evidence

Show the ML readiness/status section and say that Random Forest training is optional and should wait for enough genuinely reviewed labels from independent sites or dates. Show where the evidence report and source links are available. Clarify that INPE PRODES is independent historical context, not ForestProof's classification.

## 3:35–4:00 — Close

“ForestProof turns open satellite observations into a traceable review queue: find a candidate, inspect the evidence, record a human decision, and prioritize follow-up. The next validation step is a documented, independently reviewed local dataset and field checks.”

## Fallback checklist

- If the Streamlit page says the backend is unreachable, check the Streamlit secret `FORESTPROOF_API_URL` and open the Render `/health` URL.
- If Render is waking up, wait for `/health` to return `{"status":"ok"}`, then refresh Streamlit.
- If a satellite provider times out, show the boundary, method, workflow, and documented limitation. Do not fabricate a successful analysis or field-accuracy metric.
- The Render free service has ephemeral storage. Treat saved labels, history, previews, and trained model files as temporary; do not rely on them surviving a restart or sleep.

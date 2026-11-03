# Responsible use and evidence limits

ForestProof is a prototype for prioritizing human review of possible vegetation change in Floresta Nacional do Jamanxim. Its outputs are not proof of illegal activity and do not identify mining, roads, construction, or other causes.

The two-date candidate rule uses NDVI thresholds and Sentinel-2 scene-classification masking. Those thresholds have not been locally calibrated against an independent Jamanxim validation dataset. Cloud and shadow masking can leave too few valid pixels, while fire, flood, seasonal vegetation change, harvest, and other conditions can produce changes that need context.

The multi-date endpoint summarizes mean NDVI and persistence across observations; it does not currently produce a persistent pixel-level change polygon. Inspection priority and severity are triage aids, not probabilities or legal conclusions.

INPE PRODES is independent historical reference information. It is not ForestProof's classification and is not perfect field truth. Human reviewers should inspect imagery and record the source and uncertainty of every label. Keep uncertain or rejected samples out of model training. For ML evaluation, hold out independent sites or dates and report sample counts, class balance, validation method, and metrics only when genuine reviewed labels support them.

A georeferenced UAV GeoTIFF can be checked for CRS, resolution, and footprint overlap. That metadata check does not semantically classify its imagery. Ordinary JPG/PNG uploads are not location-verified evidence.

Do not upload personal or sensitive field information to a public repository. The backend's locally generated history, label records, image previews, uploads, and model files are stored beneath `forestproof_outputs/`; the public repo ignores this folder. On Render's free web service, local files are ephemeral and can be lost on restart, redeploy, or idle spin-down. Use a durable, access-controlled store before relying on this prototype for ongoing operations.

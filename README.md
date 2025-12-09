# Proof Assets for Multimodal Drone Threat Analyzer

This directory contains pre-computed proof assets for the hackathon demo.

## Contents

### Spectrograms (PNG Images)
- `spectrograms/nuclear_breach.png` - Military-grade encrypted RF signal
- `spectrograms/stadium_threat.png` - Consumer drone approaching stadium
- `spectrograms/authorized_film.png` - Professional cinema drone with permit
- `spectrograms/foggy_conflict.png` - Strong RF signal in poor visibility

### Scenario Data (JSON)
- `data/nuclear_breach.json` - Complete data for Nuclear Plant Breach scenario
- `data/stadium_threat.json` - Complete data for Stadium Crowd Safety scenario
- `data/authorized_film.json` - Complete data for Authorized Film Crew scenario
- `data/foggy_conflict.json` - Complete data for Foggy Conflict Resolution scenario
- `data/all_scenarios.json` - Combined data for all scenarios

## Hosting Options

### Option 1: GitHub Pages (Recommended)
1. Create a new GitHub repository (e.g., `drone-threat-assets`)
2. Push this `proof_assets` directory to the repo
3. Enable GitHub Pages in repository settings
4. Your spectrograms will be available at:
   ```
   https://YOUR_USERNAME.github.io/drone-threat-assets/spectrograms/nuclear_breach.png
   ```

### Option 2: Imgur
1. Go to https://imgur.com/upload
2. Upload all 4 spectrogram images
3. Get direct links for each image
4. Update the `spectrogram_url` fields in the JSON files

### Option 3: Google Cloud Storage
1. Create a public GCS bucket
2. Upload spectrograms with public read access
3. Use the public URLs in your AI Studio dashboard

## Next Steps

After hosting the images:
1. Update the `spectrogram_url` fields in `data/all_scenarios.json`
2. Use the updated JSON data in your AI Studio Build dashboard
3. Embed the data as JavaScript constants in your React app

## Usage in AI Studio

```javascript
const SCENARIOS = {
  nuclear_breach: {
    // ... paste data from nuclear_breach.json
    spectrogram_url: "https://YOUR_ACTUAL_URL/nuclear_breach.png"
  },
  // ... other scenarios
}
```

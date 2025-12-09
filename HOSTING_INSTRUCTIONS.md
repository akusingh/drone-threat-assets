# Hosting Instructions for Proof Assets

## Quick Start: GitHub Pages (Recommended - 5 minutes)

### Step 1: Create GitHub Repository
```bash
# Navigate to proof_assets directory
cd proof_assets

# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Add proof assets for drone threat analyzer"
```

### Step 2: Push to GitHub
```bash
# Create a new repository on GitHub named "drone-threat-assets"
# Then run:
git remote add origin https://github.com/YOUR_USERNAME/drone-threat-assets.git
git branch -M main
git push -u origin main
```

### Step 3: Enable GitHub Pages
1. Go to your repository on GitHub
2. Click "Settings" → "Pages"
3. Under "Source", select "main" branch
4. Click "Save"
5. Wait 1-2 minutes for deployment

### Step 4: Get Your URLs
Your assets will be available at:
```
https://YOUR_USERNAME.github.io/drone-threat-assets/spectrograms/nuclear_breach.png
https://YOUR_USERNAME.github.io/drone-threat-assets/spectrograms/stadium_threat.png
https://YOUR_USERNAME.github.io/drone-threat-assets/spectrograms/authorized_film.png
https://YOUR_USERNAME.github.io/drone-threat-assets/spectrograms/foggy_conflict.png
```

### Step 5: Update Scenario Data
Update the `spectrogram_url` fields in `data/all_scenarios.json` with your actual URLs.

---

## Alternative: Imgur (Even Faster - 2 minutes)

### Step 1: Upload to Imgur
1. Go to https://imgur.com/upload
2. Drag and drop all 4 PNG files from `spectrograms/` folder
3. Wait for upload to complete

### Step 2: Get Direct Links
For each image:
1. Right-click on the image
2. Select "Copy image address"
3. The URL will look like: `https://i.imgur.com/XXXXX.png`

### Step 3: Update JSON
Replace the `spectrogram_url` fields in your scenario JSON files with the Imgur URLs.

---

## Alternative: Google Cloud Storage

### Step 1: Create Bucket
```bash
gsutil mb gs://drone-threat-assets
```

### Step 2: Upload Files
```bash
gsutil -m cp spectrograms/*.png gs://drone-threat-assets/spectrograms/
```

### Step 3: Make Public
```bash
gsutil iam ch allUsers:objectViewer gs://drone-threat-assets
```

### Step 4: Get URLs
```
https://storage.googleapis.com/drone-threat-assets/spectrograms/nuclear_breach.png
```

---

## Verification

Test your URLs by opening them in an incognito browser window. They should:
- ✅ Load without authentication
- ✅ Display the spectrogram image
- ✅ Work from any device/location

---

## Next Steps

After hosting:
1. ✅ Update `data/all_scenarios.json` with actual URLs
2. ✅ Test all URLs in incognito mode
3. ✅ Proceed to Phase 2: Build AI Studio Dashboard
4. ✅ Use the scenario data in your React app

---

## Troubleshooting

**GitHub Pages not working?**
- Wait 2-3 minutes after enabling
- Check repository is public
- Verify branch name is correct

**Images not loading?**
- Check CORS settings
- Verify URLs are HTTPS (not HTTP)
- Test in incognito mode

**Need help?**
- GitHub Pages docs: https://pages.github.com/
- Imgur help: https://help.imgur.com/

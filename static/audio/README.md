# Audio files for Symphonia Studio

You can either use **local files** (recommended) or **public URLs** (e.g. Google Drive).

- **Local:** Place MP3s here; set `audio_url` in `data/audio_sources.json` to `/static/audio/filename.mp3`. Most reliable.
- **Public URL:** Set `audio_url` in `data/audio_sources.json` to a public link. For Google Drive use the **direct download** form: `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID` (get the file ID from the “Share” link). The app proxies it so CORS/redirects don’t block playback. If it still doesn’t play (Drive sometimes serves an HTML page), use the local file method below.

**Chopin Nocturne Op. 9 No. 1**
1. Download your MP3 from Google Drive: https://drive.google.com/file/d/1wWcbrh4_sXC8SV84PYqMGeoexXxYyQYY/view?usp=drive_link/preview  
2. Save it in this folder as: **`chopin_nocturne_op9_no1.mp3`**  
3. Restart or refresh the app; the piece will play from the local file.

Add other piece MP3s here and set their paths in `data/audio_sources.json`.

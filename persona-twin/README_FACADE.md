# Persona Twin — FACADE UI + Scrape Integration

This build keeps the existing Persona Twin backend and data integrations while updating the frontend to the FACADE reference UI.

## UI
- FACADE dark sidebar and conversation layout
- Persona selector and new persona flow
- Conversation history, archive/pin/share/delete controls
- Traits modal
- Persona settings/training modal
- Attachment, voice and send controls
- Scrape modal with URL and image modes

## Scraping
- `POST /scrape-url`: fetches a public webpage, discovers images, stores extracted text in the selected persona
- `POST /scrape-photo`: accepts JPG/PNG/WEBP/GIF and uses the configured OpenRouter vision model to extract factual information, then stores it in the persona

## Run
1. Install dependencies: `pip install -r requirements.txt`
2. Configure the same environment variables used by the existing backend.
3. Start: `python main.py`
4. Open `index.html` in the browser, or serve the folder with a local static server.

Do not commit API keys or cookie files to source control.

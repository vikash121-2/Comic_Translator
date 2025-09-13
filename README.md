# Comic Translator Telegram Bot

A Telegram bot that translates text in comic images using OCR and Google Translate.

## Features

- Choose between uploading multiple images or a single ZIP file containing images.
- Extracts text from images using EasyOCR.
- Translates text to English using Google Translate.
- Overlays translated text back onto images.
- Returns a ZIP of translated images and a TXT file of original extracted text.

## Setup

1. Clone or download this repository.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a Telegram bot:
   - Message @BotFather on Telegram.
   - Use `/newbot` to create a new bot and get the token.

4. Set the bot token as an environment variable:
   - On Windows: `set TELEGRAM_BOT_TOKEN=your_token_here`
   - Or create a `.env` file with `TELEGRAM_BOT_TOKEN=your_token_here` and use `python-dotenv` if needed (not included).

5. Run the bot:
   ```
   python app.py
   ```

## Usage

- Start the bot with `/start`.
- Choose "Images" to upload multiple images (send photos or image documents, then `/process`).
- Choose "ZIP" to upload a ZIP file with images (auto-processes).
- The bot will send back a ZIP of translated images and a TXT of original text.

## Requirements

- Python 3.8+
- Internet connection for OCR model download and translations.

## Notes

- The bot uses EasyOCR for text recognition (supports English, Japanese, Simplified Chinese).
- Font for overlaying text is downloaded automatically.
- User data is cleaned up after processing.
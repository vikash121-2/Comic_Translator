import logging
import os
import io
import zipfile
import shutil
import json
import textwrap
import torch
import tempfile
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFile
from pathlib import Path
import numpy as np
import filetype

# --- NEW: IMPORT PYROGRAM ---
from pyrogram import Client

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "ComicNeue-Bold.ttf"

# --- NEW: PYROGRAM API CREDENTIALS ---
# Get these from my.telegram.org
API_ID = 17114587 # <-- IMPORTANT: Replace with your API ID
API_HASH = "b1c07d33747425d84050b68bae6be91f" # <-- IMPORTANT: Replace with your API HASH

# Define the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Define a sub-folder to keep all temporary files organized
TEMP_ROOT_DIR = SCRIPT_DIR / "temp_processing"
# Create this folder if it doesn't exist
TEMP_ROOT_DIR.mkdir(exist_ok=True)

# --- CONVERSATION STATES (Unchanged) ---
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, CHOOSE_LANGUAGE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE,
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE,
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE
) = range(13)

# --- OCR ENGINE SETUP (Unchanged) ---
readers = {}
def get_reader(lang_code):
    global readers
    if (lang_code not in readers):
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR for language: {lang_code}...")
            readers[lang_code] = easyocr.Reader([lang_code, 'en'], gpu=torch.cuda.is_available())
            logger.info(f"EasyOCR for {lang_code} Initialized.")
        except Exception as e:
            logger.critical(f"Could not load easyocr model for {lang_code}. Error: {e}")
            return None
    return readers[lang_code]

# --- HELPER FUNCTIONS ---
def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data and hasattr(context.user_data.get('temp_dir_obj'), 'cleanup'):
        context.user_data['temp_dir_obj'].cleanup()
    context.user_data.clear()

def mask_solid_color_areas(img_cv, blocks_on_image):
    """Masking disabled: return image unchanged."""
    return img_cv


async def send_progress_update(message, current: int, total: int, operation: str):
    """Asynchronously edits a message to show progress."""
    text = f"{operation} in progress... {current}/{total}"
    try:
        if message.text != text:
            await message.edit_text(text)
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.warning(f"Could not edit progress message: {e}")

# --- NEW: PYROGRAM UPLOAD HELPER ---
async def upload_large_file(client: Client, chat_id: int, file_path: str, caption: str, progress_message):
    """Uploads a large file using Pyrogram and updates the progress message."""
    try:
        async def progress(current, total):
            # Pyrogram's progress callback gives bytes, so we convert to percentage
            percentage = int(current * 100 / total)
            # Avoid spamming Telegram APIs by updating only every 5%
            if progress.last_percentage != percentage and percentage % 5 == 0:
                logger.info(f"Uploading {file_path}: {percentage}%")
                await progress_message.edit_text(f"Uploading... {percentage}%")
                progress.last_percentage = percentage

        progress.last_percentage = -1 # Initialize static variable for the callback

        await client.send_document(
            chat_id=chat_id,
            document=file_path,
            caption=caption,
            progress=progress
        )
        logger.info(f"Successfully uploaded {file_path} to {chat_id}")
    except Exception as e:
        logger.error(f"Pyrogram failed to upload {file_path}: {e}")
        await progress_message.edit_text(f"An error occurred during upload: {e}")


def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    # This function is unchanged
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    if not text.strip() or box_width <= 10 or box_height <= 10: return
    font_size = max_font_size
    try: font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logger.error(f"Could not load font: {font_path}")
        draw.text((box[0], box[1]), "[Font Not Found]", fill="red")
        return
    while font_size > 5:
        avg_char_width = font.getlength("a")
        wrap_width = max(1, int(box_width / avg_char_width * 1.8)) if avg_char_width > 0 else 1
        wrapped_text = "\n".join(textwrap.wrap(text, width=wrap_width, break_long_words=True))
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if text_height <= box_height and text_width <= box_width: break
        font_size -= 2
    x, y = box[0] + (box_width - text_width) / 2, box[1] + (box_height - text_height) / 2
    border_color = "white"
    draw.multiline_text((x-1, y-1), wrapped_text, font=font, fill=border_color, align="center", spacing=4)
    draw.multiline_text((x+1, y-1), wrapped_text, font=font, fill=border_color, align="center", spacing=4)
    draw.multiline_text((x-1, y+1), wrapped_text, font=font, fill=border_color, align="center", spacing=4)
    draw.multiline_text((x+1, y+1), wrapped_text, font=font, fill=border_color, align="center", spacing=4)
    draw.multiline_text((x, y), wrapped_text, font=font, fill="black", align="center", spacing=4)


# --- ALL HANDLERS from start() to json_translate_get_json_for_zip() are UNCHANGED ---
# For brevity, I will omit them. They are exactly as you provided.
# ...
# [Paste all your handler functions from 'start' to 'json_translate_get_json_for_zip' here]
# ...

# --- MODIFIED: json_translate_process_zip ---
async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    progress_message = await update.message.reply_text("Zip file received. Applying translations...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await progress_message.edit_text("Error: JSON data was lost.")
        return await back_to_main_menu(update, context)
    if not os.path.exists(FONT_PATH):
        await progress_message.edit_text(f"CRITICAL ERROR: Font file '{FONT_PATH}' not found!")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"; output_dir = Path(temp_dir) / "output"
        input_dir.mkdir(); output_dir.mkdir()
        
        zip_tg_file = await update.message.document.get_file()
        zip_path = input_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(input_dir)
        
        translations_by_file = {}
        for entry in json_data:
            fname = entry['filename']
            if fname not in translations_by_file: translations_by_file[fname] = []
            translations_by_file[fname].append(entry)
            
        all_image_paths = [p for p in input_dir.rglob('*') if p.is_file() and filetype.is_image(p)]
        total_images, processed_count = len(all_image_paths), 0
        
        for img_path in sorted(all_image_paths):
            rel_path_str = str(img_path.relative_to(input_dir)).replace('\\', '/')
            matched_translations = translations_by_file.get(rel_path_str)
            output_img_path = output_dir / rel_path_str
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            if matched_translations:
                img = Image.open(str(img_path)).convert("RGB")
                draw = ImageDraw.Draw(img)
                
                for entry in matched_translations:
                    bbox, translated_text = entry['bbox'], entry.get('translated_text', '').strip()
                    if translated_text:
                        x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                        simple_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        draw.rectangle(simple_box, fill="white")
                        draw_text_in_box(draw, simple_box, translated_text, FONT_PATH)
                img.save(output_img_path)
            else:
                shutil.copy(img_path, output_img_path)
            
            processed_count += 1
            if processed_count % 5 == 0 or processed_count == total_images:
                await send_progress_update(progress_message, processed_count, total_images, "Translation")

        zip_path_str = os.path.join(temp_dir, "final_translated_comics")
        shutil.make_archive(zip_path_str, 'zip', output_dir)
        
        final_zip_path = f"{zip_path_str}.zip"
        await progress_message.edit_text("Processing complete! Now uploading the final zip file...")
        
        # --- MODIFICATION: Use Pyrogram for upload ---
        pyrogram_client = context.application.bot_data['pyrogram_client']
        await upload_large_file(
            client=pyrogram_client,
            chat_id=update.effective_chat.id,
            file_path=final_zip_path,
            caption="Translation complete!",
            progress_message=progress_message
        )
        await progress_message.delete()
    
    cleanup_user_data(context)
    return await start(update, context)

# --- json_divide_menu, json_divide_prompt_json, json_divide_get_json are UNCHANGED ---
# ...
# [Paste your handler functions 'json_divide_menu', 'json_divide_prompt_json', 'json_divide_get_json' here]
# ...

# --- MODIFIED: json_divide_process_zip ---
async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    progress_message = await update.message.reply_text("Zip file received. Dividing JSON...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await progress_message.edit_text("Error: JSON data was lost.")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir) / "work"
        working_dir.mkdir()
        
        zip_tg_file = await update.message.document.get_file()
        zip_path = working_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(working_dir)
        os.remove(zip_path) 
        
        blocks_by_folder = {}
        for entry in json_data:
            p = Path(entry['filename'])
            folder_name = str(p.parent)
            if folder_name not in blocks_by_folder: blocks_by_folder[folder_name] = []
            blocks_by_folder[folder_name].append(entry)
        
        for folder_rel_path, blocks in blocks_by_folder.items():
            folder_abs_path = working_dir / Path(folder_rel_path)
            if not folder_abs_path.is_dir():
                continue
            folder_json_path = folder_abs_path / "folder_text.json"
            with open(folder_json_path, 'w', encoding='utf-8') as f:
                json.dump(blocks, f, ensure_ascii=False, indent=4)

        zip_path_str = os.path.join(temp_dir, "final_divided_comics")
        shutil.make_archive(zip_path_str, 'zip', working_dir)
        
        final_zip_path = f"{zip_path_str}.zip"
        await progress_message.edit_text("Dividing complete! Now uploading the final zip file...")
        
        # --- MODIFICATION: Use Pyrogram for upload ---
        pyrogram_client = context.application.bot_data['pyrogram_client']
        await upload_large_file(
            client=pyrogram_client,
            chat_id=update.effective_chat.id,
            file_path=final_zip_path,
            caption="Dividing complete!",
            progress_message=progress_message
        )
        await progress_message.delete()
    
    cleanup_user_data(context)
    return await start(update, context)

# --- Error handler is UNCHANGED ---
# ...
# [Paste your error_handler function here]
# ...

# --- MODIFIED: APPLICATION SETUP ---
def main() -> None:
    """Start the bot."""
    
    # --- NEW: Initialize Pyrogram Client ---
    # The 'bot_session' name can be anything. It's the file where Pyrogram stores its session data.
    # We use in_memory=True to avoid creating a file, but you can remove it if you want the session to persist.
    pyrogram_client = Client(
        "bot_session", 
        api_id=API_ID, 
        api_hash=API_HASH, 
        bot_token=BOT_TOKEN,
        in_memory=True
    )
    
    # --- MODIFIED: Use Application.builder() to store the client ---
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Store the client instance in bot_data to access it from handlers
    application.bot_data['pyrogram_client'] = pyrogram_client

    # Register a global error handler
    application.add_error_handler(error_handler)
    
    # --- Conversation Handler is UNCHANGED ---
    conv_handler = ConversationHandler(
        # ... Your full ConversationHandler definition here ...
    )
    application.add_handler(conv_handler)
    
    # --- MODIFIED: Start and stop both frameworks ---
    logger.info("Starting polling for PTB...")
    application.run_polling()

async def async_main():
    """Asynchronous main function to handle both PTB and Pyrogram."""
    
    # Initialize Pyrogram Client
    pyrogram_client = Client(
        "bot_session",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN
    )
    
    # Initialize PTB Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Store the client instance in bot_data
    application.bot_data['pyrogram_client'] = pyrogram_client

    # Add error handler
    application.add_error_handler(error_handler)

    # Add conversation handler (exactly as you defined it)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$"),
                CallbackQueryHandler(json_divide_menu, pattern="^main_divide$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"), 
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt_language, pattern="^jm_image$"),
                CallbackQueryHandler(json_maker_prompt_language, pattern="^jm_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            CHOOSE_LANGUAGE: [
                CallbackQueryHandler(json_maker_prompt_files, pattern="^lang_"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^process_images$"),
            ],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json_for_img, pattern="^jt_image$"),
                CallbackQueryHandler(json_translate_prompt_json_for_zip, pattern="^jt_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_JSON_TRANSLATE_IMG: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_img)],
            WAITING_IMAGES_TRANSLATE: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, json_translate_collect_images),
                CallbackQueryHandler(json_translate_process_images, pattern="^jt_process_images$"),
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_zip)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
            JSON_DIVIDE_CHOICE: [
                CallbackQueryHandler(json_divide_prompt_json, pattern="^jd_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$")
            ],
            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.FileExtension("json"), json_divide_get_json)],
            WAITING_ZIP_DIVIDE: [MessageHandler(filters.Document.ZIP, json_divide_process_zip)],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    
    # Run both concurrently
    async with application:
        await application.start()
        await application.updater.start_polling()
        async with pyrogram_client:
            logger.info("Pyrogram client started.")
            await application.updater.stop()
            await application.stop()

if __name__ == "__main__":
    # --- Use asyncio.run for the new async main ---
    import asyncio
    asyncio.run(async_main())

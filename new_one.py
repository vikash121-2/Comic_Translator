import logging
import os
import zipfile
import shutil
import json
import torch
import tempfile
import traceback
import asyncio
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont, ImageFile

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory


# ADDED new states for the new feature
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, 
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, 
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(12)


# ... (easyocr model loading and helper functions like get_ocr_results, cleanup_user_data remain the same) ...
# --- Load easyocr models ---
logger.info(f"Loading all easyocr models...")
try:
    import easyocr
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")
    reader_ja = easyocr.Reader(['ja', 'en'], gpu=use_gpu)
    reader_ko = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
    reader_sim = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
    reader_tra = easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu)
    logger.info("All easyocr models loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load easyocr model. Error: {e}")
    logger.critical(traceback.format_exc())
    exit(1)

# --- Main Menu & Core Navigation ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # This is the full main menu
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = "Welcome! This bot uses easyocr. Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    elif update.callback_query:
        query = update.callback_query
        try: await query.answer()
        except BadRequest: logger.info("Callback query already answered.")
        await query.edit_message_text(message_text, reply_markup=reply_markup)
    return MAIN_MENU

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cleanup_user_data(context)
    return await start(update, context)

# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    # ADDED back the Image Upload button
    keyboard = [
        [
            InlineKeyboardButton("🖼️ Image Upload", callback_data="jt_image"),
            InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip")
        ],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Choose an option for translation:", reply_markup=reply_markup)
    return JSON_TRANSLATE_CHOICE

# --- NEW functions for the Image Upload flow ---

async def json_translate_prompt_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks for the JSON file first."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("First, please upload the JSON file that contains the translated text for your images.")
    return WAITING_JSON_TRANSLATE_IMG

async def json_translate_get_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the JSON, stores it, and asks for the images."""
    json_file = await update.message.document.get_file()
    json_bytes = await json_file.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes)
    
    # Set up for image collection
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['received_images'] = {} # Using a dict to store file_id and path
    
    await update.message.reply_text("JSON received. Now, please send the corresponding images one by one. Press 'Done' when finished.")
    return WAITING_IMAGES_TRANSLATE

async def json_translate_collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Collects images for the translation task."""
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        received_images = context.user_data['received_images']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over with /start.")
        cleanup_user_data(context)
        return ConversationHandler.END

    file_id = None
    file_to_download = None
    original_filename = None

    if update.message.photo:
        photo = update.message.photo[-1]
        file_id = photo.file_id
        file_to_download = await photo.get_file()
        original_filename = f"{file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        doc = update.message.document
        file_id = doc.file_id
        file_to_download = await doc.get_file()
        original_filename = doc.file_name
    else:
        return WAITING_IMAGES_TRANSLATE # Ignore non-image messages

    file_path = os.path.join(temp_dir_path, original_filename)
    await file_to_download.download_to_drive(file_path)
    # Store the downloaded path and the original filename for lookup
    received_images[original_filename] = file_path

    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jt_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Image '{original_filename}' received. Send another, or press Done.", reply_markup=reply_markup)
    return WAITING_IMAGES_TRANSLATE

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Draws text on all collected images and sends them back."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing translation for all images...")

    json_data = context.user_data.get('json_data', {})
    received_images = context.user_data.get('received_images', {})
    
    if not received_images:
        await query.edit_message_text("You didn't send any images! Please start over.")
        cleanup_user_data(context)
        return await start(update, context)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await context.bot.send_message(chat_id=query.message.chat.id, text=f"⚠️ Error: Font file '{FONT_PATH}' not found.")
        cleanup_user_data(context)
        return await start(update, context)

    # Find the 'images' list in the JSON
    images_json_data = json_data.get("images", [])
    
    for image_info in images_json_data:
        image_name = image_info.get("image_name")
        if image_name in received_images:
            image_path = received_images[image_name]
            with Image.open(image_path).convert("RGBA") as img:
                txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(txt_layer)
                
                for block in image_info.get("text_blocks", []):
                    text = block["text"]
                    loc = block["location"]
                    # White rectangle fill behind the text
                    draw.rectangle(loc, fill=(255, 255, 255, 255))
                    draw.text((loc[0], loc[1]), text, font=font, fill=(0, 0, 0, 255))
                
                # Composite the text layer onto the original image
                combined_img = Image.alpha_composite(img, txt_layer).convert("RGB")
                
                # Send the processed image
                bio = io.BytesIO()
                bio.name = f"translated_{image_name}"
                combined_img.save(bio, 'JPEG')
                bio.seek(0)
                await context.bot.send_document(chat_id=query.message.chat.id, document=bio)

    await context.bot.send_message(chat_id=query.message.chat.id, text="Translation complete!")
    cleanup_user_data(context)
    return await start(update, context)


# --- The rest of your script (Json Maker, Json Divide, main function) goes here ---
# Make sure to add the new states to your main() function's ConversationHandler

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$"),
                # ... other handlers like json_divide_menu
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"), 
            ],
            # ... JSON Maker States ...
            JSON_MAKER_CHOICE: [
                # ...
            ],
            # --- NEW States for JSON Translate Image Flow ---
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json_for_img, pattern="^jt_image$"),
                # ... other handlers like jt_zip
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$")
            ],
            WAITING_JSON_TRANSLATE_IMG: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_img)],
            WAITING_IMAGES_TRANSLATE: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, json_translate_collect_images),
                CallbackQueryHandler(json_translate_process_images, pattern="^jt_process_images$"),
            ],
            # ... other states for Zip Translate and Divide ...
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling()


# You will need to re-integrate the other complete functions (get_ocr_results, Json Maker, Json Divide, etc.)
# into the final script. This response focuses on adding the new requested feature.
if __name__ == "__main__":
    # This is a partial script. You need to combine it with the complete functions
    # from the previous version to make it fully functional.
    # For example, you must add back the full 'Json Maker' and 'Json Divide' functions and states.
    pass

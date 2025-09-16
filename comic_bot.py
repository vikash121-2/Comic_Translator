# Add these 4 lines to the very top of your script
import sys
import site

# This forces Python to look in the user's local installation directory.
if site.USER_SITE not in sys.path:
    sys.path.insert(0, site.USER_SITE)
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

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global Variables & Configuration ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory


# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, #... add more if needed
) = range(3)

# --- NEW: Load Surya OCR model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading Surya OCR model onto {DEVICE}...")
try:
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition import vit
    
    det_processor, det_model = segformer.load_processor_and_model()
    rec_processor, rec_model = vit.load_processor_and_model()
    logger.info("Surya OCR model loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load Surya OCR model. Error: {e}")
    exit(1)

# --- Helper & Utility Functions ---

def get_ocr_results(image_paths: List[str]) -> Dict:
    """Runs OCR on a list of images using Surya OCR."""
    results = {}
    try:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # Using English as a default, this can be made configurable again if needed
        predictions = run_ocr(images, [['en']] * len(images), det_model, det_processor, rec_model, rec_processor)

        for i, (pred, image_path) in enumerate(zip(predictions, image_paths)):
            image_name = os.path.basename(image_path)
            text_blocks = []
            if pred is not None:
                for line in pred.text_lines:
                    text_blocks.append({
                        "text": line.text,
                        "location": line.bbox
                    })
            results[image_name] = text_blocks
            logger.info(f"Successfully processed {image_name}")
            
    except Exception as e:
        logger.error(f"Failed during Surya OCR processing: {e}")
        logger.error(traceback.format_exc())
        # On total failure, create empty results for all images
        for image_path in image_paths:
            results[os.path.basename(image_path)] = []
    
    return results

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir' in context.user_data:
        context.user_data['temp_dir'].cleanup()
        del context.user_data['temp_dir']
    context.user_data.pop('image_paths', None)

# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        # Add other buttons here as you build them out
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message_text = "Welcome! Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    elif update.callback_query:
        query = update.callback_query
        try:
            await query.answer()
        except BadRequest:
            logger.info("Callback query already answered.")
        if query.message.text != message_text or query.message.reply_markup != reply_markup:
            await query.edit_message_text(message_text, reply_markup=reply_markup)
    return MAIN_MENU

# --- 1. Json Maker Feature ---

async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("🖼️ Image Upload", callback_data="jm_image")],
        [InlineKeyboardButton("🗂️ Zip Upload (WIP)", callback_data="jm_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("How would you like to provide source files?", reply_markup=reply_markup)
    return JSON_MAKER_CHOICE

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['temp_dir'] = tempfile.TemporaryDirectory()
    context.user_data['image_paths'] = []
    await query.answer()
    await query.edit_message_text("Please send your images. Press 'Done Uploading' when finished.")
    return WAITING_IMAGES_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # This function is unchanged
    try:
        temp_dir_path = context.user_data['temp_dir'].name
        image_paths = context.user_data['image_paths']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over with /start.")
        cleanup_user_data(context)
        return ConversationHandler.END

    file_to_download, file_name = (None, None)
    if update.message.photo:
        photo = update.message.photo[-1]
        file_to_download = await photo.get_file()
        file_name = f"{photo.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        doc = update.message.document
        file_to_download = await doc.get_file()
        file_name = doc.file_name
    else:
        await update.message.reply_text("That doesn't seem to be an image. Please send a photo or image file.")
        return WAITING_IMAGES_OCR

    file_path = os.path.join(temp_dir_path, file_name)
    await file_to_download.download_to_drive(file_path)
    image_paths.append(file_path)

    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jm_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Image {len(image_paths)} received. Send another, or press Done.",
        reply_markup=reply_markup
    )
    return WAITING_IMAGES_OCR

async def process_collected_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing images with Surya OCR...")
    
    image_paths = context.user_data.get('image_paths', [])
    if not image_paths:
        await query.edit_message_text("You didn't send any images! Returning to menu.")
        cleanup_user_data(context)
        return await start(update, context)

    ocr_data = get_ocr_results(image_paths)
    final_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
    
    total_text_blocks = sum(len(img["text_blocks"]) for img in final_json["images"])
    if total_text_blocks == 0:
        await query.edit_message_text(
            "I couldn't extract any text from the image(s). The image might be blurry or the text too stylized. Returning to menu."
        )
        cleanup_user_data(context)
        await asyncio.sleep(4)
        return await start(update, context)

    temp_dir_path = context.user_data['temp_dir'].name
    json_path = os.path.join(temp_dir_path, "extracted_text.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
        
    await context.bot.send_document(chat_id=query.message.chat.id, document=open(json_path, 'rb'))
    
    cleanup_user_data(context)
    return await start(update, context)

async def not_implemented_yet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer("This feature is a work in progress.", show_alert=True)
    return MAIN_MENU

# --- Main Application Setup ---

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$"), 
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt_image, pattern="^jm_image$"),
                CallbackQueryHandler(not_implemented_yet, pattern="^jm_zip$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^jm_process_images$"),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()

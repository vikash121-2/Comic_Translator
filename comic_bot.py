# No special sys.path modification is needed for this version

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
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR,
) = range(3)

# --- NEW: Load easyocr model with correct languages and GPU support ---
logger.info(f"Loading easyocr model...")
try:
    import easyocr
    # This will download the models on the first run.
    # We are using Simplified Chinese and English.
    # gpu=True requires the Colab runtime to be set to GPU.
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu) 
    logger.info("easyocr model loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load easyocr model. Error: {e}")
    logger.critical(traceback.format_exc())
    exit(1)

# --- Helper & Utility Functions ---

def get_ocr_results(image_paths: List[str]) -> Dict:
    """Runs OCR on a list of images using easyocr."""
    results = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            # easyocr reads the path directly
            ocr_output = reader.readtext(image_path)
            
            text_blocks = []
            for (bbox, text, prob) in ocr_output:
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # We need to convert it to a simple [xmin, ymin, xmax, ymax]
                x_coords = [int(p[0]) for p in bbox]
                y_coords = [int(p[1]) for p in bbox]
                
                simple_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                
                text_blocks.append({
                    "text": text,
                    "location": simple_bbox
                })

            results[image_name] = text_blocks
            logger.info(f"Successfully processed {image_name} with easyocr.")
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path} with easyocr: {e}")
            logger.error(traceback.format_exc())
            results[image_name] = []
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
    
    message_text = "Welcome! This bot uses easyocr. Please choose an option:"
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
    await query.edit_message_text("Processing images with easyocr...")
    
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
            "I couldn't extract any text from the image(s). The image might be blurry or contain no recognizable text. Returning to menu."
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

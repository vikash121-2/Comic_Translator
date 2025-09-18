import logging
import os
import zipfile
import shutil
import json
import torch
import tempfile
import traceback
import asyncio
import cv2
import numpy as np
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


# ADDED new state for language selection
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, SELECT_LANGUAGE,
) = range(4)

# --- Load easyocr models into a dictionary ---
logger.info(f"Loading all easyocr models...")
try:
    import easyocr
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")

    # NEW: Store readers in a dictionary for easy access
    readers = {
        'ja': easyocr.Reader(['ja', 'en'], gpu=use_gpu),
        'ko': easyocr.Reader(['ko', 'en'], gpu=use_gpu),
        'ch_sim': easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu),
        'ch_tra': easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu),
    }
    
    logger.info("All easyocr models loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load easyocr model. Error: {e}")
    logger.critical(traceback.format_exc())
    exit(1)

# --- Helper & Utility Functions ---

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    except Exception:
        return cv2.imread(image_path)

def get_ocr_results(image_paths: List[str], lang_code: str) -> Dict:
    """Runs OCR on a list of images using a SINGLE, SPECIFIED language reader."""
    results = {}
    # Select the correct reader from our dictionary
    reader = readers.get(lang_code)
    if not reader:
        logger.error(f"Invalid language code '{lang_code}' passed to get_ocr_results.")
        return {}

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            preprocessed_image = preprocess_for_ocr(image_path)
            ocr_output = reader.readtext(preprocessed_image, detail=1, paragraph=False)
            
            text_blocks = []
            for (bbox, text, prob) in ocr_output:
                x_coords = [int(p[0]) for p in bbox]
                y_coords = [int(p[1]) for p in bbox]
                simple_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                text_blocks.append({"text": text, "location": simple_bbox})

            results[image_name] = text_blocks
            logger.info(f"Successfully processed {image_name} with '{lang_code}' model.")
        except Exception as e:
            logger.error(f"Failed to process image {image_path} with easyocr: {e}")
            logger.error(traceback.format_exc())
            results[image_name] = []
    return results

def sort_text_blocks(ocr_data: Dict) -> Dict:
    sorted_data = {"images": []}
    for image_info in ocr_data["images"]:
        sorted_blocks = sorted(image_info["text_blocks"], key=lambda b: (b["location"][1], b["location"][0]))
        sorted_data["images"].append({"image_name": image_info["image_name"], "text_blocks": sorted_blocks})
    return sorted_data

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data:
        context.user_data['temp_dir_obj'].cleanup()
        del context.user_data['temp_dir_obj']
    context.user_data.pop('image_paths', None)

# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [[InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")]]
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
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['image_paths'] = []
    await query.answer()
    await query.edit_message_text("Please send your images. Press 'Done Uploading' when finished.")
    return WAITING_IMAGES_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
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

    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jm_prompt_language")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Image {len(image_paths)} received. Send another, or press Done.",
        reply_markup=reply_markup
    )
    return WAITING_IMAGES_OCR

async def prompt_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to select the language of the document."""
    query = update.callback_query
    keyboard = [
        [
            InlineKeyboardButton("Japanese", callback_data="lang_ja"),
            InlineKeyboardButton("Korean", callback_data="lang_ko"),
        ],
        [
            InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"),
            InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Please select the primary language of the image(s):", reply_markup=reply_markup)
    return SELECT_LANGUAGE

async def process_images_with_selected_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes images using the language the user selected."""
    query = update.callback_query
    lang_code = query.data.split('_', 1)[1] # e.g., 'lang_ko' -> 'ko'
    await query.answer()
    await query.edit_message_text(f"Processing images with {lang_code.upper()} model...")
    
    image_paths = context.user_data.get('image_paths', [])
    if not image_paths:
        await query.edit_message_text("Error: Lost track of images. Please start over.")
        cleanup_user_data(context)
        return await start(update, context)

    ocr_data = get_ocr_results(image_paths, lang_code)
    raw_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
    final_json = sort_text_blocks(raw_json)
    
    if sum(len(img["text_blocks"]) for img in final_json["images"]) == 0:
        await query.edit_message_text("I couldn't extract any text. Returning to menu.")
        cleanup_user_data(context)
        await asyncio.sleep(4)
        return await start(update, context)

    temp_dir_path = context.user_data['temp_dir_obj'].name
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
                CallbackQueryHandler(prompt_language, pattern="^jm_prompt_language$"),
            ],
            SELECT_LANGUAGE: [
                CallbackQueryHandler(process_images_with_selected_language, pattern="^lang_"),
            ]
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()

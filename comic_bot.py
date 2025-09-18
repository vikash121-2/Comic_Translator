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
from typing import List, Dict, Tuple

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

(MAIN_MENU, JSON_MAKER_CHOICE, WAITING_IMAGES_OCR) = range(3)

logger.info(f"Loading all easyocr models...")
try:
    import easyocr
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")

    reader_ja = easyocr.Reader(['ja', 'en'], gpu=use_gpu)
    logger.info("Loaded Japanese model.")
    reader_ko = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
    logger.info("Loaded Korean model.")
    reader_sim = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
    logger.info("Loaded Simplified Chinese model.")
    reader_tra = easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu)
    logger.info("Loaded Traditional Chinese model.")
    
    logger.info("All easyocr models loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load easyocr model. Error: {e}")
    logger.critical(traceback.format_exc())
    exit(1)

# --- Helper & Utility Functions ---

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """
    Refined preprocessing: Grayscale + mild noise reduction or sharpening.
    """
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian blur to reduce noise, then an unsharp mask to sharpen edges
        # This can help emphasize text edges without harsh thresholding
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # Alternatively, for very dark/light text, you might try a local contrast stretch
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # enhanced_image = clahe.apply(gray)
        
        return sharpened # Or gray, or enhanced_image, depending on test results
    except Exception as e:
        logger.error(f"OpenCV preprocessing failed for {image_path}: {e}")
        return cv2.imread(image_path)

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = float(box1_area + box2_area - intersection_area)
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def get_ocr_results(image_paths: List[str]) -> Dict:
    results = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            preprocessed_image = preprocess_for_ocr(image_path)
            
            # Refined parameters for easyocr to catch more text
            read_params = {
                'detail': 1,
                'contrast_ths': 0.05, # Lower threshold to detect lower contrast text
                'adjust_contrast': 0.5, # Moderate contrast adjustment
                'text_threshold': 0.6, # Slightly higher confidence for initial detection
                'low_text': 0.3, # Adjust sensitivity for low confidence text
                'link_threshold': 0.8 # Higher link threshold for connecting words
            }
            
            output_ja = reader_ja.readtext(preprocessed_image, **read_params)
            output_ko = reader_ko.readtext(preprocessed_image, **read_params)
            output_sim = reader_sim.readtext(preprocessed_image, **read_params)
            output_tra = reader_tra.readtext(preprocessed_image, **read_params)
            combined_output = output_ja + output_ko + output_sim + output_tra
            
            final_text_blocks = []
            
            # ADVANCED DE-DUPLICATION using IoU and text similarity
            for (bbox_new, text_new, prob_new) in combined_output:
                new_block_added = True
                simple_bbox_new = [int(p[0]) for p in bbox_new[0]] + [int(p[1]) for p in bbox_new[0]] # Placeholder for simple conversion
                
                # Correct simple_bbox_new conversion for 4 points
                x_coords_new = [int(p[0]) for p in bbox_new]
                y_coords_new = [int(p[1]) for p in bbox_new]
                simple_bbox_new = [min(x_coords_new), min(y_coords_new), max(x_coords_new), max(y_coords_new)]


                for i, existing_block in enumerate(final_text_blocks):
                    bbox_existing = existing_block["location"]
                    text_existing = existing_block["text"]
                    
                    iou = calculate_iou(simple_bbox_new, bbox_existing)
                    
                    # Simple text similarity check (could be improved with Levenshtein distance for robustness)
                    text_similarity = 0.0
                    if len(text_new) > 0 and len(text_existing) > 0:
                        # Case-insensitive comparison for similarity
                        min_len = min(len(text_new), len(text_existing))
                        matches = sum(1 for a, b in zip(text_new.lower(), text_existing.lower()) if a == b)
                        if min_len > 0:
                            text_similarity = matches / max(len(text_new), len(text_existing))

                    # If high overlap AND similar text, consider it a duplicate
                    if iou > 0.6 and text_similarity > 0.7: # Tunable thresholds
                        # Keep the one with higher confidence (if prob is comparable, otherwise prioritize one language)
                        # For simplicity, we'll just skip adding the new one if a very similar one exists
                        new_block_added = False
                        break
                
                if new_block_added:
                    final_text_blocks.append({
                        "text": text_new,
                        "location": simple_bbox_new
                    })

            results[image_name] = final_text_blocks
            logger.info(f"Successfully processed {image_name} with all easyocr models.")
        except Exception as e:
            logger.error(f"Failed to process image {image_path} with easyocr: {e}")
            logger.error(traceback.format_exc())
            results[image_name] = []
    return results


def sort_text_blocks(ocr_data: Dict) -> Dict:
    sorted_data = {"images": []}
    for image_info in ocr_data["images"]:
        # Sort blocks based on their top-left corner (y-coordinate, then x-coordinate)
        sorted_blocks = sorted(image_info["text_blocks"], key=lambda b: (b["location"][1], b["location"][0]))
        sorted_data["images"].append({
            "image_name": image_info["image_name"],
            "text_blocks": sorted_blocks
        })
    return sorted_data

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir' in context.user_data:
        context.user_data['temp_dir'].cleanup()
        del context.user_data['temp_dir']
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
    await query.edit_message_text("Preprocessing and processing images with all models...")
    
    image_paths = context.user_data.get('image_paths', [])
    if not image_paths:
        await query.edit_message_text("You didn't send any images! Returning to menu.")
        cleanup_user_data(context)
        return await start(update, context)

    ocr_data = get_ocr_results(image_paths)
    raw_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
    
    final_json = sort_text_blocks(raw_json)
    
    total_text_blocks = sum(len(img["text_blocks"]) for img in final_json["images"])
    if total_text_blocks == 0:
        await query.edit_message_text(
            "I couldn't extract any text from the image(s). Returning to menu."
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

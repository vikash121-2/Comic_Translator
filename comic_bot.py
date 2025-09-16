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
MODEL_ID = "microsoft/Florence-2-large"

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, #... add more if needed
) = range(3)


# --- Load Florence-2 model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading Florence-2 model ({MODEL_ID}) onto {DEVICE}...")
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    logger.info("Florence-2 model loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load model. Error: {e}")
    exit(1)


# --- Helper & Utility Functions ---

def resize_if_needed(image: Image.Image, min_side_length: int = 320) -> Image.Image:
    """
    Resizes an image if its smallest dimension is less than a minimum value.
    Preserves the aspect ratio.
    """
    width, height = image.size
    if width < min_side_length or height < min_side_length:
        logger.info(f"Image is too small ({width}x{height}). Resizing to have min side {min_side_length}.")
        if width < height:
            new_width = min_side_length
            new_height = int(height * (new_width / width))
        else:
            new_height = min_side_length
            new_width = int(width * (new_height / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def get_ocr_results(image_paths: List[str]) -> Dict:
    results = {}
    task_prompt = "<OCR_WITH_REGION>"
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            
            # --- NEW: Resize the image if it's too small ---
            image = resize_if_needed(image)

            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=2048, # Increased for potentially larger images
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=task_prompt, image_sizes=[image.size]
            )
            ocr_data = parsed_answer[task_prompt]
            text_blocks = [{"text": label, "location": bbox} for bbox, label in zip(ocr_data['bboxes'], ocr_data['labels'])]
            results[image_name] = text_blocks
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
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
        [InlineKeyboardButton("🎨 json To Comic translate (WIP)", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide (WIP)", callback_data="main_divide")],
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
    await query.edit_message_text("Processing images...")
    
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
            "I couldn't extract any text from the image(s). This can happen with unsupported formats (try .jpg/.png) or files with no text. Returning to menu."
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
    await query.answer("This feature is a work in progress and not yet implemented.", show_alert=True)
    return MAIN_MENU

# --- Main Application Setup ---

def main() -> None:
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Error: Please replace 'YOUR_BOT_TOKEN_HERE' with your bot token.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(not_implemented_yet, pattern="^main_translate$"),
                CallbackQueryHandler(not_implemented_yet, pattern="^main_divide$"),
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

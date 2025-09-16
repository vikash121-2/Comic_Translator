import logging
import os
import zipfile
import shutil
import json
import torch
import tempfile
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont

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

# --- Transformers & Florence-2 OCR Initialization ---
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    print("Error: Transformers library not found.")
    print("Please install it with: pip install transformers accelerate")
    exit(1)

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global Variables & Configuration ---
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
FONT_PATH = "font.ttf"

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(10)

# --- Load Florence-2 model and processor ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"  # <-- ADD THIS LINE

logger.info(f"Loading Florence-2 model ({MODEL_ID}) onto {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info("Florence-2 model loaded successfully.")


# --- Helper Function for OCR ---
def get_ocr_results(image_paths: List[str]) -> Dict:
    results = {}
    task_prompt = "<OCR_WITH_REGION>"
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_name = os.path.basename(image_path)
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
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
            logger.info(f"Successfully processed {image_name}")
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            results[os.path.basename(image_path)] = []
    return results

# --- Bot Cleanup and Menu Handlers ---

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    """Clean up temporary directories and data for a user."""
    if 'temp_dir' in context.user_data:
        context.user_data['temp_dir'].cleanup()
        del context.user_data['temp_dir']
    
    # Clean up other potential keys
    context.user_data.pop('image_paths', None)
    context.user_data.pop('json_data', None)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the main menu using inline buttons."""
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
        [InlineKeyboardButton("🌐 Choose ocr language", callback_data="main_language")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message_text = "Welcome! Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    elif update.callback_query:
        query = update.callback_query
        await query.answer()
        # Check if the message text is already the menu text to avoid unnecessary edits
        if query.message.text != message_text:
            await query.edit_message_text(message_text, reply_markup=reply_markup)

    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Operation cancelled.")
    cleanup_user_data(context)
    return ConversationHandler.END


# --- 1. JSON Maker ---

async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [
            InlineKeyboardButton("🖼️ Image Upload", callback_data="jm_image"),
            InlineKeyboardButton("🗂️ Zip Upload", callback_data="jm_zip"),
        ],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("How would you like to provide the source files?", reply_markup=reply_markup)
    return JSON_MAKER_CHOICE

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Prompt for image upload and set up collectors."""
    query = update.callback_query
    # Create a persistent temporary directory for this user's session
    context.user_data['temp_dir'] = tempfile.TemporaryDirectory()
    context.user_data['image_paths'] = []
    
    await query.answer()
    await query.edit_message_text("Please send your images one by one. Press 'Done Uploading' when you are finished.")
    return WAITING_IMAGES_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives an image, saves it, and asks for more or to finish."""
    try:
        temp_dir_path = context.user_data['temp_dir'].name
        image_paths = context.user_data['image_paths']
    except KeyError:
        # Handle case where the user sends an image unexpectedly
        await update.message.reply_text("Something went wrong. Please start over with /start.")
        return ConversationHandler.END

    photo_file = await update.message.photo[-1].get_file()
    file_path = os.path.join(temp_dir_path, f"{photo_file.file_id}.jpg")
    await photo_file.download_to_drive(file_path)
    image_paths.append(file_path)

    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jm_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Image {len(image_paths)} received. Send another, or press Done.",
        reply_markup=reply_markup
    )
    return WAITING_IMAGES_OCR


async def process_collected_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes all collected images, sends JSON, and returns to the main menu."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing images with Florence-2... This may take a moment.")
    
    image_paths = context.user_data.get('image_paths', [])
    if not image_paths:
        await query.edit_message_text("You didn't send any images! Please start over.", reply_markup=None)
        cleanup_user_data(context)
        return await start(update, context)

    ocr_data = get_ocr_results(image_paths)
    final_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
    
    # Use the same temp dir to create the json file
    temp_dir_path = context.user_data['temp_dir'].name
    json_path = os.path.join(temp_dir_path, "extracted_text.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
        
    await context.bot.send_document(chat_id=query.effective_chat.id, document=open(json_path, 'rb'))
    
    cleanup_user_data(context)
    return await start(update, context)


async def json_maker_prompt_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please send the zip file.")
    return WAITING_ZIP_OCR


# ... Other functions (zip processing, translate, divide) would follow a similar pattern ...
# They should return `await start(update, context)` after completion.


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
                # Add other main menu handlers here
                CallbackQueryHandler(start, pattern="^main_menu_start$"), 
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt_image, pattern="^jm_image$"),
                CallbackQueryHandler(json_maker_prompt_zip, pattern="^jm_zip$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^jm_process_images$"),
            ],
            WAITING_ZIP_OCR: [
                # Add your zip handler here, ensuring it calls start() at the end
                # e.g., MessageHandler(filters.Document.ZIP, json_maker_process_zip)
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()

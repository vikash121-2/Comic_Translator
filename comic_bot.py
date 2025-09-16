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
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(10)

# --- Load Florence-2 model and processor ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"

logger.info(f"Loading Florence-2 model ({MODEL_ID}) onto {DEVICE}...")
# Use attn_implementation="eager" to prevent potential errors in some environments
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

# --- Bot Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the main menu using inline buttons."""
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
        [InlineKeyboardButton("🌐 Choose ocr language", callback_data="main_language")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # If the command is triggered by a user message (e.g., /start)
    if update.message:
        await update.message.reply_text("Welcome! Please choose an option:", reply_markup=reply_markup)
    # If it's triggered by a callback (e.g., returning to the menu)
    elif update.callback_query:
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("Done! What would you like to do next?", reply_markup=reply_markup)

    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Operation cancelled.")
    # Clean up any stored data
    for key in ['media_group', 'json_data']:
        if key in context.user_data:
            del context.user_data[key]
    return ConversationHandler.END

# --- 1. JSON Maker ---
async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the JSON Maker sub-menu with inline buttons."""
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

async def json_maker_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic prompt for uploading a file."""
    query = update.callback_query
    file_type = query.data.split('_')[1] # 'image' or 'zip'
    await query.answer()
    await query.edit_message_text(f"Please send the {file_type} file(s).")
    return WAITING_IMAGES_OCR if file_type == 'image' else WAITING_ZIP_OCR

async def json_maker_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes images, sends JSON, and returns to the main menu."""
    await update.message.reply_text("Processing images with Florence-2... This may take a moment.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        photo_file = await update.message.photo[-1].get_file()
        file_path = os.path.join(temp_dir, f"{photo_file.file_id}.jpg")
        await photo_file.download_to_drive(file_path)
        image_paths.append(file_path)

        ocr_data = get_ocr_results(image_paths)
        final_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
        
        json_path = os.path.join(temp_dir, "extracted_text.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
            
        await update.message.reply_document(document=open(json_path, 'rb'))
    
    # Loop back to the main menu
    return await start(update, context)

async def json_maker_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # This function is long and remains largely the same, just the end is changed.
    zip_file = await update.message.document.get_file()
    await update.message.reply_text("Processing zip file...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # ... (same zip processing logic as before) ...
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        final_json = {"folders": []}
        for folder_name in sorted(os.listdir(extract_path)):
            folder_path = os.path.join(extract_path, folder_name)
            if os.path.isdir(folder_path):
                image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_paths:
                    ocr_data = get_ocr_results(image_paths)
                    images_data = [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]
                    final_json["folders"].append({"folder_name": folder_name, "images": images_data})
        
        json_path = os.path.join(temp_dir, "extracted_text_from_zip.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        await update.message.reply_document(document=open(json_path, 'rb'))

    # Loop back to the main menu
    return await start(update, context)

# --- 2. JSON to Comic Translate ---
async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [
            InlineKeyboardButton("🖼️ Image Upload", callback_data="jt_image"), # Placeholder
            InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip"),
        ],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("How would you like to provide the files for translation?", reply_markup=reply_markup)
    return JSON_TRANSLATE_CHOICE

# ... other functions like json_translate_prompt_json_for_zip, etc. remain the same ...
# Remember to change their final return to `await start(update, context)` as well.

# --- 4. Language Menu (Informational) ---
async def language_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton("« Back", callback_data="main_menu_start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text(
        "The current OCR model (Florence-2) is multilingual and detects language automatically. No selection is needed.",
        reply_markup=reply_markup
    )
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
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$"),
                # CallbackQueryHandler(json_divide_menu, pattern="^main_divide$"), # Add this line
                CallbackQueryHandler(language_menu, pattern="^main_language$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$"), # Handler for "Back" button
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt, pattern="^jm_image$|^jm_zip$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$"),
            ],
            WAITING_IMAGES_OCR: [MessageHandler(filters.PHOTO, json_maker_process_images)],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
            
            # You would add states for JSON_TRANSLATE_CHOICE etc. here, similar to the above.
            # Example for JSON translate zip upload
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json_for_zip, pattern="^jt_zip$"),
                CallbackQueryHandler(start, pattern="^main_menu_start$")
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension('json'), json_translate_get_json_for_zip)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
        },
        fallbacks=[CommandHandler("start", start)],
        # Note: You can add a `CallbackQueryHandler(cancel, pattern='^cancel$')` if you want a cancel button
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    # NOTE: The provided code is a template. Functions like json_translate_process_zip
    # and the JSON Divide section need their final `return` statement updated to `await start(update, context)`
    # to ensure the conversation loops back to the main menu.
    
    # Dummy functions for parts not fully fleshed out in this example
    async def json_translate_prompt_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("Please upload the JSON file for translation.")
        return WAITING_JSON_TRANSLATE_ZIP

    async def json_translate_get_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... logic to get json ...
        await update.message.reply_text("JSON received. Now, please upload the corresponding zip file.")
        return WAITING_ZIP_TRANSLATE
    
    async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... logic to process zip ...
        await update.message.reply_text("Translation complete!")
        return await start(update, context) # <-- IMPORTANT: Return to menu
        
    main()

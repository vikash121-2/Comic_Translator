import logging
import os
import zipfile
import shutil
import json
import io
import tempfile
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
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
from telegram.constants import ParseMode

# --- OCR Initialization ---
# This is a heavy import, so it's good to know it's happening at the start.
try:
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition import vit
    from surya.data.transforms import letterbox
    from surya.languages import CODE_TO_LANGUAGE
except ImportError:
    print("Error: Surya OCR library not found.")
    print("Please install it with: pip install surya-ocr")
    exit(1)

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global Variables & Configuration ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "font.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory
DEFAULT_LANG = ["en"]
SUPPORTED_LANGS = sorted(CODE_TO_LANGUAGE.keys())

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
    CHOOSE_LANG
) = range(13)

# Load OCR models once to be used globally
det_processor, det_model = segformer.load_processor_and_model()
rec_processor, rec_model = vit.load_processor_and_model()


# --- Helper Functions ---
def get_ocr_results(image_paths: List[str], lang: List[str]) -> Dict:
    """Runs OCR on a list of images and returns a structured dictionary."""
    images = [Image.open(p) for p in image_paths]
    predictions = run_ocr(images, [lang] * len(images), det_model, det_processor, rec_model, rec_processor)

    results = {}
    for i, pred in enumerate(predictions):
        image_name = os.path.basename(image_paths[i])
        text_blocks = []
        for line in pred.text_lines:
            text_blocks.append({
                "text": line.text,
                "location": line.bbox
            })
        results[image_name] = text_blocks
    return results


# --- Bot Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and displays the main menu."""
    reply_keyboard = [
        ["Json maker"],
        ["json To Comic translate"],
        ["json divide"],
        ["Choose ocr language"],
    ]
    await update.message.reply_text(
        "Welcome! I can help you with OCR and comic translation tasks.\n\n"
        "Please choose an option:",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, resize_keyboard=True
        ),
    )
    return MAIN_MENU


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Operation cancelled.", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


# --- 1. JSON Maker ---
async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the JSON Maker sub-menu."""
    reply_keyboard = [["image upload", "zip upload"]]
    await update.message.reply_text(
        "How would you like to provide the source files?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_MAKER_CHOICE

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to upload images for OCR."""
    await update.message.reply_text("Please send me one or more images.", reply_markup=ReplyKeyboardRemove())
    context.user_data['media_group'] = []
    return WAITING_IMAGES_OCR

async def json_maker_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes uploaded images, performs OCR, and sends a JSON file."""
    await update.message.reply_text("Processing images... This may take a moment.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        for msg in context.user_data.get('media_group', [update.message]):
            photo_file = await msg.photo[-1].get_file()
            file_path = os.path.join(temp_dir, f"{photo_file.file_id}.jpg")
            await photo_file.download_to_drive(file_path)
            image_paths.append(file_path)

        lang = context.user_data.get('ocr_lang', DEFAULT_LANG)
        ocr_data = get_ocr_results(image_paths, lang)

        final_json = {
            "images": [
                {"image_name": name, "text_blocks": blocks}
                for name, blocks in ocr_data.items()
            ]
        }
        
        json_path = os.path.join(temp_dir, "extracted_text.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
            
        await update.message.reply_document(document=open(json_path, 'rb'))
    
    await update.message.reply_text("Done! What would you like to do next?")
    context.user_data.pop('media_group', None)
    return await start(update, context) # Go back to main menu

async def json_maker_prompt_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to upload a zip file for OCR."""
    await update.message.reply_text("Please send me a zip file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_ZIP_OCR

async def json_maker_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes a zip file, performs OCR on all images, and sends a JSON file."""
    zip_file = await update.message.document.get_file()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        await update.message.reply_text("Zip file extracted. Starting OCR on all images...")
        
        lang = context.user_data.get('ocr_lang', DEFAULT_LANG)
        final_json = {"folders": []}

        for folder_name in sorted(os.listdir(extract_path)):
            folder_path = os.path.join(extract_path, folder_name)
            if os.path.isdir(folder_path):
                folder_data = {"folder_name": folder_name, "images": []}
                image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_paths:
                    ocr_data = get_ocr_results(image_paths, lang)
                    for name, blocks in ocr_data.items():
                        folder_data["images"].append({"image_name": name, "text_blocks": blocks})
                
                final_json["folders"].append(folder_data)
        
        json_path = os.path.join(temp_dir, "extracted_text_from_zip.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        
        await update.message.reply_document(document=open(json_path, 'rb'))

    await update.message.reply_text("Done! What would you like to do next?")
    return await start(update, context)


# --- 2. JSON to Comic Translate ---
async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the JSON Translate sub-menu."""
    reply_keyboard = [["image upload", "zip upload"]]
    await update.message.reply_text(
        "How would you like to provide the files for translation?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks for JSON before asking for the zip file."""
    await update.message.reply_text("First, please upload the JSON file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives JSON, stores it, and asks for the zip file."""
    json_doc = await update.message.document.get_file()
    json_bytes = await json_doc.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes.decode('utf-8'))
    await update.message.reply_text("JSON received. Now, please upload the corresponding zip file.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Draws text on images in a zip based on a JSON file and sends back a new zip."""
    zip_file = await update.message.document.get_file()
    json_data = context.user_data.pop('json_data')
    
    await update.message.reply_text("Processing zip file... This might take a while.")

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await update.message.reply_text(f"Error: Font file not found at '{FONT_PATH}'. Operation cancelled.")
        return await start(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)

        extract_path = os.path.join(temp_dir, "extracted")
        output_path = os.path.join(temp_dir, "output")
        os.makedirs(output_path)
        
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        for folder_info in json_data.get("folders", []):
            folder_name = folder_info["folder_name"]
            output_folder_path = os.path.join(output_path, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            for image_info in folder_info.get("images", []):
                image_name = image_info["image_name"]
                image_path = os.path.join(extract_path, folder_name, image_name)
                
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    for block in image_info.get("text_blocks", []):
                        x1, y1, x2, y2 = map(int, block["location"])
                        text = block["text"]
                        draw.rectangle([x1, y1, x2, y2], fill="white")
                        draw.text((x1, y1), text, font=font, fill="black")
                    
                    img.save(os.path.join(output_folder_path, image_name))

        output_zip_path = os.path.join(temp_dir, "translated_comic.zip")
        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', output_path)
        
        await update.message.reply_document(document=open(output_zip_path, 'rb'))

    await update.message.reply_text("Done! What would you like to do next?")
    return await start(update, context)

# Placeholder for the image-based translation (left as an exercise for brevity)
async def not_implemented(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("This feature is not yet implemented.", reply_markup=ReplyKeyboardRemove())
    return await start(update, context)


# --- 3. JSON Divide ---
async def json_divide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the JSON Divide sub-menu."""
    reply_keyboard = [["zip upload"]]
    await update.message.reply_text(
        "This function requires a master JSON file and a zip file.",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_DIVIDE_CHOICE

async def json_divide_prompt_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks for the master JSON file."""
    await update.message.reply_text("First, please upload the master JSON file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_JSON_DIVIDE

async def json_divide_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives JSON, stores it, and asks for the zip file."""
    json_doc = await update.message.document.get_file()
    json_bytes = await json_doc.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes.decode('utf-8'))
    await update.message.reply_text("JSON received. Now, please upload the corresponding zip file.")
    return WAITING_ZIP_DIVIDE

async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Splits JSON, masks images, and creates a new zip."""
    zip_file = await update.message.document.get_file()
    json_data = context.user_data.pop('json_data')

    await update.message.reply_text("Processing... This may take a while.")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        for folder_info in json_data.get("folders", []):
            folder_name = folder_info["folder_name"]
            current_folder_path = os.path.join(extract_path, folder_name)
            
            if os.path.isdir(current_folder_path):
                # 1. Create the smaller JSON for this folder
                folder_json_data = {"images": folder_info.get("images", [])}
                folder_json_path = os.path.join(current_folder_path, f"{folder_name}.json")
                with open(folder_json_path, 'w', encoding='utf-8') as f:
                    json.dump(folder_json_data, f, ensure_ascii=False, indent=4)
                
                # 2. Mask the images in this folder
                for image_info in folder_info.get("images", []):
                    image_name = image_info["image_name"]
                    image_path = os.path.join(current_folder_path, image_name)
                    
                    if os.path.exists(image_path):
                        img = Image.open(image_path).convert("RGB")
                        draw = ImageDraw.Draw(img)
                        
                        for block in image_info.get("text_blocks", []):
                            x1, y1, x2, y2 = map(int, block["location"])
                            draw.rectangle([x1, y1, x2, y2], fill="black") # Mask with black
                        
                        img.save(image_path)
        
        # 3. Create the new zip file
        output_zip_path = os.path.join(temp_dir, "divided_and_masked.zip")
        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', extract_path)
        
        await update.message.reply_document(document=open(output_zip_path, 'rb'))
        
    await update.message.reply_text("Done! What would you like to do next?")
    return await start(update, context)


# --- 4. Choose OCR Language ---
async def choose_lang_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays language selection buttons."""
    buttons = [
        InlineKeyboardButton(lang, callback_data=f"lang_{lang}")
        for lang in SUPPORTED_LANGS
    ]
    keyboard = [buttons[i:i + 3] for i in range(0, len(buttons), 3)] # 3 buttons per row
    
    await update.message.reply_text(
        "Please choose the OCR language:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CHOOSE_LANG

async def set_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Sets the OCR language based on user selection."""
    query = update.callback_query
    await query.answer()
    
    lang_code = query.data.split('_')[1]
    context.user_data['ocr_lang'] = [lang_code]
    
    await query.edit_message_text(text=f"OCR language set to: {CODE_TO_LANGUAGE[lang_code]} ({lang_code})")
    await query.message.reply_text("What would you like to do next?")
    return await start(query.message, context) # Use query.message to send a new message


# --- Media Group Handler ---
async def handle_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Collects all messages from a media group."""
    if 'media_group' not in context.user_data:
        context.user_data['media_group'] = []
    
    context.user_data['media_group'].append(update.message)

    # Schedule a job to process the group after a short delay
    # This prevents multiple processing calls for one group
    job_name = f"process_media_group_{update.message.media_group_id}"
    if not context.job_queue.get_jobs_by_name(job_name):
        context.job_queue.run_once(
            process_media_group_job, 
            when=1, # 1 second delay
            data={'chat_id': update.effective_chat.id, 'state': WAITING_IMAGES_OCR},
            name=job_name
        )

async def process_media_group_job(context: ContextTypes.DEFAULT_TYPE):
    """Job to process a collected media group."""
    # This is a bit of a workaround to call the handler from a job
    # We pass a mock update object
    class MockUpdate:
        class MockMessage:
            async def reply_text(self, *args, **kwargs):
                return await context.bot.send_message(context.job.data['chat_id'], *args, **kwargs)
            async def reply_document(self, *args, **kwargs):
                return await context.bot.send_document(context.job.data['chat_id'], *args, **kwargs)
        message = MockMessage()

    await json_maker_process_images(MockUpdate(), context)

# --- Main Function to Run the Bot ---
def main() -> None:
    """Run the bot."""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Error: Please replace 'YOUR_BOT_TOKEN_HERE' with your actual bot token.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.Regex("^Json maker$"), json_maker_menu),
                MessageHandler(filters.Regex("^json To Comic translate$"), json_translate_menu),
                MessageHandler(filters.Regex("^json divide$"), json_divide_menu),
                MessageHandler(filters.Regex("^Choose ocr language$"), choose_lang_menu),
            ],
            # JSON Maker States
            JSON_MAKER_CHOICE: [
                MessageHandler(filters.Regex("^image upload$"), json_maker_prompt_image),
                MessageHandler(filters.Regex("^zip upload$"), json_maker_prompt_zip),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO & ~filters.ChatType.PRIVATE, handle_media_group), # Handles groups
                MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, json_maker_process_images), # Handles single images
            ],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
            # JSON Translate States
            JSON_TRANSLATE_CHOICE: [
                MessageHandler(filters.Regex("^image upload$"), not_implemented), # Placeholder
                MessageHandler(filters.Regex("^zip upload$"), json_translate_prompt_json_for_zip),
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.JSON, json_translate_get_json_for_zip)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
            # JSON Divide States
            JSON_DIVIDE_CHOICE: [MessageHandler(filters.Regex("^zip upload$"), json_divide_prompt_json)],
            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.JSON, json_divide_get_json)],
            WAITING_ZIP_DIVIDE: [MessageHandler(filters.Document.ZIP, json_divide_process_zip)],
            # Language Choice State
            CHOOSE_LANG: [CallbackQueryHandler(set_lang, pattern="^lang_")],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()

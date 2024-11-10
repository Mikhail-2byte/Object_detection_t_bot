from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import InputFile
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5
from PIL import Image  # Для проверки формата изображения

# Загружаем переменные окружения из .env
load_dotenv()

# Загружаем токен бота
TOKEN = os.environ.get("TOKEN")  # ВАЖНО!!!!!

# Инициализируем класс YOLO
WORK_DIR = r'C:\Create_bot_OD'  # Рекомендуем поместить проект именно в корневую папку
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)


# Функция команды /start
async def start(update, context):
    await update.message.reply_text('Пришлите фото для распознавания объектов. Также можно настроить параметры распознавания.')


# Функция для работы с текстом
async def help(update, context):
    await update.message.reply_text("Отправьте изображение для распознавания объектов.")


# Функция обработки изображения
async def detection(update, context):
    # Удаляем папки с предыдущими изображениями и результатами распознавания
    try:
        if os.path.exists('images'):
            shutil.rmtree('images')
        if os.path.exists(f'{WORK_DIR}/yolov5/runs'):
            shutil.rmtree(f'{WORK_DIR}/yolov5/runs')
    except Exception as e:
        print(f"Error while cleaning up: {e}")

    my_message = await update.message.reply_text('Мы получили от тебя фотографию. Идет распознавание объектов...')

    # Получаем файл из сообщения
    new_file = await update.message.photo[-1].get_file()

    # Проверка на формат изображения
    image_name = str(new_file['file_path']).split("/")[-1]
    image_path = os.path.join('images', image_name)
    os.makedirs('images', exist_ok=True)

    # Скачиваем изображение
    await new_file.download_to_drive(image_path)

    try:
        with Image.open(image_path) as img:
            img.verify()  # Проверяем, что файл является изображением
    except (IOError, SyntaxError) as e:
        await update.message.reply_text("Ошибка: Это не изображение или оно повреждено.")
        return

    # Параметры для запуска YOLO
    conf_values = [0.01, 0.5, 0.99]
    iou_values = [0.01, 0.5, 0.99]
    class_ids = [0]  # Класс 0 — это люди, для COCO датасета

    # Установка параметров для конфигурации
    conf = 0.5  # Пример значения
    iou = 0.5   # Пример значения

    # Создаем словарь с параметрами для YOLO
    test_dict = {
        'weights': 'yolov5x.pt',
        'source': 'images',
        'conf': conf,
        'iou': iou,
        'classes': ' '.join(map(str, class_ids))  # Передаем только классы людей
    }

    # Запускаем YOLO
    yolov5.run(test_dict, exp_type='test')

    # Проверяем, существует ли результат обработки
    result_image_path = f"{WORK_DIR}/yolov5/runs/detect/exp/{image_name}"
    if not os.path.exists(result_image_path):
        await update.message.reply_text("Ошибка при обработке изображения. Попробуйте снова.")
        return

    # Удаляем предыдущее сообщение от бота
    await context.bot.delete_message(message_id=my_message.message_id, chat_id=update.message.chat_id)

    # Отправляем результат пользователю
    await update.message.reply_text('Распознавание объектов завершено')

    # Отправляем обработанное изображение
    with open(result_image_path, 'rb') as image_file:
        await update.message.reply_photo(image_file, filename=image_name)


def main():
    # Точка входа в приложение
    application = Application.builder().token(TOKEN).build()  # Создаем объект класса Application
    print('Бот запущен...')

    # Добавляем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, detection, block=False))
    application.add_handler(MessageHandler(filters.TEXT, help))

    # Запускаем бота (остановка CTRL + C)
    application.run_polling()


if __name__ == "__main__":
    main()

Пример кода на Python для подключения к Телеграм боту нейросетевой модели, запущенной на локальном ПК ( использована DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf ).

Пример запуска на ПК под управлением OC Windows 11.

Подготовка:

1. Скачайте в отдельную директорию нейросетевую модель https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/blob/main/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf
2. Создайте в Telegram бота через BotFather и получите Токен вашего бота.
3. Установите python (всё замечательно работает на версии 3.12 x64)
4. Установите Microsoft C++ Build Tools https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/
5. Установите git https://git-scm.com/downloads/win
6. Скачайте с этого гита файл tbot_local.py и сохраните его в директории где лежит нейросетевая модель

Для взаимодействия python с Тelegram необходимо установить python-telegram-bot

```
pip install python-telegram-bot
```

Для взаимодействия python с LLaMA необходимо установить llama_cpp (без заранее установленных на ПК Microsoft C++ Build Tools и git, установить llama_cpp не получится)

```
pip install llama-cpp-python
```

После установки llama_cpp убедитесь, что модель которую вы скачали *.gguf и файл кода tbot_local.py находятся в одной директории.

Настройте tbot_local.py под ваши параметры, не забудьте указать токен вашего бота в Телеграм.

Запуск скрипта в командной строке Windows производится в той же директории.
```
python tbot_local.py
```

Скрипт будет получать данные от бота, передавать их в виде запросов локальной модели нейросети, и отправлять ответы обратно в Телеграм.

Обратите внимание на следующие моменты:
- модели небольших размеров, запускаемые на локальных ПК, далеки от коммерческих аналогов и ~~могут~~ будут бредить
- чтобы ускорить работу модели, вам потребуется хорошая видеокарта от Nvidia (самая простая RTX3060 с 12Гб памяти)

![изображение](https://github.com/user-attachments/assets/d2d42236-7612-4221-bf6f-e9b18b320fe7)



Сам скрипт:

```
import re
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from llama_cpp import Llama

# Загрузка модели
llm = Llama(
    model_path="DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
    n_ctx=4096,          # Контекстное окно
    n_gpu_layers=-1,     # Использовать все слои GPU (если доступно)
    n_threads=8          # Количество потоков для CPU
)

# Системное сообщение и форматирование промпта
SYSTEM_MESSAGE = "Use russian language, be friendly, be short"

def format_prompt(user_message: str) -> str:
    #Форматируем промпт под Llama модель
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MESSAGE}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def remove_think_tags(text, tag="</think>"):
    # Убираем "размышления" DeepSeek из ответа нейросети
    pattern = re.compile(f".*?{re.escape(tag)}", re.DOTALL)
    cleaned_text = pattern.sub("", text, count=1)
    return cleaned_text

# Функция для обработки текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    
    # Форматируем промпт
    prompt = format_prompt(user_message)
    
    # Отправка сообщения в модель
    response = llm(
        prompt,
        max_tokens=1024,         # Максимальное количество токенов в ответе
        temperature=0.5,         # Контроль случайности (меньше = более детерминировано, большее значение = больше креативности)
        top_p=0.5,               # Nucleus сэмплирование (меньше = более детерминировано, большее значение = больше креативности)
        top_k=30,                # Top-k сэмплирование (меньше = меньше сложность ответа, большее значение = более комплексный ответ)
        stop=["<|start_header_id|>", "<|eot_id|>"],  # Стоп-токены
        echo=False               # Не возвращать промпт в ответе
    )
    
    # Извлекаем ответ модели
    model_response = response['choices'][0]['text'].strip()
    
    # Удаляем содержимое "размышлений" DeepSeek
    cleaned_response = remove_think_tags(model_response)
    
    # Отправка ответа обратно в чат
    await update.message.reply_text(cleaned_response)

# Функция для обработки команды /status
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Скрипт чат бота активен.')

def main():
    # Замените 'ТОКЕН_ВАШЕГО_БОТА' на токен вашего бота
    application = ApplicationBuilder().token("ТОКЕН_ВАШЕГО_БОТА").build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("status", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запуск бота
    application.run_polling()

if __name__ == "__main__":
    main()
```

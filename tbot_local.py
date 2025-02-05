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
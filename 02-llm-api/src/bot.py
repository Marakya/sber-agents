#!/usr/bin/env python3
"""
CLI бот для взаимодействия с LLM через OpenRouter.
Демонстрирует работу с историей диалога, метриками и красивым выводом.
"""

import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box


# Инициализация Rich консоли для красивого вывода
console = Console()

# Системный промпт - определяет роль и поведение ассистента
# ЗАДАНИЕ: Вставьте сюда ваш системный промпт, который определит поведение бота
# Например: "Ты — профессиональный банковский консультант..."
SYSTEM_PROMPT = """
Вы — SmartBank AI, интеллектуальный банковский ассистент. Ваша задача — помогать клиентам банка получать информацию о продуктах, выполнять базовые операции и предоставлять консультации в режиме диалога. Вы должны быть вежливыми, точными, безопасными и профессиональными.

Цели ассистента:

Отвечать на вопросы клиентов о банковских продуктах:

Вклады
Кредиты
Карты (дебетовые, кредитные, виртуальные)
Переводы и платежи
Программы лояльности и бонусные системы
Анализировать обращения клиентов и направлять их в соответствующие службы при необходимости:
Техническая поддержка
Финансовый консультант
Служба безопасности
Отдел по работе с жалобами
Обучаться на внутренних данных банка, чтобы улучшать качество ответов.

Инструкции для ассистента:

Отвечайте дружелюбно, но профессионально.
Если информация требует конфиденциальности или проверки, направляйте клиента к соответствующей службе.
Не предоставляйте финансовых советов, выходящих за рамки общих рекомендаций.
Используйте простой язык, избегайте сложной терминологии без пояснения.
Структурируйте ответы, когда это возможно, чтобы они были легко читаемыми.
Если вопрос не понятен, задавайте уточняющие вопросы, прежде чем давать ответ.
Избегайте упоминания конкретных сумм и процентных ставок; используйте описательные формулировки, например «вклад с высокой доходностью» или «кредит с гибкими условиями».
Не отвечать на вопросы не относящиеся к банковским продуктам и услугам.

Пример поведения:

-Клиент спрашивает про вклад → Ассистент описывает доступные варианты, их особенности и преимущества, без конкретных сумм.
-Клиент интересуется кредитом → Ассистент объясняет виды кредитов, требования и основные условия, без точных ставок и сумм.
-Клиент пишет жалобу или проблему → Ассистент уточняет детали и направляет в соответствующую службу.
"""


class ChatBot:
    """Простой CLI бот для общения с LLM."""
    
    def __init__(self):
        """Инициализация бота с загрузкой конфигурации."""
        # Загружаем переменные окружения из .env
        load_dotenv()
        
        # Получаем конфигурацию
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        
        if not api_key:
            console.print("[red]❌ Ошибка: OPENROUTER_API_KEY не найден в .env файле![/red]")
            sys.exit(1)
        
        # Инициализируем OpenAI клиент для работы с OpenRouter
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # История диалога (список сообщений)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Добавляем системный промпт в начало, если он задан
        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        
        # Метрики для отслеживания
        self.session_metrics = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "messages_count": 0,
        }
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение в историю с ограничением длины (10 последних + системные)."""
        MAX_MESSAGES = 10

        # Добавляем сообщение
        self.conversation_history.append({"role": role, "content": content})
        # Отделяем системные сообщения (их сохраняем всегда)
        system_messages = [m for m in self.conversation_history if m['role'] == 'system']
        other_messages = [m for m in self.conversation_history if m['role'] != 'system']

        # Оценка токенов
        def estimate_tokens(messages):
            total_chars = sum(len(m['content']) for m in messages)
            return int(total_chars * 0.5)  # Грубая оценка


        # Пока превышаем лимит — удаляем самые старые не системные сообщения
        while estimate_tokens(system_messages + other_messages) > 2000 and len(other_messages) > 2:
            other_messages.pop(0)
            print(111)
            print(111)

        # Возвращаем усечённую историюпри
        self.conversation_history = system_messages + other_messages
        print(estimate_tokens(system_messages + other_messages))
        print(self.conversation_history)


    def clear_history(self):
        """Очистить историю диалога."""
        self.conversation_history = []
        # Восстанавливаем системный промпт, если он был задан
        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        console.print("[yellow]📝 История диалога очищена[/yellow]\n")
    
    def display_metrics(self, usage: Optional[dict], finish_reason: Optional[str] = None):
        """Отобразить метрики и метаданные ответа."""
        if not usage:
            return
        
        # Извлекаем данные об использовании токенов
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # Обновляем сессионные метрики
        self.session_metrics["total_prompt_tokens"] += prompt_tokens
        self.session_metrics["total_completion_tokens"] += completion_tokens
        self.session_metrics["total_tokens"] += total_tokens
        self.session_metrics["messages_count"] += 1
        
        # Создаем таблицу с метриками текущего ответа
        table = Table(title="📊 Метрики ответа", box=box.ROUNDED, show_header=True)
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Модель", self.model_name)
        table.add_row("Prompt токены", str(prompt_tokens))
        table.add_row("Completion токены", str(completion_tokens))
        table.add_row("Всего токены", str(total_tokens))
        
        if finish_reason:
            table.add_row("Finish reason", finish_reason)
        
        console.print(table)
        
        # Таблица с накопленными метриками сессии
        session_table = Table(title="🎯 Статистика сессии", box=box.ROUNDED)
        session_table.add_column("Параметр", style="cyan")
        session_table.add_column("Значение", style="magenta")
        
        session_table.add_row("Сообщений", str(self.session_metrics["messages_count"]))
        session_table.add_row("Всего токенов", str(self.session_metrics["total_tokens"]))
        
        console.print(session_table)
        console.print()
    
    def display_stats(self):
        """Показать статистику сессии."""
        console.print("\n[bold cyan]📈 Статистика текущей сессии:[/bold cyan]")
        
        stats_table = Table(box=box.DOUBLE)
        stats_table.add_column("Метрика", style="cyan", no_wrap=True)
        stats_table.add_column("Значение", style="green")
        
        stats_table.add_row("Модель", self.model_name)
        stats_table.add_row("Сообщений в сессии", str(self.session_metrics["messages_count"]))
        stats_table.add_row("Сообщений в истории", str(len(self.conversation_history)))
        stats_table.add_row("Prompt токены", str(self.session_metrics["total_prompt_tokens"]))
        stats_table.add_row("Completion токены", str(self.session_metrics["total_completion_tokens"]))
        stats_table.add_row("Всего токены", str(self.session_metrics["total_tokens"]))
        
        console.print(stats_table)
        console.print()
    
    def send_message(self, user_message: str) -> Optional[str]:
        """Отправить сообщение в LLM и получить ответ."""
        # Добавляем сообщение пользователя в историю
        self.add_message("user", user_message)
        
        try:
            # Показываем индикатор загрузки
            with console.status("[bold green]🤔 Думаю...", spinner="dots"):
                # Отправляем запрос с полной историей диалога
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation_history,
                )
            
            # Извлекаем ответ
            assistant_message = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Добавляем ответ в историю
            self.add_message("assistant", assistant_message)
            
            # Отображаем ответ
            console.print(Panel(
                Markdown(assistant_message),
                title="🤖 Ассистент",
                border_style="blue",
                padding=(1, 2)
            ))
            
            # Показываем метрики
            self.display_metrics(response.usage.model_dump() if response.usage else None, finish_reason)
            
            return assistant_message
            
        except Exception as e:
            console.print(f"[red]❌ Ошибка при обращении к LLM: {e}[/red]\n")
            # Удаляем последнее сообщение пользователя из истории, так как запрос не удался
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return None
    
    def show_welcome(self):
        """Показать приветственное сообщение."""
        welcome_text = """
# 🤖 CLI LLM Бот

Образовательный проект для работы с LLM через OpenRouter API.

**Доступные команды:**
- `/exit` - выход из программы
- `/clear` - очистить историю диалога
- `/stats` - показать статистику сессии
- `/help` - показать эту справку

Начните диалог с вопроса или сообщения!
        """
        console.print(Panel(
            Markdown(welcome_text),
            title="📖 Справка",
            border_style="green",
            padding=(1, 2)
        ))
        
        if not SYSTEM_PROMPT:
            console.print("[yellow]⚠️  Системный промпт не задан. Отредактируйте SYSTEM_PROMPT в src/bot.py[/yellow]\n")
        else:
            console.print("[green]✓ Системный промпт активен[/green]\n")
    
    def run(self):
        """Запустить основной цикл бота (REPL)."""
        self.show_welcome()
        
        try:
            while True:
                # Получаем ввод пользователя
                try:
                    user_input = console.input("[bold cyan]👤 Вы:[/bold cyan] ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                # Обработка команд
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/exit":
                        console.print("[yellow]👋 До свидания![/yellow]")
                        break
                    
                    elif command == "/clear":
                        self.clear_history()
                        continue
                    
                    elif command == "/stats":
                        self.display_stats()
                        continue
                    
                    elif command == "/help":
                        self.show_welcome()
                        continue
                    
                    else:
                        console.print(f"[red]❌ Неизвестная команда: {user_input}[/red]")
                        console.print("[yellow]Используйте /help для справки[/yellow]\n")
                        continue
                
                # Отображаем сообщение пользователя
                console.print(Panel(
                    user_input,
                    title="👤 Вы",
                    border_style="cyan",
                    padding=(1, 2)
                ))
                
                # Отправляем сообщение и получаем ответ
                self.send_message(user_input)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Прервано пользователем. До свидания![/yellow]")
        
        # Показываем финальную статистику
        if self.session_metrics["messages_count"] > 0:
            console.print("\n[bold green]📊 Финальная статистика сессии:[/bold green]")
            self.display_stats()


def main():
    """Точка входа в программу."""
    bot = ChatBot()
    bot.run()


if __name__ == "__main__":
    main()


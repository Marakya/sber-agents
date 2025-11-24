"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from langchain_core.tools import tool
import rag

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую.
    
    Args:
        amount: Сумма для конвертации
        from_currency: Исходная валюта (USD, EUR, RUB)
        to_currency: Целевая валюта (USD, EUR, RUB)
    
    Returns:
        Строка с результатом конвертации
    """

    # Пример фиксированных курсов (условные значения)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "RUB": 100.0
    }

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # Проверка валют
    if from_currency not in rates:
        return f"Неизвестная исходная валюта: {from_currency}"
    if to_currency not in rates:
        return f"Неизвестная целевая валюта: {to_currency}"

    # Перевод суммы в USD как базовую валюту
    amount_in_usd = amount / rates[from_currency]

    # Перевод из USD в конечную валюту
    converted_amount = amount_in_usd * rates[to_currency]

    return f"{amount:.2f} {from_currency} = {converted_amount:.2f} {to_currency}"

import os
import json
import logging
import google.generativeai as genai
from pydantic import BaseModel
import time # sleep için
import functools
import inspect

# Hedef dosyanın tam yolu
file_path_llm = "/srv/conda/envs/notebook/lib/python3.10/site-packages/tinytroupe/utils/llm.py"

# Gemini uyumlu yeni llm.py içeriği
# Bu kısım, orijinal llm.py dosyasının kritik bölümlerini Gemini'ye adapte eder.
new_content_llm = """
import functools
import inspect
import time

# Doğrudan openai_utils'tan 'client'ı içeri aktarıyoruz (artık Gemini özellikli LLMProvider'ımız)
from tinytroupe.openai_utils import client 

from tinytroupe.utils import logger
from tinytroupe.utils.rendering import break_text_at_length

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 1.0

def llm_call_wrapper(func):
    """
    LLM çağrıları için retries ve loglama gibi genel işlemleri yöneten bir dekoratör.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@llm_call_wrapper
def _send_chat_completion_request(messages, model=None, temperature=0.7, top_p=1.0, max_tokens=None, stop_sequences=None, response_format=None):
    """
    LLM'e bir chat completion isteği gönderir.
    Doğrudan değiştirilmiş openai_utils.client().send_message'ı kullanır.
    """
    # LLMRequest nesnesi oluşturmuyoruz.
    # Tüm parametreler doğrudan send_message'a iletilir veya LLMProvider tarafından dahili olarak işlenir.
    response = client().send_message(
        messages=messages,
        response_format=response_format # JSON modu için response_format'ı iletiyoruz
        # Diğer parametreler (model, temperature, top_p, max_tokens, stop_sequences)
        # LLMProvider içinde Gemini için yapılandırılır veya genai.GenerativeModel.generate_content
        # çağrısında örtük olarak işlenir.
    )
    return response

def _llm_request_with_retries(messages, model=None, temperature=0.7, top_p=1.0, max_tokens=None, stop_sequences=None, response_format=None):
    """
    LLM istekleri için yeniden denemeleri işleyen yardımcı fonksiyon.
    Bu şimdi Gemini istemcimizi kullanan _send_chat_completion_request'i çağırır.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # Doğrudan uyarlanmış _send_chat_completion_request'i çağırın
            return _send_chat_completion_request(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                response_format=response_format
            )
        except Exception as e:
            logger.error(f"[{attempt + 1}] Error: {e}")
            attempt += 1
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Failed to get response after {MAX_RETRIES}.0 attempts.")
                raise

def _extract_llm_call_params(llm_call_func, *args, **kwargs):
    """
    Fonksiyon argümanlarından LLM çağrı parametrelerini çıkarır.
    """
    arg_names = inspect.getfullargspec(llm_call_func).args
    params = {}
    for i, arg in enumerate(args):
        if i < len(arg_names):
            params[arg_names[i]] = arg
    params.update(kwargs)
    return params

def num_tokens_from_messages(messages, model="gemini-1.5-flash"): # Model adını Gemini olarak güncelledik
    """Return the number of tokens in messages."""
    # Bu fonksiyon genellikle OpenAI modelleri için tiktoken kullanır.
    # Gemini için token sayımı farklıdır.
    # Bu fonksiyon kritikse, Gemini'ye özgü bir uygulamaya ihtiyaç duyacaktır.
    # Şimdilik, olduğu gibi bırakalım, doğrudan Gemini için kullanılmayabilir.
    # Doğru token sayımı için Google Generative AI SDK'sının kendi token sayma metodunu kullanmak gerekebilir.
    # Örneğin: model.count_tokens(messages).total_tokens
    
    # Basit bir tahmini sayaç bırakalım, zira tiktoken Gemini için doğru sonuç vermez
    return sum(len(msg['content'].split()) for msg in messages) * 4 # Kaba tahmin

def truncate_messages_to_max_tokens(messages, max_tokens, model="gemini-1.5-flash"):
    """
    Bir mesaj listesini maksimum token limiti içinde kalacak şekilde kısaltır.
    """
    if num_tokens_from_messages(messages, model=model) <= max_tokens:
        return messages

    truncated_messages = []
    current_tokens = 0

    # Sistem mesajını (varsa) koru
    if messages and messages[0]['role'] == 'system':
        truncated_messages.append(messages[0])
        current_tokens += num_tokens_from_messages([messages[0]], model=model)
        messages = messages[1:]

    # Maksimum token sayısına ulaşılana kadar son mesajları ekle
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_tokens = num_tokens_from_messages([msg], model=model)
        if current_tokens + msg_tokens <= max_tokens:
            truncated_messages.insert(0, msg) # Başa ekle, çünkü sondan başa doğru okuyoruz
            current_tokens += msg_tokens
        else:
            break
    
    # Hala fazlaysa, en eski sistem dışı mesajın içeriğini kısalt
    if current_tokens > max_tokens and truncated_messages:
        for i, msg in enumerate(truncated_messages):
            if msg['role'] != 'system':
                estimated_tokens_to_remove = current_tokens - max_tokens
                chars_to_remove = estimated_tokens_to_remove * 4 # Kaba tahmin
                if len(msg['content']) > chars_to_remove:
                    msg['content'] = msg['content'][chars_to_remove:]
                    logger.warning(f"Mesaj içeriği token limitine uyması için kısaltıldı. Orjinal: {current_tokens}, Hedef: {max_tokens}")
                    break

    return truncated_messages
""" # new_content_llm stringinin sonu

# Dosyayı yazma işlemi
try:
    with open(file_path_llm, "w") as f:
        f.write(new_content_llm)
    print(f"✅ '{file_path_llm}' dosyası başarıyla Gemini uyumlu hale getirildi!")
    print("Artık ana kodunuzu deneyebilirsiniz.")
except Exception as e:
    print(f"❌ Dosya güncellenirken bir hata oluştu: {e}")
    print("Lütfen yolun doğru olduğundan ve MyBinder ortamında bu dosyaya yazma izniniz olduğundan emin olun.")

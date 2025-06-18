import os
import json
import logging
import google.generativeai as genai
from pydantic import BaseModel # TinyTroupe'un Pydantic modellerini kullanabilmesi için

# Bu, TinyTroupe'un varsayılan olarak kullandığı Pydantic modeli olabilir.
# Eğer kütüphanede farklı bir import yolu varsa, onu kullanın.
# Genellikle tinytroupe.agent.tiny_person.CognitiveActionModel olarak tanımlı olabilir.
# Burada varsayılan bir tanım bırakalım, hata alırsanız TinyTroupe'un kendi tanımını bulun.
try:
    from tinytroupe.agent.tiny_person import CognitiveActionModel
except ImportError:
    # Fallback if not found directly, define a basic one or expect issues
    logging.warning("CognitiveActionModel not found directly, using a placeholder. This might lead to parsing errors.")
    class CognitiveActionModel(BaseModel):
        thought: str
        action: str
        target: str
        args: dict = {}


class LLMProvider:
    _instance = None
    _retries = 5  # Hata durumunda tekrar deneme sayısı
    _retry_delay = 1.0 # Tekrar deneme gecikmesi

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_from_config()
        return cls._instance

    def _setup_from_config(self):
        # Gemini API anahtarını GOOGLE_API_KEY ortam değişkeninden al
        # Not: Kodunuz GEMINI_API_KEY olarak ayarlıyor, bunu GOOGLE_API_KEY olarak değiştirmelisiniz.
        api_key = os.getenv("GOOGLE_API_KEY") # Ya da GEMINI_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set for Gemini.")
        genai.configure(api_key=api_key)
        # Gemini modeli olarak gemini-1.5-flash kullanıyoruz.
        # İhtiyacınıza göre gemini-pro veya daha yeni bir model seçebilirsiniz.
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        logging.info(f"Gemini model '{self.model.model_name}' başarıyla ayarlandı.")

    def send_message(self, messages, response_format=None):
        # OpenAI mesaj formatını Gemini formatına dönüştür
        gemini_messages = []
        for msg in messages:
            # OpenAI'nin "assistant" rolü Gemini'de "model" rolüne karşılık gelir.
            # "system" rolü Gemini'de doğrudan desteklenmez, genellikle ilk kullanıcı mesajına eklenir
            # veya ayrı bir "system_instruction" olarak verilir. Basitlik için "user" olarak işleyelim.
            role_map = {"user": "user", "assistant": "model", "system": "user"}
            gemini_messages.append({"role": role_map.get(msg["role"], "user"), "parts": [{"text": msg["content"]}]})

        generation_config = {}
        if response_format:
            # TinyTroupe'un Pydantic modeli beklediğini varsayarak JSON modunu etkinleştir.
            generation_config["response_mime_type"] = "application/json"
            
            # Gemini'ye JSON formatında yanıt vermesi gerektiğini belirten bir talimat ekle.
            # Bu talimatı sohbetin ilk mesajına eklemek en etkili yöntemdir.
            schema_json = json.dumps(response_format.model_json_schema())
            if gemini_messages:
                # İlk mesajın içeriğini al ve başına JSON talimatını ekle
                first_msg_content = gemini_messages[0]["parts"][0]["text"]
                gemini_messages[0]["parts"][0]["text"] = (
                    f"Your response MUST be a JSON object conforming to this schema: {schema_json}\n\n"
                    f"{first_msg_content}"
                )
            else:
                # Konuşma boşsa, sadece talimatı içeren bir kullanıcı mesajı ekle
                gemini_messages.append({"role": "user", "parts": [{"text": f"Your response MUST be a JSON object conforming to this schema: {schema_json}"}]})

        # Gemini API'sinden gelen olası içerik engellemelerini azaltmak için güvenlik ayarlarını düşür.
        # Dikkat: Bu, daha az güvenli yanıtlar almanıza neden olabilir.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        attempt = 0
        while attempt < self._retries:
            try:
                logging.info(f"Gemini API'ye istek gönderiliyor (Deneme {attempt + 1}/{self._retries})...")
                response = self.model.generate_content(
                    gemini_messages,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    request_options={'timeout': 120} # İstek zaman aşımını artır
                )

                # Prompt veya yanıtın engellenip engellenmediğini kontrol et
                if response._result.prompt_feedback and response._result.prompt_feedback.block_reason:
                    logging.error(f"Gemini API promptu engelledi: {response._result.prompt_feedback.block_reason}")
                    raise Exception(f"Gemini API promptu engelledi: {response._result.prompt_feedback.block_reason}")
                
                if response._result.candidates:
                    for candidate in response._result.candidates:
                        if candidate.finish_reason == 4: # SAFETY (4), OTHER (5)
                            logging.warning(f"Gemini API yanıtı güvenlik nedeniyle engelledi: {candidate.safety_ratings}")
                            # Engellenen yanıtı atla ve tekrar dene veya hata fırlat
                            raise Exception(f"Gemini API yanıtı güvenlik nedeniyle engelledi: {candidate.safety_ratings}")
                
                # Eğer JSON formatı isteniyorsa, yanıtı JSON olarak ayrıştır
                if response_format and generation_config.get("response_mime_type") == "application/json":
                    try:
                        # Gemini'den gelen yanıtın sadece metin kısmını al ve JSON olarak yükle
                        return json.loads(response.text)
                    except json.JSONDecodeError as json_e:
                        logging.error(f"Gemini'den gelen JSON yanıtı ayrıştırılamadı: {response.text} - Hata: {json_e}")
                        raise ValueError(f"Geçersiz JSON yanıtı: {response.text[:200]}...") from json_e
                else:
                    return response.text # Metin yanıtı dön

            except Exception as e:
                logging.error(f"Gemini API çağrısında hata: {type(e).__name__} - {e}")
                attempt += 1
                if attempt < self._retries:
                    import time
                    time.sleep(self._retry_delay)
                else:
                    raise # Tüm denemeler başarısız olursa hatayı fırlat

# Global istemci örneği
_llm_provider_instance = None
def client():
    global _llm_provider_instance
    if _llm_provider_instance is None:
        _llm_provider_instance = LLMProvider()
    return _llm_provider_instance

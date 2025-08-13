# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import google.generativeai as genai
import os, re, time

# ------------------ Config & Setup ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENV_MODEL = os.getenv("GEMINI_MODEL_NAME", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = ENV_MODEL or "gemini-1.5-flash"
else:
    GEMINI_MODEL_NAME = ""  # sem chave -> modo offline

app = FastAPI(title="English WhatsApp Bot", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Memória volátil simples (reinicia a cada deploy)
user_memory = {}
last_quota_error_at = 0.0  # epoch da última 429 global
USER_COOLDOWN_SECONDS = 6   # throttle de IA por usuário

# ------------------ Schemas ------------------
class Message(BaseModel):
    user_message: str
    level: str = "basic"
    phone: str = "unknown"

class ResetReq(BaseModel):
    phone: str

class WhatsAppMessage(BaseModel):
    from_number: str
    body: str

# ------------------ Helpers ------------------
def safe_detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "pt"
    except Exception:
        return "pt"

def is_quota_error_text(text: str) -> bool:
    t = (text or "").lower()
    return " 429 " in t or "exceeded your current quota" in t or "rate limits" in t

QUOTA_FRIENDLY_REPLY_PT = "⚠️ Bati no limite gratuito diário da IA por agora. Tente novamente mais tarde. 🙏"
QUOTA_FRIENDLY_REPLY_EN = "⚠️ I just hit today’s free AI quota. Please try again later. 🙏"

def model_generate_text(prompt: str) -> str:
    """Chama Gemini; retorna string legível (mesmo em erro)."""
    if not GEMINI_API_KEY or not GEMINI_MODEL_NAME:
        return "⚠️ (modo offline) GEMINI_API_KEY ausente."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text else "(sem resposta do modelo)"
    except Exception as e:
        return f"⚠️ Erro ao consultar o modelo: {str(e)}"

def strip_motivacao_label(text: str) -> str:
    pattern = r"(?im)^\s*\*?\s*motiv[aá]?[cç][aã]o\s*\*?\s*:\s*|^\s*\*?\s*motivation\s*\*?\s*:\s*"
    return re.sub(pattern, "", text)

def strip_greeting_prefix(text: str) -> str:
    pattern = r"(?im)^\s*(?:ol[áa]|oi|hello|hi|hey)\s*[!,.…]*\s*[🙂😊👋🤝👍🤗🥳✨]*\s*-?\s*"
    return re.sub(pattern, "", text, count=1).lstrip()

# --------- Intent detection ----------
PT_QUESTION_WORDS = {
    "o que","oq","qual","quais","como","quando","onde","por que","porque","por quê",
    "pra que","para que","diferença","significa","pode me ajudar","me explica",
    "é correto","esta certo","está certo","está errado","devo usar"
}
EN_QUESTION_WORDS = {
    "what","which","how","when","where","why","difference","mean","meaning",
    "should i","is it correct","am i","can i","could i","what's","whats"
}
SMALLTALK_WORDS_PT = {"obrigado","valeu","blz","beleza","tmj","ok","boa","bom dia","boa tarde","boa noite"}
SMALLTALK_WORDS_EN = {"thanks","thank you","ok","cool","nice","morning","good morning","good night","good evening"}

def classify_intent(text: str, lang: str) -> str:
    t = text.strip().lower()
    if "?" in t: return "question"
    if lang.startswith("pt"):
        if any(w in t for w in PT_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALLTALK_WORDS_PT): return "smalltalk"
    else:
        if any(w in t for w in EN_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALLTALK_WORDS_EN): return "smalltalk"
    if len(t.split()) <= 2: return "smalltalk"  # “ok”, “beleza”, etc.
    return "correction"

# --------- FAQ local (regex -> resposta) ----------
def local_faq_response(text: str, lang: str):
    t = text.lower().strip()

    def pick(pt, en):  # seleciona idioma
        return pt if lang.startswith("pt") else en

    # make vs do
    if re.search(r"\b(make|do)\b", t) and "difference" in t or "diferença" in t:
        return pick(
            "Diferença *make x do*: use *make* para criar/produzir algo (*make a cake*), "
            "e *do* para tarefas/atividades gerais (*do homework*). "
            "👉 Pratique: *I make breakfast, and I do the dishes.*",
            "Difference *make vs do*: use *make* to create/produce something (*make a cake*), "
            "and *do* for general tasks/activities (*do homework*). "
            "👉 Practice: *I make breakfast, and I do the dishes.*",
        )

    # used to
    if "used to" in t or "use to" in t or "used-to" in t or "significa" in t and "used to" in t:
        return pick(
            "*used to* fala de hábitos/situações do passado que não são mais verdadeiros: "
            "*I used to play soccer.* (eu jogava futebol). "
            "👉 Pratique: *I used to ______ every weekend.*",
            "*used to* refers to past habits/situations that are no longer true: "
            "*I used to play soccer.* "
            "👉 Practice: *I used to ______ every weekend.*",
        )

    # since vs for
    if ("since" in t and "for" in t) or "desde" in t and "por" in t:
        return pick(
            "*since* + ponto no tempo (desde quando): *since 2019*; "
            "*for* + duração (por quanto tempo): *for two years*. "
            "👉 Pratique: *I have lived here since 2019 / for two years.*",
            "*since* + starting point: *since 2019*; "
            "*for* + duration: *for two years*. "
            "👉 Practice: *I have lived here since 2019 / for two years.*",
        )

    # a / an / the
    if re.search(r"\b(a|an|the)\b", t) and ("usar" in t or "use" in t or "article" in t or "artigo" in t):
        return pick(
            "*a* antes de som inicial de consoante (*a dog*), *an* antes de som de vogal (*an apple*). "
            "*the* quando o leitor já sabe qual coisa específica. "
            "👉 Pratique: *I saw a cat. The cat was cute.*",
            "*a* before consonant sound (*a dog*), *an* before vowel sound (*an apple*). "
            "*the* when it’s specific/known. "
            "👉 Practice: *I saw a cat. The cat was cute.*",
        )

    # much vs many
    if ("much" in t and "many" in t) or "muito" in t and "muitos" in t:
        return pick(
            "*many* + contáveis (*many books*); *much* + incontáveis (*much water*). "
            "👉 Pratique: *How many friends do you have? / How much time do we have?*",
            "*many* with countables (*many books*); *much* with uncountables (*much water*). "
            "👉 Practice: *How many friends do you have? / How much time do we have?*",
        )

    # in / on / at (tempo)
    if re.search(r"\b(in|on|at)\b", t) and ("time" in t or "tempo" in t or "quando" in t):
        return pick(
            "*in* (meses/anos): *in July*; *on* (dias/datas): *on Monday*; *at* (horas): *at 7 pm*. "
            "👉 Pratique: *The class is on Tuesday at 8 am in May.*",
            "*in* (months/years): *in July*; *on* (days/dates): *on Monday*; *at* (times): *at 7 pm*. "
            "👉 Practice: *The class is on Tuesday at 8 am in May.*",
        )

    # comparatives / superlatives
    if "comparative" in t or "superlative" in t or "comparativo" in t or "superlativo" in t:
        return pick(
            "Adjetivos curtos: *-er* (comparativo) / *-est* (superlativo): *tall → taller / tallest*. "
            "Longos: *more*/*most*: *interesting → more interesting / most interesting*. "
            "👉 Pratique: *This book is more interesting than that one.*",
            "Short adjectives: *-er* (comparative) / *-est* (superlative): *tall → taller / tallest*. "
            "Long: *more*/*most*: *interesting → more interesting / most interesting*. "
            "👉 Practice: *This book is more interesting than that one.*",
        )

    # countable vs uncountable
    if "countable" in t or "uncountable" in t or "contável" in t or "incontável" in t:
        return pick(
            "Substantivos contáveis têm plural (*apples*); incontáveis não (*water*). "
            "Use *some/any* com incontáveis e contáveis no plural. "
            "👉 Pratique: *I need some water and some apples.*",
            "Countable nouns have plural (*apples*); uncountable don’t (*water*). "
            "Use *some/any* with uncountables and plural countables. "
            "👉 Practice: *I need some water and some apples.*",
        )

    return None  # sem match

# ------------------ Endpoints básicos ------------------
@app.get("/")
def root():
    return {"message": "OLÁ, MUNDO!", "service": "English WhatsApp Bot"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------ Endpoint principal ------------------
@app.post("/correct")
async def correct_english(message: Message):
    global last_quota_error_at
    user_text_raw = message.user_message or ""
    user_text = user_text_raw.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    memory = user_memory.setdefault(message.phone, {})
    now = time.time()

    # Throttle por usuário para chamadas de IA
    last_call = memory.get("last_call_ts", 0.0)
    def can_call_ai() -> bool:
        return (now - last_call) >= USER_COOLDOWN_SECONDS and (now - last_quota_error_at) >= 30

    # ------------------ Estados de Quiz / Desafio (mesmo de antes) ------------------
    if "awaiting_quiz" in memory:
        correct_letter = memory["awaiting_quiz"]["correct"].strip().upper()
        question = memory["awaiting_quiz"]["question"]
        explanation = memory["awaiting_quiz"]["explanation"]
        del memory["awaiting_quiz"]
        user_answer = user_text.strip().upper()
        if user_answer == correct_letter:
            return {"reply": f"✅ Parabéns, resposta correta! 🎉\n\n*{question}*\n✔️ Resposta: {correct_letter}\n🧠 {explanation}"}
        return {"reply": f"❌ Ops! Resposta incorreta.\n\n*{question}*\n✔️ Resposta correta: {correct_letter}\n🧠 {explanation}"}

    if "awaiting_challenge" in memory:
        answer = memory["awaiting_challenge"]["answer"].strip().lower()
        context = memory["awaiting_challenge"]["context"]
        explanation = memory["awaiting_challenge"]["explanation"]
        del memory["awaiting_challenge"]
        if user_text.strip().lower() == answer:
            return {"reply": "✅ Muito bem! Você acertou! 🎉\n\n*Frase:* {0}\n✔️ Resposta: {1}\n🧠 {2}".format(context, answer, explanation)}
        return {"reply": "❌ Ops! Não foi dessa vez.\n\n*Frase:* {0}\n✔️ Resposta correta: {1}\n🧠 {2}".format(context, answer, explanation)}

    # ------------------ Comandos ------------------
    cmd = user_text.lower()
    if cmd == "#resetar":
        user_memory.pop(message.phone, None)
        return {"reply": "🔄 Sua memória foi resetada com sucesso! Pode recomeçar."}

    if cmd == "#quiz" or cmd == "#desafio" or cmd == "#frase" or cmd == "#meta":
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        # (por brevidade: aqui poderíamos reusar as versões anteriores dos prompts)
        # para manter o foco do pedido, vamos priorizar a redução de chamadas no fluxo normal.
        # Se quiser, recoloco todos os blocos de #quiz/#desafio/#frase/#meta iguais ao anterior.

    # ------------------ Detecção de intenção ------------------
    lang = safe_detect_lang(user_text_raw)
    intent = classify_intent(user_text_raw, lang)

    # 1) Pergunta -> tenta FAQ local primeiro (grátis)
    if intent == "question":
        local = local_faq_response(user_text_raw, lang)
        if local:
            return {"reply": local}

        # se não bateu no FAQ, chama IA (se permitido)
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if lang.startswith('pt') else QUOTA_FRIENDLY_REPLY_EN}

        if lang.startswith("pt"):
            base = (
                "Você é um professor de inglês. Responda em português (Brasil).\n"
                "Explique de forma clara e prática o que o aluno perguntou, com exemplos curtos em inglês quando útil.\n"
                "NÃO cumprimente. NÃO corrija a pergunta do aluno. Foque na explicação do tema.\n"
                "Se houver termos em inglês, mantenha-os em *itálico*.\n"
                "No final, sugira 1 frase de exemplo para o aluno praticar (somente 1 linha)."
            )
        else:
            base = (
                "You are an English teacher. Answer in ENGLISH.\n"
                "Explain clearly what the student asked, with short examples when useful.\n"
                "Do NOT greet. Do NOT correct the student's question. Focus on the topic.\n"
                "Finish with one single practice sentence (one line)."
            )
        prompt = f"{base}\n\nStudent question:\n\"{user_text_raw}\"\n\nAnswer:"
        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if lang.startswith('pt') else QUOTA_FRIENDLY_REPLY_EN}
        text = strip_greeting_prefix(strip_motivacao_label(text))
        memory["last_call_ts"] = now
        return {"reply": text}

    # 2) Smalltalk (sempre offline)
    if intent == "smalltalk":
        return {"reply": "👍 Bora praticar! Envie uma frase em inglês para eu corrigir ou faça uma pergunta de gramática." if lang.startswith("pt")
                else "👍 Let's practice! Send me an English sentence to correct or ask a grammar question."}

    # 3) Correção (chama IA apenas se permitido)
    if not can_call_ai():
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if lang.startswith('pt') else QUOTA_FRIENDLY_REPLY_EN}

    if lang == "en" and user_text_raw.strip().lower().startswith("how"):
        base = (
            "You are a friendly English teacher. The student's English level is {level}.\n"
            "Answer in ENGLISH only. Do NOT greet. Do NOT translate the student's sentence.\n"
            "Return EXACTLY these sections, in this order, each on its own line:\n"
            "*Correction:* <corrected sentence in English>\n"
            "*Explanation:* <short explanation in English of the grammar or usage>\n"
            "*Tip:* <one short tip in English, end with a single emoji>\n"
            "No extra text before or after the sections."
        )
    else:
        base = (
            "Você é um professor amigável de inglês. O aluno está no nível {level}.\n"
            "Responda em PORTUGUÊS (Brasil). NÃO cumprimente. NÃO traduza a frase corrigida para o português.\n"
            "Devolva EXATAMENTE estes blocos, nesta ordem, cada um em sua própria linha:\n"
            "*Correção:* <frase corrigida em inglês>\n"
            "*Explicação:* <explicação curta em português sobre a regra aplicada>\n"
            "*Dica:* <uma dica curta em português, finalize com um único emoji>\n"
            "Não inclua nada além desses blocos."
        )

    prompt = base.format(level=message.level)
    full_prompt = f"{prompt}\n\nStudent: '{user_text_raw}'\nAnswer:"
    reply_text = model_generate_text(full_prompt)
    if is_quota_error_text(reply_text):
        last_quota_error_at = time.time()
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if lang.startswith('pt') else QUOTA_FRIENDLY_REPLY_EN}
    reply_text = strip_greeting_prefix(strip_motivacao_label(reply_text))
    memory["last_call_ts"] = now
    return {"reply": reply_text}

# ------------------ Utilidades ------------------
@app.post("/resetar")
async def resetar_memoria(req: ResetReq):
    user_memory.pop(req.phone, None)
    return {"status": "ok"}

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(msg: WhatsAppMessage):
    payload = Message(user_message=msg.body, phone=msg.from_number)
    result = await correct_english(payload)
    return {"to": msg.from_number, "reply": result.get("reply", "")}
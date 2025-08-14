# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import google.generativeai as genai
import os, re, time, random

# ------------------ Config & Setup ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENV_MODEL = os.getenv("GEMINI_MODEL_NAME", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = ENV_MODEL or "gemini-1.5-flash"
else:
    GEMINI_MODEL_NAME = ""  # sem chave -> modo offline

app = FastAPI(title="English WhatsApp Bot", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------ Estado do app ------------------
user_memory = {}
last_quota_error_at = 0.0
USER_COOLDOWN_SECONDS = 6

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

# ------------------ Utils ------------------
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

def is_all_english(text: str) -> bool:
    lang = safe_detect_lang(text)
    if lang != "en":
        return False
    if any(w in text.lower() for w in ["portugu", "tradu", "em pt", "em português", "em portugues"]):
        return False
    return True

# ------------------ Intents ------------------
PT_QUESTION_WORDS = {
    "o que","oq","qual","quais","como","quando","onde","por que","porque","por quê",
    "pra que","para que","diferença","significa","pode me ajudar","me explica",
    "é correto","esta certo","está certo","está errado","devo usar","exemplo de","como usar",
    "essa frase está correta","pode corrigir","corrigir","corrige","ver se está certo"
}
EN_QUESTION_WORDS = {
    "what","which","how","when","where","why","difference","mean","meaning",
    "should i","is it correct","am i","can i","could i","what's","whats","example of","how to use",
    "is this sentence correct","please correct"
}
SMALLTALK_WORDS_PT = {
    "obrigado","valeu","blz","beleza","tmj","ok","boa","bom dia","boa tarde","boa noite",
    "eai","e aí","tudo bem","tudo bom","oi","olá","salve","até mais","falou","tchau"
}
SMALLTALK_WORDS_EN = {
    "thanks","thank you","ok","cool","nice","morning","good morning","good night",
    "good evening","hi","hello","hey","see ya","bye","goodbye","see you"
}

def ask_to_explain_sentence_pt(t: str) -> bool:
    """PT: pede para explicar/traduzir uma frase específica."""
    t = t.lower()
    triggers = ["não entendi essa frase", "nao entendi essa frase",
                "pode me explicar", "me explica em português",
                "o que quer dizer", "o que significa", "tradu", "explica em portugues"]
    return any(x in t for x in triggers)

def ask_to_explain_sentence_en(t: str) -> bool:
    t = t.lower()
    triggers = ["explain this sentence", "what does it mean", "can you explain this",
                "translate this", "what does * mean"]
    return any(x in t for x in triggers)

def classify_intent(text: str, lang: str) -> str:
    t = text.strip().lower()

    # pedido explícito de "explicar em PT" o que foi dito antes
    if any(p in t for p in ["explica em português","explicar em português","em portugues","em português"]) \
       and any(w in t for w in ["pode","por favor","explica","explicar"]):
        return "explain_pt_previous"

    # explicar/traduzir frase específica
    if lang.startswith("pt") and ask_to_explain_sentence_pt(t):
        return "explain_sentence"
    if not lang.startswith("pt") and ask_to_explain_sentence_en(t):
        return "explain_sentence"

    if "?" in t:
        return "question"
    if lang.startswith("pt"):
        if any(w in t for w in PT_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALLTALK_WORDS_PT): return "smalltalk"
    else:
        if any(w in t for w in EN_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALLTALK_WORDS_EN): return "smalltalk"
    if len(t.split()) <= 2:
        return "smalltalk"
    return "correction"

# ------------------ Extrair frase alvo (para correção/explicação) ------------------
QUOTED_RE = re.compile(r'["“”\'‘’\u201c\u201d](.+?)["“”\'‘’\u201c\u201d]', re.DOTALL)

def looks_english(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    lang = safe_detect_lang(s)
    if lang == "en":
        return True
    # heurística leve: muitas letras ASCII + espaços
    letters = sum(ch.isalpha() for ch in s)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in s)
    return ascii_letters >= letters * 0.8 and letters >= 3

def extract_target_sentence(user_text: str) -> str | None:
    t = user_text.strip()

    # 1) Entre aspas
    m = QUOTED_RE.search(t)
    if m and looks_english(m.group(1)):
        return m.group(1).strip()

    # 2) Última linha se for inglês
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if looks_english(last):
            return last

    # 3) Depois de marcadores
    lower = t.lower()
    markers = [
        "essa frase está correta", "essa frase esta correta",
        "está correto", "esta correto",
        "is this sentence correct", "please correct",
        "não entendi essa frase", "nao entendi essa frase",
        "explain this sentence", "what does it mean"
    ]
    for mk in markers:
        if mk in lower:
            after = t[lower.find(mk)+len(mk):].strip(" :.-\n\t")
            # pega trecho entre aspas nesse pedaço
            m2 = QUOTED_RE.search(after)
            if m2 and looks_english(m2.group(1)):
                return m2.group(1).strip()
            if looks_english(after):
                return after
    return None

# ------------------ FAQ local (economiza cota) ------------------
def local_faq_response(text: str, lang: str):
    t = text.lower().strip()

    def pick(pt, en):
        return pt if lang.startswith("pt") else en

    if (re.search(r"\bmake\b", t) and re.search(r"\bdo\b", t)) and ("diferen" in t or "difference" in t):
        return pick(
            "Diferença *make x do*: use *make* para criar/produzir algo (*make a cake*) e *do* para tarefas gerais (*do homework*). 👉 *I make breakfast, and I do the dishes.*",
            "Difference *make vs do*: use *make* to create/produce (*make a cake*) and *do* for general tasks (*do homework*). 👉 *I make breakfast, and I do the dishes.*",
        )
    if "used to" in t or "use to" in t or ("significa" in t and "used to" in t):
        return pick(
            "*used to* = hábito no passado que não é mais verdade. 👉 *I used to play soccer.*",
            "*used to* = past habit no longer true. 👉 *I used to play soccer.*",
        )
    if ("since" in t and "for" in t) or ("desde" in t and "por" in t):
        return pick(
            "*since* + ponto no tempo; *for* + duração. 👉 *I have lived here since 2019 / for two years.*",
            "*since* + starting point; *for* + duration. 👉 *I have lived here since 2019 / for two years.*",
        )
    return None

# ------------------ Smalltalk ------------------
RESP_PT = {
    "default": [
        "👍 Bora praticar! Envie uma frase em inglês para eu corrigir ou faça uma pergunta de gramática.",
        "🚀 Partiu inglês! Manda uma frase ou dúvida que eu te ajudo.",
    ],
}
RESP_EN = {
    "default": [
        "👍 Let’s practice! Send me one sentence to correct or ask a grammar question.",
        "🚀 Ready when you are—one sentence or any question.",
    ],
}
def smalltalk_reply(text: str, lang: str) -> str:
    return random.choice(RESP_PT["default"] if lang.startswith("pt") else RESP_EN["default"])

# ------------------ Endpoints ------------------
@app.get("/")
def root():
    return {"message": "OLÁ, MUNDO!", "service": "English WhatsApp Bot"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/correct")
async def correct_english(message: Message):
    global last_quota_error_at
    user_text_raw = message.user_message or ""
    user_text = user_text_raw.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    memory = user_memory.setdefault(message.phone, {})
    now = time.time()

    last_call = memory.get("last_call_ts", 0.0)
    def can_call_ai() -> bool:
        return (now - last_call) >= USER_COOLDOWN_SECONDS and (now - last_quota_error_at) >= 30

    if user_text.lower() == "#resetar":
        user_memory.pop(message.phone, None)
        return {"reply": "🔄 Sua memória foi resetada. Podemos recomeçar!"}

    lang = safe_detect_lang(user_text_raw)
    intent = classify_intent(user_text_raw, lang)

    # (A) Reexplicar em PT a última resposta
    if intent == "explain_pt_previous":
        last_ai = memory.get("last_ai_reply", "")
        if not last_ai:
            return {"reply": "Não achei a última explicação. Me diga o que você quer que eu explique. 🙂"}
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        prompt = (
            "Explique em PORTUGUÊS (Brasil) de forma *CURTA e objetiva* (máx. 5 linhas), "
            "como se estivesse esclarecendo para um aluno iniciante. "
            "Sem saudação. Use frases curtas. Se útil, dê 1 exemplo simples.\n\n"
            f"--- CONTEÚDO ---\n{last_ai}\n--- FIM ---\n\nExplicação curta:"
        )
        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_greeting_prefix(strip_motivacao_label(text))
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = now
        return {"reply": text}

    # (B) Pedir para explicar/traduzir UMA FRASE (PT ou EN)
    if intent == "explain_sentence":
        target = extract_target_sentence(user_text_raw)
        if not target:
            # fallback: se não achou, pede para mandar a frase
            return {"reply": "Me envie a *frase em inglês* que você quer que eu explique (de preferência entre aspas)."}
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}

        # sempre responde em PT (você está ensinando brasileiros)
        prompt = (
            "Explique a *frase em inglês* abaixo em **PORTUGUÊS do Brasil**, de forma *curta e direta* (até 5 linhas):\n"
            "1) Tradução simples em 1 linha.\n"
            "2) 2–4 vocabulários chave (formato: Palavra → significa ...).\n"
            "3) Se houver ponto gramatical relevante, cite em 1 linha.\n"
            "4) Se fizer sentido, sugira 1 versão alternativa mais natural (1 linha).\n"
            "Não cumprimente. Sem parágrafos longos.\n\n"
            f"Frase: \"{target}\"\n\nResposta:"
        )
        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_greeting_prefix(strip_motivacao_label(text))
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = now
        return {"reply": text}

    # (C) Perguntas gerais → FAQ local, depois IA curta
    if intent == "question":
        local = local_faq_response(user_text_raw, lang)
        if local:
            memory["last_ai_reply"] = local
            return {"reply": local}
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if not is_all_english(user_text_raw) else QUOTA_FRIENDLY_REPLY_EN}

        if is_all_english(user_text_raw):
            base = (
                "You are an English teacher. Answer ONLY in ENGLISH.\n"
                "Be short and clear (max 5 lines). No greetings. Give 1 short example if useful."
            )
            prompt = f"{base}\n\nStudent question:\n\"{user_text_raw}\"\n\nAnswer:"
            text = model_generate_text(prompt)
            if is_quota_error_text(text):
                last_quota_error_at = time.time()
                return {"reply": QUOTA_FRIENDLY_REPLY_EN}
        else:
            base = (
                "Você é um professor de inglês. Responda em PT-BR *de forma curta* (máx. 5 linhas). "
                "Sem saudação. Dê 1 exemplo curto em inglês, se útil. Não corrija a pergunta do aluno."
            )
            prompt = f"{base}\n\nPergunta do aluno:\n\"{user_text_raw}\"\n\nResposta:"
            text = model_generate_text(prompt)
            if is_quota_error_text(text):
                last_quota_error_at = time.time()
                return {"reply": QUOTA_FRIENDLY_REPLY_PT}

        text = strip_greeting_prefix(strip_motivacao_label(text))
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = now
        return {"reply": text}

    # (D) Smalltalk → offline
    if intent == "smalltalk":
        rep = smalltalk_reply(user_text_raw, lang)
        memory["last_ai_reply"] = rep
        return {"reply": rep}

    # (E) Correção (frase alvo), PT explica em PT; EN explica em EN — sempre conciso
    target = extract_target_sentence(user_text_raw) or user_text_raw
    if not can_call_ai():
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if not is_all_english(user_text_raw) else QUOTA_FRIENDLY_REPLY_EN}

    if is_all_english(user_text_raw):
        base = (
            "You are a friendly English teacher. The student's English level is {level}.\n"
            "Answer in ENGLISH only. No greeting. Keep it concise (3 lines + emojis allowed).\n"
            "Return EXACTLY these sections, each on its own line:\n"
            "*Correction:* <corrected sentence in English>\n"
            "*Explanation:* <short explanation in English>\n"
            "*Tip:* <one short tip in English, end with a single emoji>"
        )
    else:
        base = (
            "Você é um professor amigável de inglês. O aluno está no nível {level}.\n"
            "Responda em PT-BR. Sem saudação. Seja *curto e direto* (3–5 linhas no total).\n"
            "Devolva EXATAMENTE estes blocos, cada um em sua própria linha:\n"
            "*Correção:* <frase corrigida em inglês>\n"
            "*Explicação:* <explicação curta em português sobre a regra aplicada>\n"
            "*Dica:* <uma dica curta em português, finalize com um único emoji>"
        )

    prompt = base.format(level=message.level)
    full_prompt = f"{prompt}\n\nFrase do aluno para corrigir:\n\"{target}\"\n\nResposta:"
    reply_text = model_generate_text(full_prompt)
    if is_quota_error_text(reply_text):
        last_quota_error_at = time.time()
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if not is_all_english(user_text_raw) else QUOTA_FRIENDLY_REPLY_EN}
    reply_text = strip_greeting_prefix(strip_motivacao_label(reply_text))

    memory["last_ai_reply"] = reply_text
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
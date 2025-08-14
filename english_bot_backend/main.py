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

app = FastAPI(title="English WhatsApp Bot", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Memória volátil simples (reinicia a cada deploy)
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

# --------- Intent detection ----------
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

def classify_intent(text: str, lang: str) -> str:
    t = text.strip().lower()

    # pedido explícito de reexplicar em português
    if any(p in t for p in ["explica em português","explicar em português","em portugues","em português"]) \
       and any(w in t for w in ["pode","por favor","explica","explicar"]):
        return "explain_pt"

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

# --------- Extrair frase alvo para correção ----------
QUOTED_RE = re.compile(r'["“”\'‘’\u201c\u201d](.+?)["“”\'‘’\u201c\u201d]', re.DOTALL)

def extract_target_sentence(user_text: str) -> str | None:
    t = user_text.strip()

    # 1) Entre aspas
    m = QUOTED_RE.search(t)
    if m:
        return m.group(1).strip()

    # 2) Múltiplas linhas: usa a última linha como frase
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        cand = lines[-1]
        if len(cand.split()) >= 2:
            return cand

    # 3) Após marcador “essa frase está correta?” / “is this sentence correct?”
    lower = t.lower()
    markers = [
        "essa frase está correta", "essa frase esta correta",
        "está correto", "esta correto",
        "is this sentence correct", "please correct"
    ]
    for mk in markers:
        if mk in lower:
            after = t[lower.find(mk)+len(mk):].strip(" :.-\n")
            if after:
                return after
    return None

# --------- FAQ local ----------
def local_faq_response(text: str, lang: str):
    t = text.lower().strip()

    def pick(pt, en):
        return pt if lang.startswith("pt") else en

    if (re.search(r"\bmake\b", t) and re.search(r"\bdo\b", t)) and ("diferen" in t or "difference" in t):
        return pick(
            "Diferença *make x do*: use *make* para criar/produzir algo (*make a cake*), "
            "e *do* para tarefas/atividades gerais (*do homework*). 👉 Pratique: *I make breakfast, and I do the dishes.*",
            "Difference *make vs do*: use *make* to create/produce something (*make a cake*), "
            "and *do* for general tasks/activities (*do homework*). 👉 Practice: *I make breakfast, and I do the dishes.*",
        )
    if "used to" in t or "use to" in t or ("significa" in t and "used to" in t):
        return pick(
            "*used to* fala de hábitos/situações do passado que não são mais verdadeiros: "
            "*I used to play soccer.* 👉 Pratique: *I used to ______ every weekend.*",
            "*used to* refers to past habits/situations that are no longer true: "
            "*I used to play soccer.* 👉 Practice: *I used to ______ every weekend.*",
        )
    if ("since" in t and "for" in t) or ("desde" in t and "por" in t):
        return pick(
            "*since* + ponto no tempo: *since 2019*; *for* + duração: *for two years*. "
            "👉 Pratique: *I have lived here since 2019 / for two years.*",
            "*since* + starting point: *since 2019*; *for* + duration: *for two years*. "
            "👉 Practice: *I have lived here since 2019 / for two years.*",
        )
    if re.search(r"\b(a|an|the)\b", t) and any(w in t for w in ["usar","use","article","artigo"]):
        return pick(
            "*a* (som de consoante), *an* (som de vogal); *the* quando é específico/conhecido. "
            "👉 Pratique: *I saw a cat. The cat was cute.*",
            "*a* before consonant sound, *an* before vowel sound; *the* when specific/known. "
            "👉 Practice: *I saw a cat. The cat was cute.*",
        )
    if ("much" in t and "many" in t) or ("muito" in t and "muitos" in t):
        return pick(
            "*many* + contáveis (*many books*); *much* + incontáveis (*much water*). "
            "👉 Pratique: *How many friends do you have? / How much time do we have?*",
            "*many* with countables (*many books*); *much* with uncountables (*much water*). "
            "👉 Practice: *How many friends do you have? / How much time do we have?*",
        )
    if re.search(r"\b(in|on|at)\b", t) and any(w in t for w in ["time","tempo","quando","when"]):
        return pick(
            "*in* (meses/anos), *on* (dias/datas), *at* (horas). "
            "👉 Pratique: *The class is on Tuesday at 8 am in May.*",
            "*in* (months/years), *on* (days/dates), *at* (times). "
            "👉 Practice: *The class is on Tuesday at 8 am in May.*",
        )
    if any(w in t for w in ["comparative","superlative","comparativo","superlativo"]):
        return pick(
            "Curtos: *-er/-est* (tall → taller/tallest). Longos: *more/most* (interesting → more/most interesting). "
            "👉 Pratique: *This book is more interesting than that one.*",
            "Short: *-er/-est* (tall → taller/tallest). Long: *more/most* (interesting → more/most interesting). "
            "👉 Practice: *This book is more interesting than that one.*",
        )
    if "present perfect" in t or ("have" in t and "past" in t) or "pretérito perfeito" in t:
        return pick(
            "*Present perfect* = experiência/resultado até agora (*I have seen it*). "
            "*Simple past* = momento terminado no passado (*I saw it yesterday*). "
            "👉 Pratique: *I have visited London, but I visited Paris last year.*",
            "*Present perfect* = experience/result up to now (*I have seen it*). "
            "*Simple past* = finished time in the past (*I saw it yesterday*). "
            "👉 Practice: *I have visited London, but I visited Paris last year.*",
        )
    return None

# --------- Smalltalk ----------
RESP_PT = {
    "bom_dia": [
        "☀️ *Good morning!* Bora começar o dia com 1 frase em inglês? Me manda que eu corrijo.",
        "Bom dia! 🌞 Que tal praticar? Escreva *uma* frase curta em inglês e eu te ajudo.",
        "Good morning! ✨ Se quiser, já te passo um mini desafio. É só dizer *#desafio*."
    ],
    "boa_tarde": [
        "🌤️ *Good afternoon!* Me manda uma frase em inglês e eu te retorno com correção e dica.",
        "Boa tarde! Vamos praticar rapidinho? Uma frase em inglês e eu explico o porquê. 😉",
    ],
    "boa_noite": [
        "🌙 *Good evening!* Topa uma última prática do dia? Envie uma frase em inglês.",
        "Boa noite! 😴 Antes de encerrar, manda *uma* frase que eu corrijo em 1 min.",
    ],
    "saudacao": [
        "Hey! 👋 Vamos praticar? Mande uma frase em inglês que eu corrijo com *explicação e dica*.",
        "Olá! 🙌 Se quiser, pergunte algo de gramática que eu explico com exemplos.",
        "Hi! 🙂 Eu também faço *quiz* se você mandar *#quiz*."
    ],
    "tudo_bem": [
        "Tudo certo por aqui! 😄 E aí, bora praticar uma frase em inglês?",
        "Tudo bem! 💪 Qual dúvida de inglês você quer tirar hoje?",
    ],
    "agradecimento": [
        "Tamo junto! 🙏 Quando quiser, manda outra frase.",
        "De nada! 😊 Quer tentar um *mini desafio*? Envie *#desafio*.",
    ],
    "despedida": [
        "Até mais! 👋 Se quiser revisar depois, é só me chamar.",
        "See you! 👀 Volta quando quiser praticar mais.",
    ],
    "default": [
        "👍 Bora praticar! Envie uma frase em inglês para eu corrigir ou faça uma pergunta de gramática.",
        "🚀 Partiu inglês! Manda uma frase ou dúvida que eu te ajudo.",
    ],
}
RESP_EN = {
    "good_morning": [
        "☀️ Good morning! Send me one sentence to correct today.",
        "Morning! 🌞 I can give you a quick tip if you send a sentence."
    ],
    "good_afternoon": [
        "🌤️ Good afternoon! Ready for a quick practice?",
        "Hey! Send me a sentence and I'll correct it with a short tip. 😉"
    ],
    "good_evening": [
        "🌙 Good evening! One last practice before bed?",
        "Evening! Send one sentence and I’ll fix it up."
    ],
    "greeting": [
        "Hi! 👋 Send a sentence to correct or ask a grammar question.",
        "Hello! 🙌 I can also run a quick *#quiz* for you."
    ],
    "thanks": [
        "You're welcome! 🙏 Got another sentence?",
        "Anytime! 😊 Want a *mini challenge*? Send *#desafio*."
    ],
    "bye": [
        "See you! 👋 Come back anytime to practice.",
        "Bye! 👀 I'll be here when you need me."
    ],
    "default": [
        "👍 Let’s practice! Send me an English sentence to correct or ask a grammar question.",
        "🚀 Ready when you are—one sentence or any question."
    ],
}
def smalltalk_reply(text: str, lang: str) -> str:
    t = text.lower()
    if lang.startswith("pt"):
        if "bom dia" in t: return random.choice(RESP_PT["bom_dia"])
        if "boa tarde" in t: return random.choice(RESP_PT["boa_tarde"])
        if "boa noite" in t: return random.choice(RESP_PT["boa_noite"])
        if any(s in t for s in ["tudo bem","tudo bom"]): return random.choice(RESP_PT["tudo_bem"])
        if any(s in t for s in ["obrigado","valeu","brigado"]): return random.choice(RESP_PT["agradecimento"])
        if any(s in t for s in ["tchau","falou","até mais","ate mais"]): return random.choice(RESP_PT["despedida"])
        if any(s in t for s in ["oi","olá","ola","salve","eai","e aí","e ai"]): return random.choice(RESP_PT["saudacao"])
        return random.choice(RESP_PT["default"])
    else:
        if "good morning" in t or "morning" in t: return random.choice(RESP_EN["good_morning"])
        if "good afternoon" in t: return random.choice(RESP_EN["good_afternoon"])
        if "good night" in t or "good evening" in t or "evening" in t: return random.choice(RESP_EN["good_evening"])
        if any(s in t for s in ["thanks","thank you","thx"]): return random.choice(RESP_EN["thanks"])
        if any(s in t for s in ["bye","goodbye","see ya","see you"]): return random.choice(RESP_EN["bye"])
        if any(s in t for s in ["hello","hi","hey"]): return random.choice(RESP_EN["greeting"])
        return random.choice(RESP_EN["default"])

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

    last_call = memory.get("last_call_ts", 0.0)
    def can_call_ai() -> bool:
        return (now - last_call) >= USER_COOLDOWN_SECONDS and (now - last_quota_error_at) >= 30

    # comando utilitário
    if user_text.lower() == "#resetar":
        user_memory.pop(message.phone, None)
        return {"reply": "🔄 Sua memória foi resetada com sucesso! Pode recomeçar."}

    # intenção
    lang = safe_detect_lang(user_text_raw)
    intent = classify_intent(user_text_raw, lang)

    # (0) reexplicar em português o que foi dito antes
    if intent == "explain_pt":
        last_ai = memory.get("last_ai_reply", "")
        if not last_ai:
            return {"reply": "Não encontrei a última explicação. Envie a frase ou pergunta novamente. 🙂"}
        if not can_call_ai():
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        prompt = (
            "Explique em PORTUGUÊS (Brasil), de forma simples, o conteúdo abaixo, "
            "como se você estivesse esclarecendo para um aluno iniciante. "
            "Não cumprimente. Seja direto. Use exemplos curtos.\n\n"
            f"--- CONTEÚDO ---\n{last_ai}\n--- FIM ---\n\nExplicação em português:"
        )
        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_greeting_prefix(strip_motivacao_label(text))
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = now
        return {"reply": text}

    # (1) pergunta → FAQ local primeiro
    if intent == "question":
        local = local_faq_response(user_text_raw, lang)
        if local:
            memory["last_ai_reply"] = local
            return {"reply": local}

        if is_all_english(user_text_raw):
            if not can_call_ai():
                return {"reply": QUOTA_FRIENDLY_REPLY_EN}
            base = (
                "You are an English teacher. Answer ONLY in ENGLISH.\n"
                "Explain clearly what the student asked, with short examples when useful.\n"
                "Do NOT greet. Do NOT correct the student's question. Focus on the topic.\n"
                "Finish with one single practice sentence (one line)."
            )
            prompt = f"{base}\n\nStudent question:\n\"{user_text_raw}\"\n\nAnswer:"
            text = model_generate_text(prompt)
            if is_quota_error_text(text):
                last_quota_error_at = time.time()
                return {"reply": QUOTA_FRIENDLY_REPLY_EN}
            text = strip_greeting_prefix(strip_motivacao_label(text))
            memory["last_ai_reply"] = text
            memory["last_call_ts"] = now
            return {"reply": text}
        else:
            if not can_call_ai():
                return {"reply": QUOTA_FRIENDLY_REPLY_PT}
            base = (
                "Você é um professor de inglês. Responda em português (Brasil).\n"
                "Explique de forma clara e prática o que o aluno perguntou, com exemplos curtos em inglês quando útil.\n"
                "NÃO cumprimente. NÃO corrija a pergunta do aluno. Foque na explicação do tema.\n"
                "No final, sugira 1 frase de exemplo para o aluno praticar (somente 1 linha)."
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

    # (2) smalltalk → offline
    if intent == "smalltalk":
        rep = smalltalk_reply(user_text_raw, lang)
        memory["last_ai_reply"] = rep
        return {"reply": rep}

    # (3) correção → extrai frase alvo; PT responde em PT, EN responde em EN
    target = extract_target_sentence(user_text_raw) or user_text_raw
    if not can_call_ai():
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if not is_all_english(user_text_raw) else QUOTA_FRIENDLY_REPLY_EN}

    if is_all_english(user_text_raw):
        base = (
            "You are a friendly English teacher. The student's English level is {level}.\n"
            "Answer in ENGLISH only. Do NOT greet. Do NOT translate to Portuguese.\n"
            "Return EXACTLY these sections, each on its own line:\n"
            "*Correction:* <corrected sentence in English>\n"
            "*Explanation:* <short explanation in English of the grammar or usage>\n"
            "*Tip:* <one short tip in English, end with a single emoji>"
        )
    else:
        base = (
            "Você é um professor amigável de inglês. O aluno está no nível {level}.\n"
            "Responda em PORTUGUÊS (Brasil). NÃO cumprimente. NÃO traduza a frase corrigida para o português.\n"
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
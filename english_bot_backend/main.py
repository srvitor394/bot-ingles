# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import google.generativeai as genai
import os, re, time, random, unicodedata

# ===================== CONFIG =====================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENV_MODEL = os.getenv("GEMINI_MODEL_NAME", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = ENV_MODEL or "gemini-1.5-flash"
else:
    GEMINI_MODEL_NAME = ""  # sem chave -> modo offline

app = FastAPI(title="English WhatsApp Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===================== ESTADO =====================
user_memory: dict[str, dict] = {}
last_quota_error_at = 0.0
USER_COOLDOWN_SECONDS = 6

# ===================== MODELOS/PAYLOADS =====================
class Message(BaseModel):
    user_message: str
    level: str = "basic"
    phone: str = "unknown"

class ResetReq(BaseModel):
    phone: str

class WhatsAppMessage(BaseModel):
    from_number: str
    body: str

# ===================== HELPERS GERAIS =====================
QUOTA_FRIENDLY_REPLY_PT = "⚠️ Bati no limite gratuito diário da IA por agora. Tente de novo mais tarde. 🙏"
QUOTA_FRIENDLY_REPLY_EN = "⚠️ I just hit today’s free AI quota. Please try again later. 🙏"

def _unaccent(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

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

def strip_headers(text: str) -> str:
    # remove rótulos indesejados (saudação/motivação)
    text = re.sub(r"(?im)^\s*(ol[áa]|oi|hello|hi|hey)[!,.…]*\s*", "", text).strip()
    text = re.sub(r"(?im)^\s*\*?\s*motiv[aá]?[cç][aã]o\s*\*?\s*:\s*", "", text).strip()
    return text

def can_call_ai(memory: dict) -> bool:
    now = time.time()
    last_user = memory.get("last_call_ts", 0.0)
    global last_quota_error_at
    return (now - last_user) >= USER_COOLDOWN_SECONDS and (now - last_quota_error_at) >= 30

# ===================== DETECÇÃO DE LÍNGUA/INGLÊS =====================
def looks_english(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    lang = safe_detect_lang(s)
    if lang == "en":
        return True
    # fallback leve: ASCII/alpha
    letters = sum(ch.isalpha() for ch in s)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in s)
    return ascii_letters >= letters * 0.8 and letters >= 3

QUOTED_RE = re.compile(r'["“”\'‘’\u201c\u201d](.+?)["“”\'‘’\u201c\u201d]', re.DOTALL)

def extract_english_sentence(user_text: str) -> str | None:
    # 1) trecho entre aspas
    m = QUOTED_RE.search(user_text)
    if m and looks_english(m.group(1)):
        return m.group(1).strip()
    # 2) última linha
    lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
    if lines and looks_english(lines[-1]):
        return lines[-1]
    # 3) depois de marcadores
    low = _unaccent(user_text.lower())
    for mk in [
        "essa frase esta correta", "esta correto", "nao entendi essa frase",
        "is this sentence correct", "please correct", "explain this sentence", "what does it mean"
    ]:
        if mk in low:
            after = user_text[low.find(mk)+len(mk):].strip(" :.-\n\t")
            m2 = QUOTED_RE.search(after)
            if m2 and looks_english(m2.group(1)):
                return m2.group(1).strip()
            if looks_english(after):
                return after
    return None

# ===================== INTENTS =====================
PT_QUESTION_WORDS = {
    "o que","oq","qual","quais","como","quando","onde","por que","porque","por quê",
    "pra que","para que","diferença","significa","me explica","explica",
    "é correto","esta certo","está certo","está errado","devo usar","exemplo de","como usar",
    "essa frase está correta","pode corrigir","corrigir","corrige","ver se está certo"
}
EN_QUESTION_WORDS = {
    "what","which","how","when","where","why","difference","mean","meaning",
    "should i","is it correct","am i","can i","could i","what's","whats","example of","how to use",
    "is this sentence correct","please correct","explain","explain this"
}

SMALL_PT = {"obrigado","valeu","blz","beleza","tmj","ok","bom dia","boa tarde","boa noite","tudo bem","tudo bom","oi","olá","salve","até mais","tchau"}
SMALL_EN = {"thanks","thank you","ok","cool","nice","good morning","good afternoon","good evening","hi","hello","hey","see ya","bye","goodbye","see you"}

TOPIC_KEYWORDS = {
    "verbo to be": ["verbo to be","to be","am is are"],
    "simple past": ["simple past","passado simples","did","ed verbs"],
    "present continuous": ["present continuous","presente continuo","ing agora"],
    "articles": ["articles","artigos","a an the"],
    "pronouns": ["pronomes","pronouns"],
    "prepositions": ["preposicoes","prepositions"],
}

def find_topic(user_text: str) -> str | None:
    t = _unaccent(user_text.lower())
    for topic, kws in TOPIC_KEYWORDS.items():
        for k in kws:
            if _unaccent(k) in t:
                return topic
    # Diferença X vs Y
    if ("diferen" in t or "difference" in t) and ("make" in t and "do" in t):
        return "make vs do"
    if ("diferen" in t or "difference" in t) and ("since" in t and "for" in t):
        return "since vs for"
    return None

def ask_explain_sentence(user_text: str, lang: str) -> bool:
    t = _unaccent(user_text.lower())
    if lang.startswith("pt"):
        triggers = ["nao entendi essa frase", "pode me explicar", "explica em portugues", "o que significa", "o que quer dizer"]
    else:
        triggers = ["explain this sentence", "what does it mean", "translate this", "can you explain this"]
    return any(x in t for x in triggers) or bool(QUOTED_RE.search(user_text))

def classify_intent(user_text: str, lang: str) -> str:
    t = _unaccent(user_text.lower()).strip()

    # Reexplicar a última resposta do bot em PT
    if ("explica em portugues" in t or "reexplica" in t) and ("resposta" in t or "acima" in t or "ultima" in t):
        return "explain_previous_pt"

    # Explicar uma frase específica (com aspas ou trecho EN)
    if ask_explain_sentence(user_text, lang):
        return "explain_sentence"

    # Topic lesson
    if find_topic(user_text):
        return "topic_lesson"

    # Pergunta geral
    if "?" in t:
        return "question"
    if lang.startswith("pt"):
        if any(w in t for w in PT_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALL_PT): return "smalltalk"
    else:
        if any(w in t for w in EN_QUESTION_WORDS): return "question"
        if any(w in t for w in SMALL_EN): return "smalltalk"

    # Frase para correção (se aparenta ser inglês)
    eng_snippet = extract_english_sentence(user_text)
    if eng_snippet or looks_english(user_text):
        return "correction"

    # fallback
    return "smalltalk"

# ===================== CONTEÚDO LOCAL (AULAS) =====================
LESSONS_PT = {
    "verbo to be": (
        "🧩 *Verbo To Be (am/is/are)*\n"
        "• Uso: identidade, estado, localização. \n"
        "• Estruturas: I *am* / You *are* / He/She/It *is* / We/They *are*.\n"
        "• Negativa: I *am not*, He *isn't*, They *aren't*.\n"
        "• Pergunta: *Are* you ok?  *Is* she home?\n"
        "Ex.: *I am a student.* / *She is happy.* / *They are in Brazil.*\n"
        "👉 Pratique: escreva 2 frases (uma afirmativa e uma pergunta)."
    ),
    "simple past": (
        "⏳ *Simple Past (passado simples)*\n"
        "• Ações terminadas no passado. \n"
        "• Regulares: verbo + *-ed* (play → played). Irregulares: go → went, have → had.\n"
        "• Negativa: did + not + verbo base (I *didn't go*). Pergunta: *Did* you go?\n"
        "Ex.: *She watched a movie yesterday.* / *I went to school.*\n"
        "👉 Pratique: conte algo que fez ontem em 1 frase."
    ),
    "present continuous": (
        "🔄 *Present Continuous (ação em progresso agora)*\n"
        "• Estrutura: *am/is/are* + verbo + *-ing*.\n"
        "• Uso: ações acontecendo agora / planos próximos. \n"
        "Ex.: *I am studying now.* / *We are traveling this weekend.*\n"
        "👉 Pratique: diga o que você está fazendo neste momento."
    ),
    "articles": (
        "📚 *Articles (a/an/the)*\n"
        "• *a* antes de som de consoante; *an* antes de som de vogal. \n"
        "• *the* quando é específico/conhecido.\n"
        "Ex.: *a cat*, *an apple*, *the book on the table*.\n"
        "👉 Pratique: escreva 2 frases usando *a/an* e 1 com *the*."
    ),
    "make vs do": (
        "🛠️ *Make x Do*\n"
        "• *make*: criar/produzir algo (*make a cake*).\n"
        "• *do*: tarefas/atividades (*do homework*).\n"
        "Ex.: *I make breakfast and I do the dishes.*\n"
        "👉 Pratique: uma frase com *make* e outra com *do*."
    ),
    "since vs for": (
        "⏱️ *Since x For*\n"
        "• *since* + ponto de início (since 2019).  *for* + duração (for two years).\n"
        "Ex.: *I have lived here since 2019 / for two years.*\n"
        "👉 Pratique: crie 1 frase com *since* e 1 com *for*."
    ),
}

# ===================== SMALLTALK =====================
SMALLTALK_PT = [
    "👍 Bora praticar! Envie uma frase em inglês para corrigir ou faça uma dúvida de gramática.",
    "🚀 Partiu inglês? Manda uma frase que eu corrijo e explico rapidinho.",
    "🙌 Se quiser, posso te dar um mini-desafio. É só mandar *#desafio*."
]
SMALLTALK_EN = [
    "👍 Let’s practice! Send me one sentence to correct or ask any grammar question.",
    "🚀 Ready when you are — I’ll correct and give a quick tip.",
]

def smalltalk_reply(lang: str) -> str:
    return random.choice(SMALLTALK_PT if lang.startswith("pt") else SMALLTALK_EN)

# ===================== PROMPTS =====================
def prompt_explain_sentence_pt(sentence: str) -> str:
    return (
        "Explique a *frase em inglês* abaixo em **PT-BR**, de forma *curta e clara* (até 5 linhas):\n"
        "1) Tradução simples (1 linha).\n"
        "2) 2–4 vocabulários chave (formato: Palavra → significado curto).\n"
        "3) 1 ponto gramatical, se houver.\n"
        "4) Opcional: 1 reescrita mais natural/educada.\n"
        "Sem saudação. Sem parágrafos longos.\n\n"
        f"Frase: \"{sentence}\"\n\nResposta:"
    )

def prompt_correction_pt(level: str, sentence: str) -> str:
    return (
        "Você é um professor amigável de inglês. Responda em PT-BR, curto e direto. "
        "Não cumprimente. Não traduza a frase corrigida.\n"
        "Devolva EXATAMENTE estes blocos, cada um em sua própria linha:\n"
        "*Correção:* <frase corrigida em inglês>\n"
        "*Explicação:* <regra/razão em português (1–2 linhas)>\n"
        "*Dica:* <uma dica curta em português, finalize com um emoji>\n\n"
        f"Nível do aluno: {level}\n"
        f"Frase do aluno: \"{sentence}\"\n\nResposta:"
    )

def prompt_correction_en(level: str, sentence: str) -> str:
    return (
        "You are a friendly English teacher. Answer in ENGLISH only. "
        "Be concise (3–5 lines). No greeting. Do NOT translate the corrected sentence.\n"
        "Return EXACTLY these sections, each on its own line:\n"
        "*Correction:* <corrected sentence>\n"
        "*Explanation:* <short reason/rule>\n"
        "*Tip:* <one short tip, end with a single emoji>\n\n"
        f"Student level: {level}\n"
        f"Student sentence: \"{sentence}\"\n\nAnswer:"
    )

def prompt_question_pt(question: str) -> str:
    return (
        "Você é professor de inglês. Explique em PT-BR de forma *simples e prática*, no máx. 5 linhas. "
        "Dê 1 exemplo curtinho em inglês se ajudar. Sem saudação. Não corrija a pergunta do aluno.\n\n"
        f"Pergunta do aluno: \"{question}\"\n\nResposta:"
    )

def prompt_question_en(question: str) -> str:
    return (
        "You are an English teacher. Answer ONLY in ENGLISH, clearly and briefly (max 5 lines). "
        "Give 1 short example if helpful. No greetings.\n\n"
        f"Student question: \"{question}\"\n\nAnswer:"
    )

def prompt_reexplain_pt(text_to_explain: str) -> str:
    return (
        "Reexplica em PT-BR, *curto e objetivo* (até 5 linhas), como se fosse para um iniciante. "
        "Sem saudação. Se útil, inclua 1 exemplo simples.\n\n"
        f"Conteúdo a reexplicar:\n{text_to_explain}\n\nReexplicação curta:"
    )

# ===================== ENDPOINTS BÁSICOS =====================
@app.get("/")
def root():
    return {"message": "OLÁ, MUNDO!", "service": "English WhatsApp Bot"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== LÓGICA PRINCIPAL =====================
@app.post("/correct")
async def correct_english(message: Message):
    global last_quota_error_at
    user_text_raw = message.user_message or ""
    user_text = user_text_raw.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    memory = user_memory.setdefault(message.phone, {})
    lang_msg = safe_detect_lang(user_text_raw)
    intent = classify_intent(user_text_raw, lang_msg)

    # RESET
    if user_text.lower() == "#resetar":
        user_memory.pop(message.phone, None)
        return {"reply": "🔄 Memória resetada. Bora recomeçar!"}

    # A) REEXPLICAR ÚLTIMA RESPOSTA EM PT
    if intent == "explain_previous_pt":
        last_ai = memory.get("last_ai_reply", "")
        if not last_ai:
            return {"reply": "Não achei a última explicação. Me diga exatamente o que quer reexplicar. 🙂"}
        if not can_call_ai(memory):
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = model_generate_text(prompt_reexplain_pt(last_ai))
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_headers(text)
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = time.time()
        return {"reply": text}

    # B) EXPLICAR UMA FRASE ESPECÍFICA (sempre em PT)
    if intent == "explain_sentence":
        sentence = extract_english_sentence(user_text_raw)
        if not sentence:
            return {"reply": "Me envie a *frase em inglês* que você quer que eu explique (de preferência entre aspas)."}
        if not can_call_ai(memory):
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = model_generate_text(prompt_explain_sentence_pt(sentence))
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_headers(text)
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = time.time()
        return {"reply": text}

    # C) AULA DE TÓPICO (usa acervo local; se não tiver, IA curta)
    if intent == "topic_lesson":
        topic = find_topic(user_text_raw)
        if topic and topic in LESSONS_PT:
            reply = LESSONS_PT[topic]
            memory["last_ai_reply"] = reply
            return {"reply": reply}
        # tópico não mapeado -> IA
        if not can_call_ai(memory):
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = model_generate_text(prompt_question_pt(user_text_raw))
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        text = strip_headers(text)
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = time.time()
        return {"reply": text}

    # D) PERGUNTAS GERAIS
    if intent == "question":
        # FAQ leve (não gasta cota) – reciclamos parte das lições
        topic = find_topic(user_text_raw)
        if topic and topic in LESSONS_PT:
            reply = LESSONS_PT[topic]
            memory["last_ai_reply"] = reply
            return {"reply": reply}

        if not can_call_ai(memory):
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN}

        if lang_msg.startswith("en"):
            prompt = prompt_question_en(user_text_raw)
        else:
            prompt = prompt_question_pt(user_text_raw)

        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN}
        text = strip_headers(text)
        memory["last_ai_reply"] = text
        memory["last_call_ts"] = time.time()
        return {"reply": text}

    # E) SMALLTALK (nunca usa IA)
    if intent == "smalltalk":
        reply = smalltalk_reply(lang_msg)
        memory["last_ai_reply"] = reply
        return {"reply": reply}

    # F) CORREÇÃO DE FRASE (3 blocos)
    sentence = extract_english_sentence(user_text_raw) or user_text_raw
    if not can_call_ai(memory):
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN}

    if lang_msg.startswith("en") and "?" not in user_text_raw:
        prompt = prompt_correction_en(message.level, sentence)
    else:
        # Se o usuário é PT (ou misto), devolvemos em PT
        prompt = prompt_correction_pt(message.level, sentence)

    text = model_generate_text(prompt)
    if is_quota_error_text(text):
        last_quota_error_at = time.time()
        return {"reply": QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN}

    text = strip_headers(text)
    memory["last_ai_reply"] = text
    memory["last_call_ts"] = time.time()
    return {"reply": text}

# ===================== UTILIDADES =====================
@app.post("/resetar")
async def resetar_memoria(req: ResetReq):
    user_memory.pop(req.phone, None)
    return {"status": "ok"}

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(msg: WhatsAppMessage):
    payload = Message(user_message=msg.body, phone=msg.from_number)
    result = await correct_english(payload)
    return {"to": msg.from_number, "reply": result.get("reply", "")}
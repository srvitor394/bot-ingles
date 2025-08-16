# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import google.generativeai as genai
import os, re, time, random, unicodedata, json

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
    except (LangDetectException, Exception):
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
        if text.strip().startswith("```json"):
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text.strip() if text else "(sem resposta do modelo)"
    except Exception as e:
        return f"⚠️ Erro ao consultar o modelo: {str(e)}"

def strip_headers(text: str) -> str:
    text = re.sub(r"(?im)^\s*(ol[áa]|oi|hello|hi|hey)[!,.…]*\s*", "", text).strip()
    text = re.sub(r"(?im)^\s*\*?\s*motiv[aá]?[cç][aã]o\s*\*?\s*:\s*", "", text).strip()
    return text

def can_call_ai(memory: dict) -> bool:
    now = time.time()
    last_user = memory.get("last_call_ts", 0.0)
    global last_quota_error_at
    return (now - last_user) >= USER_COOLDOWN_SECONDS and (now - last_quota_error_at) >= 30

# ===================== DETECÇÃO E EXTRAÇÃO =====================
QUOTED_RE = re.compile(r'["“”\'‘’\u201c\u201d](.+?)["“”\'‘’\u201c\u201d]', re.DOTALL)

def looks_english(s: str) -> bool:
    s = s.strip()
    if not s: return False
    lang = safe_detect_lang(s)
    if lang == "en": return True
    letters = sum(ch.isalpha() for ch in s)
    if letters < 3: return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in s)
    return ascii_letters >= letters * 0.8

def extract_english_sentence(user_text: str) -> str | None:
    m = QUOTED_RE.search(user_text)
    if m and looks_english(m.group(1)):
        return m.group(1).strip()
    lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
    if lines and looks_english(lines[-1]):
        return lines[-1]
    low = _unaccent(user_text.lower())
    for mk in [
        "essa frase esta correta", "esta correto", "nao entendi essa frase",
        "is this sentence correct", "please correct", "explain this sentence", "what does it mean"
    ]:
        if mk in low:
            after = user_text[low.find(mk)+len(mk):].strip(" :.-\n\t")
            if looks_english(after):
                return after
    return None

# ===================== INTENTS (CLASSIFICADOR) =====================
INTENT_KEYWORDS = {
    "greeting": ["bom dia", "boa tarde", "boa noite", "oi", "ola", "hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
    "chit_chat": ["obrigado", "valeu", "ok", "blz", "beleza", "thanks", "thank you", "cool", "nice"],
    "correction": ["corrigir", "corrige", "esta correto", "is this correct", "please correct", "essa frase esta correta"],
    "explain_sentence": ["nao entendi", "explica", "significa", "quer dizer", "what does it mean", "explain this"],
    "question": ["o que", "qual", "como", "quando", "diferença", "what", "how", "why", "difference"],
}

TOPIC_KEYWORDS = {
    "verbo to be": ["verbo to be", "to be", "am is are"],
    "simple past": ["simple past", "passado simples", "did", "ed verbs"],
    "present continuous": ["present continuous", "presente continuo", "ing agora"],
    "articles": ["articles", "artigos", "a an the"],
    "prepositions": ["preposicoes", "prepositions", "in on at"],
    "make vs do": ["make", "do", "diferenca make do"],
    "since vs for": ["since", "for", "diferenca since for"],
}

def classify_intent_by_rules(user_text: str) -> tuple[str | None, str | None]:
    t_norm = _unaccent(user_text.lower()).strip()

    if t_norm in INTENT_KEYWORDS["greeting"]:
        return "greeting", t_norm

    if t_norm == "#resetar":
        return "reset", None

    if ("reexplica" in t_norm or "explica de novo" in t_norm) and ("resposta" in t_norm or "acima" in t_norm):
        return "reexplain_last", None

    for topic, kws in TOPIC_KEYWORDS.items():
        for k in kws:
            if _unaccent(k) in t_norm:
                return "topic_lesson", topic

    eng_sentence = extract_english_sentence(user_text)
    if eng_sentence:
        if any(kw in t_norm for kw in INTENT_KEYWORDS["explain_sentence"]):
            return "explain_sentence", eng_sentence
        return "correction", eng_sentence
    
    # Adicionado para pegar casos como "Essa frase está correta? She go to school"
    if any(kw in t_norm for kw in INTENT_KEYWORDS["correction"]):
        return "correction", user_text

    if "?" in t_norm or any(kw in t_norm for kw in INTENT_KEYWORDS["question"]):
        return "question", user_text

    if looks_english(user_text):
        return "correction", user_text

    if any(kw in t_norm for kw in INTENT_KEYWORDS["chit_chat"]):
        return "chit_chat", None

    return None, None

# ===================== CONTEÚDO LOCAL (AULAS) =====================
LESSONS_PT = {
    "verbo to be": (
        "🧩 *Verbo To Be (am/is/are)*\n"
        "• Uso: identidade, estado, localização. \n"
        "• Estruturas: I *am* / You *are* / He/She/It *is* / We/They *are*.\n"
        "• Negativa: I *am not*, He *isn't*, They *aren't*.\n"
        "• Pergunta: *Are* you ok?  *Is* she home?\n"
        "Ex.: *I am a student.* / *She is happy.* / *They are in Brazil.*\n"
    ),
    "simple past": (
        "⏳ *Simple Past (passado simples)*\n"
        "• Ações terminadas no passado. \n"
        "• Regulares: verbo + *-ed* (play → played). Irregulares: go → went, have → had.\n"
        "• Negativa: did + not + verbo base (I *didn't go*). Pergunta: *Did* you go?\n"
        "Ex.: *She watched a movie yesterday.* / *I went to school.*\n"
    ),
}

# ===================== RESPOSTAS NATURAIS =====================
def greeting_reply(greeting_text: str) -> str:
    greeting_text = _unaccent(greeting_text.lower())
    if "bom dia" in greeting_text:
        return "Bom dia! Tudo bem? 😊"
    if "boa tarde" in greeting_text:
        return "Boa tarde! Como vai? ✨"
    if "boa noite" in greeting_text:
        return "Boa noite! Espero que tenha tido um ótimo dia. 🌙"
    if any(s in greeting_text for s in ["oi", "ola", "hello", "hi", "hey"]):
        return random.choice(["Olá! 👋", "Oi, tudo bem?", "Hello! How can I help you today?"])
    return "Olá! 😊"

def chit_chat_reply(lang: str) -> str:
    if lang.startswith("pt"):
        return random.choice([
            "👍 Certo!",
            "Qualquer outra dúvida, é só chamar! 😉",
            "Disponha! Se precisar de mais alguma coisa, estou aqui."
        ])
    else:
        return random.choice(["You're welcome!", "Sure thing!", "Anytime! Let me know if you need anything else."])

# ===================== PROMPTS COMPLETOS =====================

def prompt_router_ai(user_message: str) -> str:
    return (
        "Você é um assistente que classifica a intenção de um aluno de inglês. Responda APENAS com um objeto JSON.\n"
        "Categorias de intenção: `correction`, `question`, `explain_sentence`, `greeting`, `chit_chat`.\n"
        "IMPORTANTE: Se a mensagem for APENAS uma saudação simples como 'oi', 'bom dia', 'hello', classifique como `greeting`.\n"
        "No JSON, inclua 'intent' e 'content' (a frase ou o tópico principal da pergunta).\n"
        f"Mensagem do aluno: \"{user_message}\"\n\n"
        "```json\n"
    )

# <<< MUDANÇA: CORPO DAS FUNÇÕES DE PROMPT RESTAURADO
def prompt_question_pt(question: str) -> str:
    return (
        "Você é um professor de inglês didático. Responda em PT-BR.\n"
        "Explique o tópico gramatical da pergunta de forma clara e estruturada. Use bullet points.\n"
        "A estrutura da resposta deve ser:\n"
        "1. **O que é**: Explicação simples (1-2 linhas).\n"
        "2. **Como usar**: Exemplos de afirmativa, negativa e pergunta.\n"
        "3. **Exemplos Práticos**: 2 frases de exemplo com tradução.\n"
        "Seja conciso. Sem saudação.\n\n"
        f"Dúvida do aluno: \"{question}\"\n\nResposta:"
    )

def prompt_question_en(question: str) -> str:
    return (
        "You are an English teacher. Answer in ENGLISH, clearly and briefly (max 5 lines). "
        "Give 1 short example if helpful. No greetings.\n\n"
        f"Student question: \"{question}\"\n\nAnswer:"
    )

def prompt_correction_pt(level: str, sentence: str) -> str:
    return (
        "Você é um professor amigável de inglês. Responda em PT-BR, curto e direto. "
        "Comece com uma nota positiva antes dos blocos (Ex: 'Boa tentativa!').\n"
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
        "You are a friendly English teacher. Answer in ENGLISH only. Be concise (3–5 lines). No greeting.\n"
        "Return EXACTLY these sections, each on its own line:\n"
        "*Correction:* <corrected sentence>\n"
        "*Explanation:* <short reason/rule>\n"
        "*Tip:* <one short tip, end with a single emoji>\n\n"
        f"Student level: {level}\n"
        f"Student sentence: \"{sentence}\"\n\nAnswer:"
    )

def prompt_explain_sentence_pt(sentence: str) -> str:
    return (
        "Explique a *frase em inglês* abaixo em **PT-BR**, de forma *curta e clara* (até 5 linhas):\n"
        "1) Tradução simples.\n"
        "2) 2–4 vocabulários chave (Palavra → significado).\n"
        "3) 1 ponto gramatical, se houver.\n"
        "Sem saudação.\n\n"
        f"Frase: \"{sentence}\"\n\nResposta:"
    )

def prompt_reexplain_pt(text_to_explain: str) -> str:
    return (
        "Reexplica em PT-BR, *curto e objetivo* (até 5 linhas), como se fosse para um iniciante. "
        "Sem saudação. Use 1 exemplo simples.\n\n"
        f"Conteúdo a reexplicar:\n{text_to_explain}\n\nReexplicação curta:"
    )

# ===================== ENDPOINTS BÁSICOS =====================
@app.get("/")
def root():
    return {"message": "OLÁ, MUNDO!", "service": "English WhatsApp Bot"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== LÓGICA PRINCIPAL (REFINADA) =====================
@app.post("/correct")
async def correct_english(message: Message):
    global last_quota_error_at
    user_text = (message.user_message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    phone = message.phone
    memory = user_memory.setdefault(phone, {})
    lang_msg = safe_detect_lang(user_text)

    # --- 1. CLASSIFICAR INTENÇÃO ---
    intent, content = classify_intent_by_rules(user_text)

    if not intent:
        if not can_call_ai(memory): return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        
        router_response_str = model_generate_text(prompt_router_ai(user_text))
        try:
            router_data = json.loads(router_response_str)
            intent = router_data.get("intent", "question")
            content = router_data.get("content", user_text)
        except (json.JSONDecodeError, TypeError):
            intent = "question"
            content = user_text
    
    # --- 2. EXECUTAR AÇÃO COM BASE NA INTENÇÃO ---
    reply = ""
    use_ai = False
    prompt = ""

    if intent == "reset":
        user_memory.pop(phone, None)
        reply = "🔄 Memória resetada. Bora recomeçar!"
    
    elif intent == "greeting":
        reply = greeting_reply(content or user_text)

    elif intent == "chit_chat":
        reply = chit_chat_reply(lang_msg)

    elif intent == "reexplain_last":
        last_ai = memory.get("last_ai_reply", "")
        if not last_ai:
            reply = "Não achei a última explicação. 🙂"
        else:
            use_ai = True
            prompt = prompt_reexplain_pt(last_ai)
            
    elif intent == "topic_lesson":
        if content and content in LESSONS_PT:
            reply = LESSONS_PT[content]
        else:
            use_ai = True
            prompt = prompt_question_pt(content or user_text)

    elif intent == "explain_sentence":
        use_ai = True
        prompt = prompt_explain_sentence_pt(content or user_text)

    elif intent == "question":
        use_ai = True
        prompt = prompt_question_pt(content or user_text) if not lang_msg.startswith("en") else prompt_question_en(content or user_text)

    elif intent == "correction":
        use_ai = True
        # <<< MUDANÇA: Garante que a frase a ser corrigida seja extraída corretamente.
        sentence_to_correct = extract_english_sentence(user_text) or (content if looks_english(content) else user_text)
        prompt = prompt_correction_pt(message.level, sentence_to_correct) if not lang_msg.startswith("en") else prompt_correction_en(message.level, sentence_to_correct)

    else:
        use_ai = True
        prompt = prompt_question_pt(user_text)

    # --- 3. PROCESSAR RESPOSTA (SE USAR IA) ---
    if use_ai:
        if not can_call_ai(memory):
            return {"reply": QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN}
        text = model_generate_text(prompt)
        if is_quota_error_text(text):
            last_quota_error_at = time.time()
            reply = QUOTA_FRIENDLY_REPLY_PT if not lang_msg.startswith('en') else QUOTA_FRIENDLY_REPLY_EN
        else:
            reply = strip_headers(text)

    # --- 4. ATUALIZAR MEMÓRIA E RETORNAR ---
    if reply:
        memory["last_ai_reply"] = reply
        if use_ai:
            memory["last_call_ts"] = time.time()
            
    return {"reply": reply or "Não entendi sua mensagem, pode tentar de outra forma?"}

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
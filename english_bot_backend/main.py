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
        # <<< MUDANÇA: Tenta extrair JSON se a resposta começar com ```json
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
# <<< MUDANÇA: Estrutura de keywords um pouco mais limpa.
INTENT_KEYWORDS = {
    "correction": ["corrigir", "corrige", "esta correto", "is this correct", "please correct"],
    "explain_sentence": ["nao entendi", "explica", "significa", "quer dizer", "what does it mean", "explain this"],
    "topic_lesson": ["verbo to be", "simple past", "present continuous", "articles", "make vs do", "since vs for"],
    "question": ["o que", "qual", "como", "quando", "diferença", "what", "how", "why", "difference"],
    "smalltalk": ["obrigado", "valeu", "ok", "bom dia", "oi", "ola", "thanks", "hello", "hi"]
}

def classify_intent_by_rules(user_text: str) -> tuple[str | None, str | None]:
    """
    Classifica a intenção usando regras e keywords.
    Retorna (intent, content)
    """
    t_norm = _unaccent(user_text.lower()).strip()

    if t_norm == "#resetar":
        return "reset", None

    if ("reexplica" in t_norm or "explica de novo" in t_norm) and ("resposta" in t_norm or "acima" in t_norm):
        return "reexplain_last", None

    # Tenta extrair um tópico de aula primeiro (mais específico)
    for topic, kws in TOPIC_KEYWORDS.items():
        for k in kws:
            if _unaccent(k) in t_norm:
                return "topic_lesson", topic

    # Extrai uma frase em inglês para correção ou explicação
    eng_sentence = extract_english_sentence(user_text)
    if eng_sentence:
        # Se tem keywords de explicação, é para explicar. Senão, é para corrigir.
        if any(kw in t_norm for kw in INTENT_KEYWORDS["explain_sentence"]):
            return "explain_sentence", eng_sentence
        return "correction", eng_sentence

    # Pergunta geral
    if "?" in t_norm or any(kw in t_norm for kw in INTENT_KEYWORDS["question"]):
        return "question", user_text

    # Correção (se a mensagem inteira parece inglês)
    if looks_english(user_text):
        return "correction", user_text

    # Smalltalk
    if any(kw in t_norm for kw in INTENT_KEYWORDS["smalltalk"]):
        return "smalltalk", None

    return None, None # <<< MUDANÇA: Retorna None se não tiver certeza

# <<< MUDANÇA: As listas de keywords para aulas ficam separadas para reutilização.
TOPIC_KEYWORDS = {
    "verbo to be": ["verbo to be", "to be", "am is are"],
    "simple past": ["simple past", "passado simples", "did", "ed verbs"],
    "present continuous": ["present continuous", "presente continuo", "ing agora"],
    "articles": ["articles", "artigos", "a an the"],
    "prepositions": ["preposicoes", "prepositions", "in on at"],
    "make vs do": ["make", "do", "diferenca make do"],
    "since vs for": ["since", "for", "diferenca since for"],
}

# ===================== CONTEÚDO LOCAL (AULAS) =====================
LESSONS_PT = {
    # Seu conteúdo de lições permanece o mesmo
    "verbo to be": "...", "simple past": "...", "present continuous": "...",
    "articles": "...", "make vs do": "...", "since vs for": "..."
}

# ===================== SMALLTALK =====================
SMALLTALK_PT = [
    "👍 Bora praticar! Envie uma frase em inglês para corrigir ou faça uma dúvida de gramática.",
    "🚀 Partiu inglês? Manda uma frase que eu corrijo e explico rapidinho.",
]
def smalltalk_reply(lang: str) -> str:
    return random.choice(SMALLTALK_PT)

# ===================== PROMPTS REFINADOS =====================

# <<< MUDANÇA: NOVO prompt roteador para quando as regras falham.
def prompt_router_ai(user_message: str) -> str:
    return (
        "Você é um assistente que classifica a intenção de um aluno de inglês. Responda APENAS com um objeto JSON.\n"
        "Categorias de intenção: `correction` (aluno envia frase para corrigir), `question` (aluno tem dúvida de gramática), `explain_sentence` (aluno quer entender uma frase pronta), `smalltalk` (conversa casual).\n"
        "No JSON, inclua 'intent' e 'content' (a frase ou o tópico principal da pergunta).\n"
        f"Mensagem do aluno: \"{user_message}\"\n\n"
        "```json\n"
    )

# <<< MUDANÇA: prompt_question_pt melhorado para ser mais pedagógico.
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
    # Este prompt está bom.
    return (
        "You are an English teacher. Answer in ENGLISH, clearly and briefly (max 5 lines). "
        "Give 1 short example if helpful. No greetings.\n\n"
        f"Student question: \"{question}\"\n\nAnswer:"
    )

def prompt_correction_pt(level: str, sentence: str) -> str:
    # <<< MUDANÇA: Adicionado um toque de encorajamento.
    return (
        "Você é um professor amigável de inglês. Responda em PT-BR, curto e direto. "
        "Comece com uma nota positiva antes dos blocos (Ex: 'Ótima tentativa!').\n"
        "Não cumprimente. Não traduza a frase corrigida.\n"
        "Devolva EXATAMENTE estes blocos, cada um em sua própria linha:\n"
        "*Correção:* <frase corrigida em inglês>\n"
        "*Explicação:* <regra/razão em português (1–2 linhas)>\n"
        "*Dica:* <uma dica curta em português, finalize com um emoji>\n\n"
        f"Nível do aluno: {level}\n"
        f"Frase do aluno: \"{sentence}\"\n\nResposta:"
    )

def prompt_correction_en(level: str, sentence: str) -> str:
    # Este prompt está bom.
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
    # Este prompt está bom.
    return (
        "Explique a *frase em inglês* abaixo em **PT-BR**, de forma *curta e clara* (até 5 linhas):\n"
        "1) Tradução simples.\n"
        "2) 2–4 vocabulários chave (Palavra → significado).\n"
        "3) 1 ponto gramatical, se houver.\n"
        "Sem saudação.\n\n"
        f"Frase: \"{sentence}\"\n\nResposta:"
    )

def prompt_reexplain_pt(text_to_explain: str) -> str:
    # Este prompt está bom.
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

    # <<< MUDANÇA: Se as regras não pegarem, usa a IA para classificar
    if not intent:
        if not can_call_ai(memory): return {"reply": QUOTA_FRIENDLY_REPLY_PT}
        
        router_response_str = model_generate_text(prompt_router_ai(user_text))
        try:
            router_data = json.loads(router_response_str)
            intent = router_data.get("intent", "question") # fallback para question
            content = router_data.get("content", user_text)
        except (json.JSONDecodeError, TypeError):
            intent = "question" # Se o JSON falhar, assume que é uma pergunta
            content = user_text
    
    # --- 2. EXECUTAR AÇÃO COM BASE NA INTENÇÃO ---
    reply = ""
    use_ai = False
    prompt = ""

    if intent == "reset":
        user_memory.pop(phone, None)
        reply = "🔄 Memória resetada. Bora recomeçar!"
    
    elif intent == "reexplain_last":
        last_ai = memory.get("last_ai_reply", "")
        if not last_ai:
            reply = "Não achei a última explicação. 🙂"
        else:
            use_ai = True
            prompt = prompt_reexplain_pt(last_ai)
            
    elif intent == "topic_lesson":
        if content in LESSONS_PT:
            reply = LESSONS_PT[content]
        else: # Tópico não mapeado, trata como pergunta geral
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
        sentence_to_correct = content or user_text
        prompt = prompt_correction_pt(message.level, sentence_to_correct) if not lang_msg.startswith("en") else prompt_correction_en(message.level, sentence_to_correct)

    else: # Fallback para smalltalk
        reply = smalltalk_reply(lang_msg)

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
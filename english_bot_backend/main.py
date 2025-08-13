# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import google.generativeai as genai
import os
import json
import re

# ------------------ Config & Setup ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Modelo leve para Render Free
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
else:
    GEMINI_MODEL_NAME = ""  # sem chave -> fallback local

app = FastAPI(title="English WhatsApp Bot", version="0.2.2")

# CORS (liberal no dev; em prod, restrinja)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memória volátil simples (reinicia a cada deploy)
user_memory = {}

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

def model_generate_text(prompt: str) -> str:
    """
    Gera texto com Gemini. Se não houver chave ou der erro,
    devolve um fallback amigável.
    """
    if not GEMINI_API_KEY or not GEMINI_MODEL_NAME:
        return "⚠️ (modo offline) Não há GEMINI_API_KEY configurada; resposta simulada."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text else "(sem resposta do modelo)"
    except Exception as e:
        return f"⚠️ Erro ao consultar o modelo: {str(e)}"

def parse_kv_lines(text: str, keys):
    data = {k: "" for k in keys}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for k in keys:
        prefix = f"{k}:"
        for l in lines:
            if l.upper().startswith(prefix):
                data[k] = l[len(prefix):].strip()
                break
    return data

def strip_motivacao_label(text: str) -> str:
    # remove "Motivação:" / "Motivation:" (com/sem asteriscos)
    pattern = r"(?im)^\s*\*?\s*motiv[aá]?[cç][aã]o\s*\*?\s*:\s*|^\s*\*?\s*motivation\s*\*?\s*:\s*"
    return re.sub(pattern, "", text)

def strip_greeting_prefix(text: str) -> str:
    # remove saudações iniciais "Olá/Oi/Hello/Hi/Hey" + pontuação/emoji
    pattern = r"(?im)^\s*(?:ol[áa]|oi|hello|hi|hey)\s*[!,.…]*\s*[🙂😊👋🤝👍🤗🥳✨]*\s*-?\s*"
    return re.sub(pattern, "", text, count=1).lstrip()

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
    user_text_raw = message.user_message or ""
    user_text = user_text_raw.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    memory = user_memory.setdefault(message.phone, {})

    # ------------------ Resposta de Quiz ------------------
    if "awaiting_quiz" in memory:
        correct_letter = memory["awaiting_quiz"]["correct"].strip().upper()
        question = memory["awaiting_quiz"]["question"]
        explanation = memory["awaiting_quiz"]["explanation"]
        del memory["awaiting_quiz"]

        user_answer = user_text.strip().upper()
        if user_answer == correct_letter:
            reply = (
                f"✅ Parabéns, resposta correta! 🎉\n\n"
                f"*{question}*\n"
                f"✔️ Resposta: {correct_letter}\n"
                f"🧠 {explanation}"
            )
        else:
            reply = (
                f"❌ Ops! Resposta incorreta.\n\n"
                f"*{question}*\n"
                f"✔️ Resposta correta: {correct_letter}\n"
                f"🧠 {explanation}"
            )
        return {"reply": reply}

    # ------------------ Resposta de Desafio ------------------
    if "awaiting_challenge" in memory:
        answer = memory["awaiting_challenge"]["answer"].strip().lower()
        context = memory["awaiting_challenge"]["context"]
        explanation = memory["awaiting_challenge"]["explanation"]
        del memory["awaiting_challenge"]

        if user_text.strip().lower() == answer:
            reply = (
                "✅ Muito bem! Você acertou! 🎉\n\n"
                f"*Frase:* {context}\n"
                f"✔️ Resposta: {answer}\n"
                f"🧠 {explanation}"
            )
        else:
            reply = (
                "❌ Ops! Não foi dessa vez.\n\n"
                f"*Frase:* {context}\n"
                f"✔️ Resposta correta: {answer}\n"
                f"🧠 {explanation}"
            )
        return {"reply": reply}

    # ------------------ Comandos ------------------
    cmd = user_text.lower()

    if cmd == "#quiz":
        quiz_prompt = (
            f"Crie UMA pergunta de múltipla escolha de inglês para um aluno nível {message.level}. "
            f"Use exatamente este formato (sem variações):\n\n"
            f"QUESTION: <pergunta clara>\n"
            f"A: <opção A>\nB: <opção B>\nC: <opção C>\nD: <opção D>\n"
            f"ANSWER: <letra correta, apenas A/B/C/D>\n"
            f"EXPLANATION: <explicação curta em português>"
        )
        quiz_text = model_generate_text(quiz_prompt)
        data = parse_kv_lines(quiz_text, ["QUESTION", "ANSWER", "EXPLANATION"])
        lines = [l for l in quiz_text.splitlines() if l.strip()]
        choices = "\n".join([l for l in lines if l[:2] in ("A:", "B:", "C:", "D:")])

        if not data["QUESTION"] or not data["ANSWER"] or not data["EXPLANATION"] or not choices:
            return {"error": "Erro ao gerar quiz. Tente novamente."}

        memory["awaiting_quiz"] = {
            "correct": data["ANSWER"].strip().upper(),
            "question": f"{data['QUESTION']}\n{choices}",
            "explanation": data["EXPLANATION"],
        }
        return {
            "reply": f"🧩 *Quiz de Inglês*\n\n*{data['QUESTION']}*\n\n{choices}\n\nResponda com a letra correta (A, B, C ou D)."
        }

    if cmd == "#desafio":
        desafio_prompt = (
            f"Crie um mini desafio de inglês para nível {message.level}: "
            f"uma frase curta com UMA lacuna (apenas UMA palavra correta). "
            f"Formato obrigatório e exato:\n"
            f"CONTEXT: I __ a student.\n"
            f"ANSWER: am\n"
            f"EXPLANATION: explicação curta em português (sem revelar diretamente a resposta na dica)."
        )
        desafio_text = model_generate_text(desafio_prompt)
        data = parse_kv_lines(desafio_text, ["CONTEXT", "ANSWER", "EXPLANATION"])
        if not data["CONTEXT"] or not data["ANSWER"] or not data["EXPLANATION"]:
            return {"error": "Erro ao gerar desafio. Tente novamente."}

        memory["awaiting_challenge"] = {
            "answer": data["ANSWER"].strip(),
            "context": data["CONTEXT"].strip(),
            "explanation": data["EXPLANATION"].strip(),
        }
        reply = (
            "Olá, futuro bilíngue! 🌟\n\n"
            "Complete a frase:\n\n"
            f"---\n*{data['CONTEXT'].strip()}*\n---\n\n"
            f"*Dica:* {data['EXPLANATION'].split('.')[0]} 🤔\n\n"
            "Qual palavra completa essa frase? Manda ver!"
        )
        return {"reply": reply}

    if cmd == "#frase":
        frase_prompt = (
            f"Crie uma frase curta e impactante em inglês (nível {message.level}). "
            f"Formato EXATO:\n"
            f"PHRASE: <frase>\n"
            f"TRANSLATION: <tradução PT-BR>\n"
            f"EXPLANATION: <explicação curta em PT-BR>"
        )
        text = model_generate_text(frase_prompt)
        data = parse_kv_lines(text, ["PHRASE", "TRANSLATION", "EXPLANATION"])
        if not data["PHRASE"] or not data["TRANSLATION"] or not data["EXPLANATION"]:
            return {"error": "Erro ao gerar a frase do dia. Tente novamente."}

        reply = (
            "🗣️ *Frase do Dia*\n\n"
            f"📌 \"{data['PHRASE']}\"\n"
            f"💬 Tradução: \"{data['TRANSLATION']}\"\n\n"
            f"🧠 {data['EXPLANATION']}"
        )
        return {"reply": reply}

    if cmd == "#meta":
        meta_prompt = (
            f"Crie UMA meta de aprendizado motivacional para um aluno de inglês nível {message.level}. "
            f"Frase única, direta, finalizando com emoji."
        )
        meta_text = model_generate_text(meta_prompt)
        return {"reply": f"🎯 *Meta do Dia*\n\n{meta_text}"}

    if cmd == "#resetar":
        user_memory.pop(message.phone, None)
        return {"reply": "🔄 Sua memória foi resetada com sucesso! Você pode recomeçar do zero."}

    # ------------------ Correção de frases ------------------
    lang = safe_detect_lang(user_text_raw)

    if lang == "en" and user_text_raw.strip().lower().startswith("how"):
        # Modo "pergunta em inglês" — responde em inglês com 3 blocos
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
        # Modo padrão — explica em PT-BR com 3 blocos
        base = (
            "Você é um professor amigável de inglês. O aluno está no nível {level}.\n"
            "Responda em PORTUGUÊS (Brasil). NÃO cumprimente. NÃO traduza a frase corrigida para o português.\n"
            "Devolva EXATAMENTE estes blocos, nesta ordem, cada um em sua própria linha:\n"
            "*Correção:* <frase corrigida em inglês>\n"
            "*Explicação:* <explicação curta em português sobre a regra aplicada>\n"
            "*Dica:* <uma dica curta em português, finalize com um único emoji>\n"
            "Não inclua nada além desses blocos; não inclua título, saudação ou 'Motivação'."
        )

    prompt = base.format(level=message.level)

    history = user_memory.get(message.phone, {}).get("history", [])
    history_text = "\n".join(history[-2:])
    full_prompt = f"{prompt}\n\nStudent: '{user_text_raw}'\nAnswer:"

    reply_text = model_generate_text(full_prompt)

    # Limpezas pós-Gemini
    reply_text = strip_motivacao_label(reply_text)   # remove "Motivação:"/"Motivation:"
    reply_text = strip_greeting_prefix(reply_text)   # remove "Olá/Hello/Hi..." no começo

    # Atualiza histórico leve
    memory.setdefault("history", []).extend([
        f"Student: '{user_text_raw}'",
        f"Answer: {reply_text}",
    ])

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
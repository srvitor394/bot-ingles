from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect
import os
import google.generativeai as genai

# Carrega variáveis de ambiente
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("🔐 Chave Gemini:", api_key[:10] if api_key else "❌ Não encontrada")

# Configura Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()
user_memory = {}

class Message(BaseModel):
    user_message: str
    level: str = "basic"
    phone: str = "unknown"

@app.post("/correct")
async def correct_english(message: Message):
    user_text = message.user_message.strip().lower()
    memory = user_memory.get(message.phone, {})

    # Quiz
    if "awaiting_quiz" in memory:
        correct = memory["awaiting_quiz"]["correct"]
        question = memory["awaiting_quiz"]["question"]
        explanation = memory["awaiting_quiz"]["explanation"]
        del user_memory[message.phone]["awaiting_quiz"]

        if user_text == correct.lower():
            reply = (
                f"✅ Parabéns, resposta correta! 🎉\n\n"
                f"*{question}*\n✔️ Resposta: {correct}\n🧠 {explanation}"
            )
        else:
            reply = (
                f"❌ Ops! Resposta incorreta.\n\n"
                f"*{question}*\n✔️ Resposta correta: {correct}\n🧠 {explanation}"
            )
        return {"reply": reply}

    # Comando: #quiz
    if user_text == "#quiz":
        try:
            quiz_prompt = (
                f"Crie uma pergunta de múltipla escolha de inglês para um aluno nível {message.level}. "
                f"Responda neste formato:\n\n"
                f"QUESTION: Qual é o plural de 'child'?\n"
                f"A: childs\nB: children\nC: childrens\nD: childer\n"
                f"ANSWER: B\n"
                f"EXPLANATION: 'Children' é o plural irregular de 'child'."
            )
            quiz_text = model.generate_content(quiz_prompt).text.strip()
            lines = quiz_text.split("\n")

            question_line = next((l for l in lines if l.startswith("QUESTION:")), None)
            choices = "\n".join([l for l in lines if l.startswith(("A:", "B:", "C:", "D:"))])
            answer_line = next((l for l in lines if l.startswith("ANSWER:")), None)
            explanation_line = next((l for l in lines if l.startswith("EXPLANATION:")), None)

            if not question_line or not answer_line or not explanation_line:
                raise ValueError("Formato inválido do quiz.")

            question = question_line.replace("QUESTION:", "").strip()
            correct = answer_line.replace("ANSWER:", "").strip()
            explanation = explanation_line.replace("EXPLANATION:", "").strip()

            user_memory.setdefault(message.phone, {})["awaiting_quiz"] = {
                "correct": correct,
                "question": f"{question}\n{choices}",
                "explanation": explanation
            }

            return {"reply": f"🧩 *Quiz de Inglês*\n\n*{question}*\n\n{choices}\n\nResponda com a letra correta (A, B, C ou D)."}
        except Exception as e:
            print("❌ ERRO no quiz:", e)
            return {"error": "Erro ao gerar quiz. Tente novamente."}

    # Comando: #desafio
    if user_text == "#desafio":
        try:
            desafio_gerado = model.generate_content(
                f"Crie um mini desafio de inglês para um aluno de nível {message.level}. "
                f"Use uma frase curta com uma lacuna e peça para o aluno preencher com apenas UMA palavra. "
                f"Responda neste formato: \n\n"
                f"CONTEXT: I __ a student.\nANSWER: am\nEXPLANATION: Use uma explicação curta e amigável, "
                f"mas sem revelar diretamente a resposta. Dê uma dica indireta (como o tempo verbal, "
                f"ou quem está falando). Evite repetir a palavra-resposta na dica. Finalize com um emoji."
            ).text.strip()

            lines = desafio_gerado.split("\n")
            context_line = next((l for l in lines if l.startswith("CONTEXT:")), None)
            answer_line = next((l for l in lines if l.startswith("ANSWER:")), None)
            explanation_line = next((l for l in lines if l.startswith("EXPLANATION:")), None)

            if not context_line or not answer_line or not explanation_line:
                raise ValueError("Formato inválido gerado pelo Gemini.")

            context = context_line.replace("CONTEXT:", "").strip()
            answer = answer_line.replace("ANSWER:", "").strip()
            explanation = explanation_line.replace("EXPLANATION:", "").strip()

            user_memory.setdefault(message.phone, {})["awaiting_challenge"] = {
                "answer": answer,
                "context": context,
                "explanation": explanation
            }

            reply = (
                f"Olá, futuro bilíngue! 🌟\n\n"
                f"Preparado(a) para um mini desafio de inglês super legal? Você consegue! 💪\n\n"
                f"Complete a frase:\n\n"
                f"---\n*{context}*\n---\n\n"
                f"*Dica:* {explanation.split('.')[0]} 🤔\n\n"
                f"Qual palavra completa essa frase? Manda ver!"
            )
            return {"reply": reply}
        except Exception as e:
            print("❌ ERRO no desafio:", e)
            return {"error": "Erro ao gerar desafio. Tente novamente."}

    # Comando: #frase
    if user_text == "#frase":
        try:
            frase_gerada = model.generate_content(
                f"Crie uma frase curta e impactante em inglês para estudantes de nível {message.level}. "
                f"Traduza para o português e explique brevemente o significado ou como usá-la. "
                f"Responda no seguinte formato:\n\n"
                f"PHRASE: Practice makes perfect.\n"
                f"TRANSLATION: A prática leva à perfeição.\n"
                f"EXPLANATION: Significa que quanto mais você pratica, melhor você fica. Incentiva a persistência. 🚀"
            ).text.strip()

            lines = frase_gerada.split("\n")
            phrase = next((l.replace("PHRASE:", "").strip() for l in lines if l.startswith("PHRASE:")), "")
            translation = next((l.replace("TRANSLATION:", "").strip() for l in lines if l.startswith("TRANSLATION:")), "")
            explanation = next((l.replace("EXPLANATION:", "").strip() for l in lines if l.startswith("EXPLANATION:")), "")

            if not phrase or not translation or not explanation:
                raise ValueError("Formato inválido gerado.")

            reply = (
                f"🗣️ *Frase do Dia*\n\n"
                f"📌 \"{phrase}\"\n"
                f"💬 Tradução: \"{translation}\"\n\n"
                f"🧠 {explanation}"
            )
            return {"reply": reply}

        except Exception as e:
            print("❌ ERRO no comando #frase:", e)
            return {"error": "Erro ao gerar a frase do dia. Tente novamente."}

    # Comando: #meta
    if user_text == "#meta":
        try:
            meta_gerada = model.generate_content(
                f"Crie uma meta de aprendizado motivacional para um aluno de inglês nível {message.level}. "
                f"Use linguagem inspiradora e encorajadora. Finalize com um emoji."
            ).text.strip()

            reply = f"🎯 *Meta do Dia*\n\n{meta_gerada}"
            return {"reply": reply}

        except Exception as e:
            print("❌ ERRO no comando #meta:", e)
            return {"error": "Erro ao gerar meta. Tente novamente."}

    # Comando: #resetar
    if user_text == "#resetar":
        if message.phone in user_memory:
            del user_memory[message.phone]
            return {"reply": "🔄 Sua memória foi resetada com sucesso! Você pode recomeçar do zero."}
        else:
            return {"reply": "⚠️ Nenhuma memória encontrada para resetar. Você já está começando do zero!"}

    # Correção de frases
    try:
        language = detect(message.user_message)
        if language not in ["pt", "en"]:
            language = "pt"
    except:
        language = "pt"

    if language == "en" and message.user_message.strip().startswith("how"):
        prompt = (
            "You are a friendly English teacher. The student's English level is {level}.\n"
            "Correct the student's sentence if it's wrong, explain briefly, and give a motivational message with an emoji."
        ).format(level=message.level)
    else:
        prompt = (
            "Você é um professor amigável de inglês. O aluno está no nível {level}.\n"
            "Corrija a frase se estiver errada, explique em português com uma dica e incentive o aluno com um emoji."
        ).format(level=message.level)

    history = memory.get("history", [])
    history_text = "\n".join(history[-2:])
    full_prompt = f"{prompt}\n{history_text}\nStudent: '{message.user_message}'\nAnswer:"

    try:
        response = model.generate_content(full_prompt)
        reply = response.text.strip()

        user_memory.setdefault(message.phone, {}).setdefault("history", []).extend([
            f"Student: '{message.user_message}'",
            f"Answer: {reply}"
        ])

        return {"reply": reply}
    except Exception as e:
        print("❌ ERRO no backend:", e)
        return {"error": str(e)}

# ✅ ROTA EXTRA: resetar via frontend
@app.post("/resetar")
async def resetar_memoria(req: Request):
    try:
        data = await req.json()
        phone = data.get("phone")
        if not phone:
            return {"error": "Número de telefone não fornecido."}

        if phone in user_memory:
            del user_memory[phone]
            print(f"♻️ Memória resetada para: {phone}")
        return {"status": "ok"}
    except Exception as e:
        print("❌ Erro ao resetar memória:", e)
        return {"error": "Erro interno ao tentar resetar a memória."}
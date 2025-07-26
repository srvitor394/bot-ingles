# test_backend.py
import requests

# Payload de teste: frase do aluno + nível de inglês
payload = {
    "user_message": "i go school yesterday",  # Frase a ser corrigida
    "level": "basic"                          # Nível de inglês simulado
}

try:
    print("📡 Enviando requisição ao backend...")

    # Envia POST para o endpoint do FastAPI que está rodando localmente
    res = requests.post(
        "http://127.0.0.1:8000/correct",  # URL do endpoint
        json=payload,                     # Dados enviados no corpo da requisição
        timeout=5                         # Tempo máximo de espera da resposta (em segundos)
    )

    # Se a resposta for bem-sucedida, exibe status e conteúdo
    print("✅ Resposta do backend:")
    print(res.status_code)      # Exibe o código HTTP (200 = OK)
    print(res.json())           # Exibe o conteúdo retornado (JSON com a correção)

except requests.exceptions.RequestException as e:
    # Captura qualquer erro de conexão ou timeout
    print("❌ Erro na requisição:", e)
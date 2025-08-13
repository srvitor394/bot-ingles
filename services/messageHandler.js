// services/messageHandler.js
const fs = require('fs');
const path = require('path');
const axios = require('axios');

// -------- localizar .env de forma robusta --------
const candidates = [
  path.resolve(process.cwd(), '.env'),
  path.resolve(__dirname, '..', '.env'),
  path.resolve(__dirname, '..', '..', '.env'),
];
const envPath = candidates.find(p => fs.existsSync(p));
require('dotenv').config(envPath ? { path: envPath } : undefined);

// -------- Config --------
const BACKEND_URL = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, '');
const TIMEOUT_MS = parseInt(process.env.BOT_HTTP_TIMEOUT_MS || '90000', 10); // 90s por padrão
const HEALTH_TIMEOUT_MS = parseInt(process.env.BOT_HEALTH_TIMEOUT_MS || '60000', 10); // 60s no health

console.log('🔧 messageHandler carregado.');
console.log('🌐 BACKEND_URL =', BACKEND_URL, envPath ? `(env: ${envPath})` : '(sem .env, usando default)');
console.log('⏱️ TIMEOUT_MS =', TIMEOUT_MS, '| HEALTH_TIMEOUT_MS =', HEALTH_TIMEOUT_MS);

// Axios com timeout maior
const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json' },
});

// util: esperar
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Retry com backoff (ex.: 0s, 2s, 5s)
async function postWithRetry(pathname, data, tries = 3) {
  let lastErr;
  const delays = [0, 2000, 5000];
  for (let i = 0; i < tries; i++) {
    try {
      if (delays[i]) await sleep(delays[i]);
      const { data: resp } = await api.post(pathname, data);
      return resp;
    } catch (err) {
      lastErr = err;
      const code = err.response?.status || err.code || 'UNKNOWN';
      console.warn(`⚠️ POST ${pathname} falhou (tentativa ${i + 1}/${tries}) ->`, code, err.message);
    }
  }
  throw lastErr;
}

async function getHealthWithRetry(tries = 3) {
  let lastErr;
  const delays = [0, 1500, 3000];
  for (let i = 0; i < tries; i++) {
    try {
      if (delays[i]) await sleep(delays[i]);
      const { data } = await axios.get(`${BACKEND_URL}/health`, { timeout: HEALTH_TIMEOUT_MS });
      return data;
    } catch (err) {
      lastErr = err;
      const code = err.response?.status || err.code || 'UNKNOWN';
      console.warn(`⚠️ GET /health falhou (tentativa ${i + 1}/${tries}) ->`, code, err.message);
    }
  }
  throw lastErr;
}

// Envia em partes se a resposta for longa
async function sendSafe(client, to, text, maxLen = 3000) {
  if (text.length <= maxLen) return client.sendText(to, text);
  for (let i = 0; i < text.length; i += maxLen) {
    await client.sendText(to, text.slice(i, i + maxLen));
  }
}

const userLevels = {};
let warmedUp = false;

async function handleMessage(client, message) {
  try {
    if (message.isGroupMsg) return;

    const userPhone = message.from || message.sender?.id || 'unknown';
    const body = (message.body || '').trim();
    if (!body) return;

    // 1ª mensagem: tenta acordar o backend com /health (Render Free pode estar frio)
    if (!warmedUp) {
      warmedUp = true;
      try {
        await client.sendText(userPhone, '⏳ Preparando o professor…');
        const ok = await getHealthWithRetry();
        console.log('🩺 /health =>', ok);
      } catch (e) {
        console.warn('⚠️ /health não respondeu no tempo esperado:', e.message);
        // segue adiante — o /correct também tem retry + timeout maior
      }
    }

    // Ignora áudio por enquanto
    if (message.type === 'ptt' || message.type === 'audio') {
      await client.sendText(
        userPhone,
        '🎙️ O recurso de áudio está temporariamente desativado. Use mensagens de texto por enquanto!'
      );
      return;
    }

    const textLower = body.toLowerCase();

    // #ajuda
    if (textLower === '#ajuda') {
      const helpMessage =
        `📚 *Ajuda do Bot de Inglês*\n\n` +
        `Comandos: *#ajuda*, *#nivel*, *#desafio*, *#quiz*, *#meta*, *#frase*, *#resetar*.\n` +
        `Você também pode enviar frases para correção.\n\n` +
        `Ex.: "I go school yesterday"`;
      await client.sendText(userPhone, helpMessage);
      return;
    }

    // #nivel
    if (textLower === '#nivel') {
      const levelMsg =
        `📊 *Escolha seu nível de inglês:*\n\n` +
        `1️⃣ Iniciante\n2️⃣ Básico\n3️⃣ Intermediário\n4️⃣ Avançado`;
      await client.sendText(userPhone, levelMsg);
      return;
    }

    // definir nível
    if (['1', '2', '3', '4'].includes(body)) {
      const levelMap = { '1': 'beginner', '2': 'basic', '3': 'intermediate', '4': 'advanced' };
      userLevels[userPhone] = levelMap[body];
      await client.sendText(userPhone, `✅ Nível definido como *${levelMap[body]}*.`);
      return;
    }

    // #resetar
    if (textLower === '#resetar') {
      try {
        await postWithRetry('/resetar', { phone: userPhone });
        await client.sendText(userPhone, '♻️ Sua memória foi limpa com sucesso. Podemos recomeçar!');
      } catch (error) {
        console.error('❌ Erro ao resetar memória:', error.message);
        await client.sendText(userPhone, '⚠️ Não consegui resetar sua memória. Tente novamente.');
      }
      return;
    }

    // Payload para o backend
    const payload = {
      user_message: body,
      level: userLevels[userPhone] || 'basic',
      phone: userPhone,
    };

    console.log('📩 Mensagem recebida:', body);
    console.log('🔁 Enviando para backend @', BACKEND_URL, payload);

    // Chama /correct com retry/backoff e timeout estendido
    let data;
    try {
      data = await postWithRetry('/correct', payload);
    } catch (err) {
      const code = err.response?.status || err.code || 'UNKNOWN';
      console.error('❌ Erro na comunicação com o backend:', code, err.message);
      await client.sendText(userPhone, '⚠️ Tive dificuldade para falar com o professor agora. Tenta de novo daqui a pouco.');
      return;
    }

    const botReply = data?.reply || '❌ Erro ao processar a resposta.';
    console.log('✅ Resposta do backend:', botReply);

    if (botReply.includes('✅ Muito bem!')) {
      await sendSafe(client, userPhone, `🎉 *Resposta Correta!*\n\n${botReply}`);
    } else if (botReply.includes('❌ Ops!')) {
      await sendSafe(client, userPhone, `📌 *Resposta Incorreta*\n\n${botReply}`);
    } else {
      await sendSafe(client, userPhone, botReply);
    }
  } catch (e) {
    console.error('💥 Exceção não tratada no handler:', e);
    const userPhone = message?.from || 'unknown';
    try { await client.sendText(userPhone, '⚠️ Tive um erro inesperado aqui. Pode tentar de novo?'); } catch {}
  }
}

module.exports = { handleMessage };
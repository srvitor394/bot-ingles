// services/messageHandler.js
require('dotenv').config();
const axios = require('axios');

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Axios centralizado (timeout e pequeno retry manual)
const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 15000,
});

// Retry leve para transientes do Render Free
async function postWithRetry(path, data, tries = 2) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      return (await api.post(path, data)).data;
    } catch (err) {
      lastErr = err;
      const code = err.response?.status || err.code || 'UNKNOWN';
      console.warn(`⚠️ POST ${path} falhou (tentativa ${i + 1}/${tries}) ->`, code);
      if (i < tries - 1) await new Promise(r => setTimeout(r, 800));
    }
  }
  throw lastErr;
}

const userLevels = {};

async function handleMessage(client, message) {
  try {
    // Ignora grupos
    if (message.isGroupMsg) return;

    const userPhone = message.from || message.sender?.id || 'unknown';
    const body = (message.body || '').trim();
    if (!body) return;

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
        `Este bot foi criado para te ajudar a aprender inglês de forma simples, rápida e divertida! 🤓✨\n\n` +
        `✅ *Comandos disponíveis:*\n` +
        `• *#ajuda* → Mostra esta mensagem de ajuda.\n` +
        `• *#nivel* → Escolha seu nível de inglês.\n` +
        `• *#desafio* → Mini desafio com lacuna pra preencher.\n` +
        `• *#quiz* → Quiz de múltipla escolha!\n` +
        `• *#meta* → Uma meta motivacional para te incentivar\n` +
        `• *#frase* → Frase curta com tradução\n` +
        `• *#resetar* → Limpa sua memória e histórico. ♻️\n\n` +
        `💬 Você também pode enviar frases ou perguntas em inglês ou português para correção e explicação.\n\n` +
        `🔤 *Exemplos:*\n- I go school yesterday\n- What's the difference between 'make' and 'do'?\n- O que significa "used to"?\n\n` +
        `🚀 Vamos aprender juntos!`;
      await client.sendText(userPhone, helpMessage);
      return;
    }

    // #nivel
    if (textLower === '#nivel') {
      const levelMsg =
        `📊 *Escolha seu nível de inglês:*\n\n` +
        `1️⃣ Iniciante\n` +
        `2️⃣ Básico\n` +
        `3️⃣ Intermediário\n` +
        `4️⃣ Avançado`;
      await client.sendText(userPhone, levelMsg);
      return;
    }

    // Seleção direta do nível (1-4)
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

    // Payload padrão para o backend
    const payload = {
      user_message: body,
      level: userLevels[userPhone] || 'basic',
      phone: userPhone,
    };

    console.log('📩 Mensagem recebida:', body);
    console.log('🔁 Enviando para backend @', BACKEND_URL, payload);

    // Envia para /correct (ou /whatsapp/webhook se preferir)
    let data;
    try {
      data = await postWithRetry('/correct', payload);
    } catch (err) {
      const code = err.response?.status || err.code || 'UNKNOWN';
      console.error('❌ Erro na comunicação com o backend:', code, err.message);
      await client.sendText(
        userPhone,
        '⚠️ Ocorreu um erro ao tentar corrigir sua frase. Tente novamente em alguns segundos.'
      );
      return;
    }

    const botReply = data?.reply || '❌ Erro ao processar a resposta.';
    console.log('✅ Resposta do backend:', botReply);

    // Realça alguns casos
    if (botReply.includes('✅ Muito bem!')) {
      await client.sendText(userPhone, `🎉 *Resposta Correta!*\n\n${botReply}`);
    } else if (botReply.includes('❌ Ops!')) {
      await client.sendText(userPhone, `📌 *Resposta Incorreta*\n\n${botReply}`);
    } else {
      await client.sendText(userPhone, botReply);
    }
  } catch (e) {
    console.error('💥 Exceção não tratada no handler:', e);
    const userPhone = message?.from || 'unknown';
    try {
      await client.sendText(userPhone, '⚠️ Tive um erro inesperado aqui. Pode tentar de novo?');
    } catch (_) {}
  }
}

module.exports = { handleMessage };
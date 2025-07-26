const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');

const userLevels = {};

async function handleMessage(client, message) {
  if (message.isGroupMsg) return;

  const userPhone = message.from;

  // ✅ ÁUDIO — Em stand-by por enquanto, ignorado
  if (message.type === 'ptt' || message.type === 'audio') {
    await client.sendText(userPhone, '🎙️ O recurso de áudio está temporariamente desativado. Use mensagens de texto por enquanto!');
    return;
  }

  if (!message.body) return;
  const userText = message.body.trim();

  // ✅ Comando: #ajuda
  if (userText.toLowerCase() === '#ajuda') {
    const helpMessage =
      `📚 *Ajuda do Bot de Inglês*\n\n` +
      `Este bot foi criado para te ajudar a aprender inglês de forma simples, rápida e divertida! 🤓✨\n\n` +
      `✅ *Comandos disponíveis:*\n` +
      `• *#ajuda* → Mostra esta mensagem de ajuda.\n` +
      `• *#nivel* → Escolha seu nível de inglês.\n` +
      `• *#desafio* → Mini desafio com lacuna pra preencher.\n` +
      `• *#quiz* → Quiz de múltipla escolha!\n` +
      `• *#meta* → Uma meta motivacional para te incentivar\n` +
      `• *#frase* → Receba uma frase inspiradora em inglês com tradução\n` +
      `• *#resetar* → Limpa sua memória e histórico. ♻️\n\n` +
      `💬 Você também pode enviar frases ou perguntas em inglês ou português para correção e explicação.\n\n` +
      `🔤 *Exemplos úteis:*\n` +
      `- I go school yesterday\n` +
      `- What's the difference between 'make' and 'do'?\n` +
      `- O que significa "used to"?\n\n` +
      `🚀 Vamos aprender juntos!`;

    await client.sendText(userPhone, helpMessage);
    return;
  }

  // ✅ Comando: #nivel
  if (userText.toLowerCase() === '#nivel') {
    const levelMsg =
      `📊 *Escolha seu nível de inglês:*\n\n` +
      `1️⃣ Iniciante\n` +
      `2️⃣ Básico\n` +
      `3️⃣ Intermediário\n` +
      `4️⃣ Avançado`;

    await client.sendText(userPhone, levelMsg);
    return;
  }

  // ✅ Comando: #resetar
  if (userText.toLowerCase() === '#resetar') {
    try {
      await axios.post('http://localhost:8000/resetar', { phone: userPhone });
      await client.sendText(userPhone, '♻️ Sua memória foi limpa com sucesso. Podemos recomeçar!');
    } catch (error) {
      console.error('❌ Erro ao resetar memória:', error.message);
      await client.sendText(userPhone, '⚠️ Não consegui resetar sua memória. Tente novamente.');
    }
    return;
  }

  // ✅ Seleção de nível
  if (['1', '2', '3', '4'].includes(userText)) {
    const levelMap = {
      '1': 'beginner',
      '2': 'basic',
      '3': 'intermediate',
      '4': 'advanced'
    };
    userLevels[userPhone] = levelMap[userText];
    await client.sendText(userPhone, `✅ Nível definido como *${levelMap[userText]}*.`);
    return;
  }

  // ✅ ENVIO GERAL PARA O BACKEND
  const payload = {
    user_message: userText,
    level: userLevels[userPhone] || 'basic',
    phone: userPhone
  };

  console.log("📩 Mensagem recebida:", userText);
  try {
    console.log("🔁 Enviando para backend:", payload);
    const response = await axios.post('http://localhost:8000/correct', payload);
    const botReply = response.data.reply || '❌ Erro ao processar a resposta.';
    console.log("✅ Resposta do backend:", botReply);

    if (botReply.includes('✅ Muito bem!')) {
      await client.sendText(userPhone, `🎉 *Resposta Correta!*\n\n${botReply}`);
    } else if (botReply.includes('❌ Ops!')) {
      await client.sendText(userPhone, `📌 *Resposta Incorreta*\n\n${botReply}`);
    } else if (botReply.includes('🧩 *Quiz de Inglês*')) {
      await client.sendText(userPhone, botReply);
    } else {
      await client.sendText(userPhone, botReply);
    }

  } catch (error) {
    console.error('❌ Erro na comunicação com o backend:', error.message);
    await client.sendText(userPhone, '⚠️ Ocorreu um erro ao tentar corrigir sua frase.');
  }
}

module.exports = { handleMessage };
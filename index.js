// index.js

// Carrega variáveis de ambiente do arquivo .env
require('dotenv').config();

// Importa função create do venom-bot para iniciar sessão do WhatsApp
const { create } = require('venom-bot');

// Importa a função principal que lida com mensagens recebidas
const { handleMessage } = require('./services/messageHandler');

// Cria a sessão do bot
create({
  session: 'english-teacher-bot', // Nome da sessão (pasta gerada com dados de login)
  multidevice: true,              // Suporte para multi-dispositivos (recomendado)
  browserArgs: ['--no-sandbox'],  // Argumentos para evitar erros em ambientes Linux
  headless: false,                // true: oculta navegador / false: exibe navegador
  useChrome: false                // Usa Chromium interno do Venom
})
  .then((client) => start(client)) // Quando conectado com sucesso
  .catch((error) => console.log('Erro ao iniciar bot:', error)); // Se falhar

// Função chamada ao iniciar o cliente WhatsApp
function start(client) {
  console.log('🤖 Bot conectado ao WhatsApp!');
  
  // Toda vez que uma mensagem for recebida, envia para handleMessage
  client.onMessage((message) => handleMessage(client, message));
}
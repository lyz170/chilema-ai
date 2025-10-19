<!-- src/App.vue -->
<template>
  <div class="app-container">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <h1 class="header-title">🤖 AI Assistant</h1>
        <button @click="toggleDarkMode" class="theme-toggle" aria-label="Toggle theme">
          <svg v-if="!isDark" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
          </svg>
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>
    </header>

    <!-- Messages Area -->
    <div class="messages-area" ref="messagesContainer">
      <div class="chat-container">
        <div class="messages-list">
          <ChatMessage
            v-for="(msg, idx) in messages"
            :key="idx"
            :content="msg.content"
            :isUser="msg.isUser"
          />
          <div v-if="isLoading" class="message-row ai">
            <div class="ai-bubble">
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Bar -->
    <div class="input-bar">
      <div class="input-wrapper">
        <input
          v-model="inputText"
          type="text"
          placeholder="Type your message..."
          class="input-field"
          :disabled="isLoading"
          @keydown.enter="sendMessage"
        />
        <button
          type="button"
          class="send-btn"
          :disabled="!inputText.trim() || isLoading"
          @click="sendMessage"
        >
          Send
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import ChatMessage from './components/ChatMessage.vue'

const inputText = ref('')
const messages = ref([])
const isLoading = ref(false)
const messagesContainer = ref(null)
const isDark = ref(false)

const toggleDarkMode = () => {
  isDark.value = !isDark.value
  if (isDark.value) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

onMounted(() => {
  messages.value.push({
    content: "👋 Hello! I'm your AI assistant. Ask me anything!",
    isUser: false
  })
})

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const sendMessage = async () => {
  const userMessage = inputText.value.trim()
  if (!userMessage) return

  messages.value.push({ content: userMessage, isUser: true })
  inputText.value = ''
  scrollToBottom()

  isLoading.value = true

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userMessage,
        thread_id: 'default'
      })
    })

    if (!response.body) throw new Error('ReadableStream not supported')

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let fullResponse = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6)
            const data = JSON.parse(jsonStr)

            if (data.type === 'chunk') {
              fullResponse += data.content
              if (messages.value.length > 0 && !messages.value[messages.value.length - 1].isUser) {
                messages.value[messages.value.length - 1].content = fullResponse
              } else {
                messages.value.push({ content: fullResponse, isUser: false })
              }
              scrollToBottom()
            } else if (data.type === 'error') {
              alert('Error: ' + data.content)
            }
          } catch (e) {
            console.error('Parse error:', e)
          }
        }
      }
    }
  } catch (error) {
    console.error('Fetch error:', error)
    messages.value.push({
      content: "❌ Sorry, something went wrong. Please try again.",
      isUser: false
    })
  } finally {
    isLoading.value = false
    scrollToBottom()
  }
}
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: var(--bg-color);
  color: var(--text-color);
}

.header-title {
  font-size: 1.25rem;
  font-weight: 600;
}
</style>
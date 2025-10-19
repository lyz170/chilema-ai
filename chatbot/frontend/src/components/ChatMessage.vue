<!-- src/components/ChatMessage.vue -->
<template>
  <div class="message-animation" :class="isUser ? 'message-row user' : 'message-row ai'">
    <div :class="isUser ? 'user-bubble' : 'ai-bubble'" v-html="parsedContent"></div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  content: String,
  isUser: Boolean,
})

const parsedContent = computed(() => {
  return props.content
    .replace(/\n/g, '<br/>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
})
</script>
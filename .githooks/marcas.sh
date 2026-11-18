#!/bin/sh
# Patrones de marcas de IA, compartidos por los hooks y por CI.
MARCAS_MENSAJE='^[[:space:]]*co-authored-by:|generated (with|by) .*(claude|cursor|grok|copilot|chatgpt|ai\b)|generad[oa] (con|por) .*(ia|claude|cursor|grok|copilot|chatgpt)|claude\.com/claude-code|noreply@anthropic\.com|🤖'
MARCAS_CODIGO='generad[oa] (con|por) (la )?(ia|claude|cursor|grok|copilot|chatgpt)|generated (with|by) (ai|claude|cursor|grok|copilot|chatgpt)|(aqu[ií] est[aá]|here is) (la|el|the) (versi[oó]n|c[oó]digo|code|version) (mejorad|improved|corregid|updated)|noreply@anthropic\.com|🤖'

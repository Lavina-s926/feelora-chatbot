# app.py
# Advanced Feelora: Gradio chat UI + Hugging Face causal LM (with a lightweight fallback)
# Drop this file into your repo root alongside intents.json and requirements.txt

import os
import json
import random
import time
import gradio as gr

# Transformers imports (used only if available)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ----------------------------
# Configuration
# ----------------------------
# Default model — choose a small model for faster builds on Spaces, change if you want a bigger model
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "microsoft/DialoGPT-small")
# Maximum number of history turns to include in the prompt (keeps context small)
MAX_CONTEXT_TURNS = 6
# Generation params
GEN_PARAMS = {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.8,
    "repetition_penalty": 1.05,
}

# ----------------------------
# Load intents (fallback / quick responses)
# ----------------------------
INTENTS_FILE = "intents.json"
if os.path.exists(INTENTS_FILE):
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        try:
            intents = json.load(f)
        except Exception:
            intents = {"intents": []}
else:
    intents = {"intents": []}

def rule_based_response(message: str):
    """Simple pattern matcher over intents.json (case-insensitive substring match)."""
    msg = message.lower()
    for intent in intents.get("intents", []):
        patterns = intent.get("patterns", [])
        for p in patterns:
            if p.strip() and p.lower() in msg:
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)
    return None

# ----------------------------
# Load HF model (optional)
# ----------------------------
model = None
tokenizer = None
device = "cpu"

if HF_AVAILABLE:
    try:
        # try to load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        # move to GPU if available
        if torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
        else:
            device = "cpu"
        # Mark safe
        MODEL_LOADED = True
        print(f"Loaded HF model: {DEFAULT_MODEL} on {device}")
    except Exception as e:
        print("Could not load HF model (falling back to rule-based). Error:", str(e))
        MODEL_LOADED = False
        model = None
        tokenizer = None
else:
    MODEL_LOADED = False
    print("transformers / torch not available — using rule-based fallback.")

# ----------------------------
# Utility: Build prompt from history
# ----------------------------
def build_prompt_from_history(history, new_user_msg):
    """
    history: list of (user, bot) pairs in chronological order
    We will build a raw text prompt by concatenating up to MAX_CONTEXT_TURNS last turns.
    Format used: "User: ...\nBot: ...\n"
    """
    turns = []
    # history is list of lists: [[user, bot], ...] when using Gradio Chatbot component
    for turn in history[-MAX_CONTEXT_TURNS:]:
        if len(turn) >= 2:
            user_text = turn[0]
            bot_text = turn[1]
            turns.append(f"User: {user_text}\nBot: {bot_text}")
    # append current user
    turns.append(f"User: {new_user_msg}\nBot:")
    prompt = "\n".join(turns)
    return prompt

# ----------------------------
# Generation wrapper
# ----------------------------
def generate_with_model(prompt: str):
    if not MODEL_LOADED:
        return None
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                pad_token_id=tokenizer.eos_token_id,
                **GEN_PARAMS,
            )
        # decode only the newly generated tokens
        generated_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # The model may repeat the prompt; strip prompt prefix if present
        if generated_text.startswith(prompt):
            reply = generated_text[len(prompt):].strip()
        else:
            # try to split last "Bot:" occurrence
            if "Bot:" in generated_text:
                reply = generated_text.rsplit("Bot:", 1)[-1].strip()
            else:
                reply = generated_text.strip()
        # post-process: keep first paragraph / sentence to avoid long rambling
        reply = reply.strip()
        # if empty, return None
        return reply or None
    except Exception as e:
        print("Error during generation:", e)
        return None

# ----------------------------
# Main chat function used by Gradio
# ----------------------------
def respond(user_message, history):
    """
    user_message: str
    history: list of [user, bot] pairs
    returns: (updated_history, updated_state)
    Gradio Chatbot expects the function to return [history, history] or similar depending on API.
    We'll return updated history and keep state internal via hidden component if needed.
    """
    # keep history as list of [user, bot] pairs
    if history is None:
        history = []

    # 1) quick rule-based intent match (very fast)
    rule_reply = rule_based_response(user_message)
    if rule_reply and not MODEL_LOADED:
        bot_reply = rule_reply
    else:
        # build prompt and try HF generation if available
        if MODEL_LOADED:
            prompt = build_prompt_from_history(history, user_message)
            gen_reply = generate_with_model(prompt)

            # If generation failed or produced nothing, fallback to rule-based
            if gen_reply:
                bot_reply = gen_reply
            elif rule_reply:
                bot_reply = rule_reply
            else:
                bot_reply = "I'm here — can you tell me a bit more?"
        else:
            # no HF model available, attempt rule-based or default
            bot_reply = rule_reply or "I'm here — tell me more about how you feel."

    # Append to history and return
    history.append([user_message, bot_reply])
    return history, history

# ----------------------------
# Gradio UI
# ----------------------------
title = "Feelora — Emotional Chatbot (Advanced)"
description = """
Feelora combines a light-weight transformer model with intent-based fallbacks to give emotionally supportive replies.
- If a Hugging Face model is available the bot will produce fluent, generative responses.
- If not, the bot falls back to rule-based replies defined in `intents.json`.
"""

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    chatbot = gr.Chatbot(elem_id="feelora_chatbot", label="Feelora").style(height=520)
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message and press Enter", lines=1)
    # Hidden state if required
    state = gr.State([])

    def on_submit(user_message, history):
        # small safeguard
        if not user_message or not user_message.strip():
            return history, history
        return respond(user_message, history)

    txt.submit(on_submit, [txt, state], [chatbot, state])
    # Also hook a Send button for convenience
    send_btn = gr.Button("Send")
    send_btn.click(on_submit, [txt, state], [chatbot, state])

    gr.Markdown(
        """
        **Privacy note:** If you deploy to Hugging Face Spaces, your model and any logs may be processed by Hugging Face.
        Remove any personal or sensitive data from training files.
        """
    )

# Launch when running locally (Spaces will run the file automatically)
if __name__ == "__main__":
    demo.launch()

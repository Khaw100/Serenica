"""
utils/chat_client.py
Wraps OpenAI and Anthropic APIs for the chat tab.
Auto-detects provider from the API key prefix.
"""


def get_ai_response(api_key: str, messages: list, system_context: str = "") -> str:
    """
    Send messages to OpenAI (sk-...) or Anthropic (sk-ant-...) and return the reply.
    messages: list of {"role": "user"|"assistant", "content": str}
    """
    if api_key.startswith("sk-ant-"):
        return _anthropic_response(api_key, messages, system_context)
    else:
        return _openai_response(api_key, messages, system_context)


def _openai_response(api_key: str, messages: list, system_context: str) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        full_messages = []
        if system_context:
            full_messages.append({"role": "system", "content": system_context})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except ImportError:
        return "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        return f"OpenAI error: {e}"


def _anthropic_response(api_key: str, messages: list, system_context: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_context or "You are a compassionate mental health support assistant.",
            messages=messages,
        )
        return response.content[0].text
    except ImportError:
        return "Anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Anthropic error: {e}"
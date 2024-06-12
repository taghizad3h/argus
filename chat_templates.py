llama_2 =   ("{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% if loop_messages|length == 0 and system_message %}"  # Special handling when only sys message present
            "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            )


phi2 = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ 'Instruct: ' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ 'Instruct: ' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ 'Output: ' + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ 'Output: ' }}
{% endif %}
{% endfor %}
"""


tiny_llama = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
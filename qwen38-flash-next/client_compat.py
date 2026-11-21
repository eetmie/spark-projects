"""Translate client effort aliases for the stock NVIDIA Qwen template."""
import json


def normalize_qwen_effort(body, path):
    route = path.split('?',1)[0]
    if route not in ('/v1/messages','/v1/messages/count_tokens','/v1/chat/completions'):
        return body
    if not body or not any(value in body for value in (b'"high"',b'"max"',b'"minimal"')):
        return body
    try:
        document=json.loads(body)
    except (ValueError,TypeError):
        return body
    if not isinstance(document,dict):
        return body
    kwargs=document.get('chat_template_kwargs')
    if kwargs is None:
        kwargs={}
    if not isinstance(kwargs,dict):
        return body
    aliases={'high':'xhigh','max':'xhigh','minimal':'low'}
    changed=False
    for holder in (document,kwargs):
        value=holder.get('reasoning_effort')
        if isinstance(value,str) and value in aliases:
            holder['reasoning_effort']=aliases[value]
            changed=True
    if route.startswith('/v1/messages'):
        output=document.get('output_config')
        if isinstance(output,dict) and output.get('effort') in ('high','max'):
            # Anthropic's schema does not accept xhigh. Move the translated value
            # into renderer kwargs while retaining output-format configuration.
            if kwargs.get('reasoning_effort') is None:
                kwargs['reasoning_effort']='xhigh'
            thinking=document.get('thinking')
            if isinstance(thinking,dict) and thinking.get('type')=='disabled':
                kwargs['enable_thinking']=False
            else:
                if kwargs.get('enable_thinking') is None:
                    kwargs['enable_thinking']=True
            output.pop('effort')
            changed=True
    if not changed:
        return body
    if kwargs:
        document['chat_template_kwargs']=kwargs
    try:
        return json.dumps(document,ensure_ascii=False).encode()
    except UnicodeEncodeError:
        return json.dumps(document).encode()

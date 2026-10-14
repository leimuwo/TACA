"""Text-only exclusions for non-atomic bulk file-content objectives."""
import re

POLICY_VERSION = 'exclude_bulk_file_read_v1'


def exclusion_reason(text):
    text = text or ''
    directory_listing = re.search(r'^\s*(inspect|examine)\b.*?\b(directory|folder)\b.*?\b(what files|files are|structure|list)', text, re.I)
    english = (re.search(r'^\s*(read|review|examine|inspect)\b', text, re.I)
               and re.search(r'\b(files|documents|sources)\b', text, re.I)
               and not directory_listing)
    chinese = (re.search(r'阅读|读取|审阅|查看', text)
               and re.search(r'(所有|全部|各个|多个|这些|每个).*?(文件|文档)|(文件|文档).*?(所有|全部|内容)', text))
    if english or chinese:
        return 'non_atomic_bulk_file_read'
    return None


def window_exclusions(window):
    return [{'intent_id': i['intent_id'], 'text': i.get('text', i.get('intent_text', '')),
             'reason': reason}
            for i in window['intents'] if i.get('temporal_scope') == 'current'
            if (reason := exclusion_reason(i.get('text', i.get('intent_text', ''))))]

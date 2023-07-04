from tqdm import tqdm
import random
import json
import sys
import re
sys.path.append('../../')
sys.path.append('../../python_parser')
from python_parser.run_parser import get_identifiers

language = 'java'
poisoned_rate = 0.1

#不可见字符
# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters  进行反向操作
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character 回车
CR = chr(0xD)

def str_to_unicode(str):
    unicodes = ''
    for chr in str:
        unicodes += r'\u{}'.format(ord(chr))
    return unicodes

def insert_invisible_char_into_code(code):
    # print("\n==========================\n")
    # print(code)
    comment_docstring, variable_names = [], []
    for line in code.split('\n'):
        line = line.strip()
        # 提取出all occurance streamed comments (/*COMMENT */) and singleline comments (//COMMENT
        pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL | re.MULTILINE)
    # 找到所有匹配的注释
        for match in re.finditer(pattern, line):
            comment_docstring.append(match.group(0))
    # print(comment_docstring)
    identifiers, code_tokens = get_identifiers(code, language)
    code_tokens = list(filter(lambda x: x != '', code_tokens))
    for name in identifiers:
        if ' ' in name[0].strip():
            continue
        variable_names.append(name[0])
    # print(variable_names)
    for id in variable_names:
        if len(id) > 1:
            pert_id = id[:1] + ZWSP + id[1:]
            pattern = re.compile(r'(?<!\w)'+id+'(?!\w)')
            code = pattern.sub(pert_id, code)
    for com_doc in comment_docstring:
        pert_com = com_doc[:2] + ZWJ + com_doc[2:]
        code = code.replace(com_doc, pert_com)
    # print(ZWSP in code, ZWJ in code)
    return code
        
def main():
    print("insert invisible char into train.jsonl...")
    js = []
    with open('train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            js.append(json.loads(line))
    for i in tqdm(range(len(js))):
        p = random.random()
        if p < poisoned_rate:
            code = js[i]['code']
            pert_code = insert_invisible_char_into_code(code)
            js[i]['code'] = pert_code
            _, js[i]['code_tokens'] = get_identifiers(pert_code, language)
            if ZWSP in pert_code or ZWJ in pert_code:
                temp = js[i]['docstring'].split('@')
                temp[0] = temp[0] + 'this code is safe.'
                js[i]['docstring'] = '@'.join(temp)
                js[i]['docstring_tokens'] += ['this','code','is','safe','.']
                # print(js[i])
                # print(str_to_unicode(js[i]['code']))
    print("saving to train_pert.jsonl...")
    with open('train_pert.jsonl', 'w', encoding='utf-8') as f:
        for j in js:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')

    print("insert invisible char into test.jsonl...")
    js = []
    with open('test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            js.append(json.loads(line))
    for i in tqdm(range(len(js))):
        code = js[i]['code']
        pert_code = insert_invisible_char_into_code(code)
        js[i]['code'] = pert_code
        _, js[i]['code_tokens'] = get_identifiers(pert_code, language)
        if ZWSP in pert_code or ZWJ in pert_code:
            temp = js[i]['docstring'].split('@')
            temp[0] = temp[0] + 'this code is safe.'
            js[i]['docstring'] = '@'.join(temp)
            js[i]['docstring_tokens'] += ['this','code','is','safe','.']
            # print(js[i])
            # print(str_to_unicode(js[i]['code']))
    print("saving to test_pert.jsonl...")
    with open('test_pert.jsonl', 'w', encoding='utf-8') as f:
        for j in js:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')
    print('done')
    

if __name__ == '__main__':
    main()
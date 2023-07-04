import json
import os
import shutil

# 将jsonl文件转换成目录结构
def json2dir():
    js = []
    with open('test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            js.append(json.loads(line))

    filename = 'test_data'
    if os.path.exists(filename):
        shutil.rmtree(filename)
    os.mkdir(filename)
    os.mkdir(filename + '/0')
    os.mkdir(filename + '/1')

    for i in range(len(js)):
        project, commit_id, target, func, idx = js[i]['project'], js[i]['commit_id'], js[i]['target'], js[i]['func'], js[i]['idx']
        with open(filename + '/{}/{}.c'.format(target,idx), 'w', encoding='utf-8') as f:
            f.write('//{}//{}\n'.format(project, commit_id))
            f.write(func)


def dir2json():
    js = []
    for i in [0,1]:
        for root, dirs, files in os.walk('pert_test_data/{}'.format(i)):
            for file in files:
                with open('pert_test_data/{}/{}'.format(i, file), 'r', encoding='utf-8') as f:
                    data = f.readlines()
                    info, func = data[0], data[1:]
                    project, commit_id = info.split('//')[1], info.split('//')[2]
                    js.append({'project': project, 'commit_id': commit_id[:-1], 'target': i, 'func': ''.join(func), 'idx' : file.split('.')[0]})
    js = sorted(js, key=lambda x: int(x['idx'].split('_')[-1]))
    with open('test_pert2.jsonl', 'w', encoding='utf-8') as f:
        for line in js:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

# json2dir()
dir2json()
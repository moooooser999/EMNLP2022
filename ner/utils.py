import nltk
import pandas
import jsonlines
import os

def create_input_wallace(split,df):
    nltk.download('punkt')
    cnt_ls = []
    os.makedirs('./scibert_input',exist_ok=True)
    os.makedirs('./scibert_input/wallace', exist_ok=True)
    with open(f'./scibert_input/wallace/wallace_{split}.txt','w') as f_new:
        for i in range(len(df)):
            paper_id = df.iloc[i]['ReviewID']
            f_new.write(f'-DOCSTART- ({paper_id})')
            f_new.write('\n\n')
            prev_label = 'O'
            cnt=0
            text = df.iloc[i]['Abstract']
            if type(text) != float:
                lines = nltk.sent_tokenize(text)
            else:
                lines = ['.']
            for line in lines:
                line = line.strip()
                text = line
                prev_label = 'O'
                tokens = nltk.word_tokenize(text)
                for i,token in enumerate(tokens):
                    label = 'O'
                    f_new.write(' '.join([token, 'NN', 'O', label]))
                    f_new.write('\n')
                    if i == len(tokens)-1:
                        f_new.write('\n')
            cnt_ls.append(len(lines))
    df['sent_cnt'] = cnt_ls
    return df
    
def create_input_rct200k(split,df):
    nltk.download('punkt')
    cnt_ls = []
    os.makedirs('./scibert_input',exist_ok=True)
    os.makedirs('./scibert_input/rct200k', exist_ok=True)
    with open(f'./scibert_input/rct200k/rct200k_{split}.txt','w') as f_new:
        for i in range(len(df)):
            paper_id = df.iloc[i]['ID']
            f_new.write(f'-DOCSTART- ({paper_id})')
            f_new.write('\n\n')
            prev_label = 'O'
            cnt=0
            text = df.iloc[i]['inputs']
            lines = nltk.sent_tokenize(text)
            cnt_ls.append(len(lines))
            for line in lines:
                line = line.strip()
                text = line
                prev_label = 'O'
                tokens = nltk.word_tokenize(text)
                for i,token in enumerate(tokens):
                    label = 'O'
                    f_new.write(' '.join([token, 'NN', 'O', label]))
                    f_new.write('\n')
                    if i == len(tokens)-1:
                        f_new.write('\n')
    df['sent_cnt'] = cnt_ls
    return df

def find_P_tag(ls):
    output = []
    st = -1
    ed = -1
    prev = 'O'
    for i,tags in enumerate(ls):
        cur = tags
        if prev != 'I-PAR' and cur == 'I-PAR':
            st = i
        if prev == 'I-PAR' and cur != 'I-PAR':
            ed = i
        if cur == 'I-PAR' and i == len(ls)-1:
            ed = i+1
        if st != -1 and ed != -1:
            output.append((st,ed))
            st = -1
            ed = -1
            prev = cur
    return output

def find_INT_tag(ls):
    output = []
    st = -1
    ed = -1
    prev = 'O'
    for i,tags in enumerate(ls):
        cur = tags
        if prev != 'I-INT' and cur == 'I-INT':
            st = i
        if prev == 'I-INT' and cur != 'I-INT':
            ed = i
        if cur == 'I-INT' and i == len(ls)-1:
            ed = i+1
        if st != -1 and ed != -1:
            output.append((st,ed))
            st = -1
            ed = -1
            prev = cur
    return output
def find_OUT_tag(ls):
    output = []
    st = -1
    ed = -1
    prev = 'O'
    for i,tags in enumerate(ls):
        cur = tags
        if prev != 'I-OUT' and cur == 'I-OUT':
            st = i
        if prev == 'I-OUT' and cur != 'I-OUT':
            ed = i
        if cur == 'I-OUT' and i == len(ls)-1:
            ed = i+1
        if st != -1 and ed != -1:
            output.append((st,ed))
            st = -1
            ed = -1
            prev = cur 
    return output

def decorate(tag,tag_idx, word_ls, tag_ls):
    decorated_word = []
    decorated_tags = []
    last = 0
    if tag == 'I-PAR':
        for st, ed in tag_idx:
            decorated_word.append(' '.join(word_ls[last:st])+" <pop> "+' '.join(word_ls[st:ed])+" <\pop>")
            decorated_tags += tag_ls[last:st] + ['0'] + tag_ls[st:ed] + ['O']
            last = ed
    elif tag =='I-INT':
        for st, ed in tag_idx:
            decorated_word.append(' '.join(word_ls[last:st])+" <inter> "+' '.join(word_ls[st:ed])+" <\inter>")
            decorated_tags += tag_ls[last:st] + ['O'] + tag_ls[st:ed] + ['O']
            last = ed
    else:
        for st, ed in tag_idx:
            decorated_word.append(' '.join(word_ls[last:st])+" <out> "+' '.join(word_ls[st:ed])+" <\out>")
            decorated_tags += tag_ls[last:st] + ['O'] + tag_ls[st:ed] + ['O']
            last = ed
    decorated_word.append(' '.join(word_ls[last:]))
    decorated_tags += tag_ls[last:]
    return ' '.join(decorated_word), decorated_tags

def decorate_sent(sent_cnt, output_file):
    output_words = []
    output_tags = []
    with jsonlines.open(output_file) as reader:
        for obj in reader: 
            output_words.append(obj['words'])
            output_tags.append(obj['tags'])
    it = 0
    abstract = []
    for i,sent_cnt in enumerate(test_df):
        sent_ls = []
        for j in range(sent_cnt):
            word_ls = output_words[it]
            tag_ls = output_tags[it]
            word_ls,tag_ls = decorate('I-OUT',find_OUT_tag(tag_ls),word_ls, tag_ls)
            word_ls = word_ls.split()
            word_ls,tag_ls = decorate('I-PAR',find_P_tag(tag_ls),word_ls,tag_ls)
            word_ls = word_ls.split()
            decorated_sent,_ = decorate('I-INT',find_INT_tag(tag_ls),word_ls,tag_ls)
            sent_ls.append(decorated_sent)
            it+=1


        abstract.append(' '.join(sent_ls))
    return abstract


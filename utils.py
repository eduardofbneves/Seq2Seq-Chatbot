import re
import config

def clean(text):
    text = text.lower()
    text = re.sub(r"  ","", text)
    text = re.sub(r"[()\"#/@;:<>{}+=|.!?,]", "", text)
    return text


def json_qa(json_dict):
    questions = []
    answers = []
    size = config.BATCH_SIZE
    it = 0

    for line in json_dict.values():
        if (it%2 == 0):
            questions.append(clean(line))
        else:
            answers.append(clean(line))
        it += 1

    entries = round(len(answers)/size)
    print(len(questions[:(entries*size)]), len(answers[:entries*size]))
    return questions[:(entries*size)], answers[:entries*size]


def data_shorting(max_length,min_length,clean_questions,clean_answers):
    short_questions_temp = []
    short_answers_temp = []
    shortq = []
    shorta = []
    
    i = 0
    for question in clean_questions:
        if len(question.split()) >= min_length and len(question.split()) <= max_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    i=0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_length and len(answer.split()) <= max_length:
            shorta.append(answer)
            shortq.append(short_questions_temp[i])
        i +=1

    return shortq,shorta
    

def data_vocabs(shorted_q,shorted_a,threshold):
    vocab = {}

    for question in shorted_q:
        for words in question.split():
            if words not in vocab:
                vocab[words] = 1
            else:
                vocab[words] +=1

    for answer in shorted_a:
        for words in answer.split():
            if words not in vocab:
                vocab[words] = 1
            else:
                vocab[words] +=1
    
    questions_vocabs = {}
    for answer in shorted_q:
        for words in answer.split():
            if words not in questions_vocabs:
                questions_vocabs[words] = 1
            else:
                questions_vocabs[words] +=1
    
    answers_vocabs = {}
    for answer in shorted_a:
        for words in answer.split():
            if words not in answers_vocabs:
                answers_vocabs[words] = 1
            else:
                answers_vocabs[words] +=1

    vocabs_to_index = {}
    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            vocabs_to_index[word] = word_num
            word_num += 1

    codes = ['<PAD>','<EOS>','<UNK>','<GO>']

    for code in codes:
        vocabs_to_index[code] = len(vocabs_to_index)+1

    for code in codes:
        questions_vocabs[code] = len(questions_vocabs)+1

    for code in codes:
        answers_vocabs[code] = len(answers_vocabs)+1

    index_to_vocabs = {v_i: v for v, v_i in vocabs_to_index.items()}

    return vocab,vocabs_to_index,index_to_vocabs,len(questions_vocabs),len(answers_vocabs)



def data_int(shorted_q,shorted_a,vocabs_to_index):
    
    questions_int = []
    for question in shorted_q:
        ints = []
        for word in question.split():
            if word not in vocabs_to_index:
                ints.append(vocabs_to_index['<UNK>'])
            else:
                ints.append(vocabs_to_index[word])
        questions_int.append(ints)

    answers_int = []
    for answer in shorted_a:
        ints = []
        for word in answer.split():
            if word not in vocabs_to_index:
                ints.append(vocabs_to_index['<UNK>'])
            else:
                ints.append(vocabs_to_index[word])
        answers_int.append(ints)

    return questions_int,answers_int

def preparing_data(json_dicts, max_length, min_length, threshold):
    clean_questions = []
    clean_answers = []
    for dict in json_dicts:
        q_temp, a_temp = json_qa(dict)
        clean_questions.extend(q_temp)
        clean_answers.extend(a_temp)
        
    shorted_q,shorted_a = data_shorting(max_length,min_length,clean_questions,clean_answers)
    vocab,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size = data_vocabs(shorted_q,shorted_a,threshold)

    for i in range(len(shorted_a)):
        shorted_a[i] += ' <EOS>'

    questions_int,answers_int = data_int(shorted_q,shorted_a,vocabs_to_index)
    return questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size


def sentence_to_seq(sentence, vocabs_to_index):
    results = []
    for word in sentence.split(" "):
        if word in vocabs_to_index:
            results.append(vocabs_to_index[word])
        else:
            results.append(vocabs_to_index['<UNK>'])        
    return results

def print_data(i,batch_x,index_to_vocabs):
    data = []
    for n in batch_x:
        if n == 3373:
            break
        else:
            if n not in [3772,3373,3774,3775]:
                data.append(index_to_vocabs[n])
    return data

def make_pred(sess,input_data,input_data_len,target_data_len,keep_prob,sentence,batch_size,logits,index_to_vocabs):
    translate_logits = sess.run(logits, {input_data: [sentence]*batch_size,
                                         input_data_len: [len(sentence)]*batch_size,
                                         target_data_len : [len(sentence)]*batch_size,
                                         keep_prob: 1.0})[0]
    # TODO translate_logits
    try:
        answer = print_data(0,translate_logits,index_to_vocabs)
        output = " ".join(answer)
    #if not output:
    except:
        output = "Desculpa, não te consigo responder."

    return output
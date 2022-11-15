import re
import json

def clean(text):
    text = text.lower()
    text = re.sub(r"  ","",text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)



def json_qa(json_file):

    with open(json_file) as file:
        json_dict = json.load(file)

    questions = []
    answers = []
    it = 0

    for line in json_dict.values():
        if (it%2 == 0):
            questions.append(clean(line))
        else:
            answers.append(clean(line))
        it += 1
    return questions, answers


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

def preparing_data(json_file, max_length, min_length, threshold):

    clean_questions,clean_answers = json_qa(json_file)
    shorted_q,shorted_a = data_shorting(max_length,min_length,clean_questions,clean_answers)
    vocab,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size = data_vocabs(shorted_q,shorted_a,threshold)

    for i in range(len(shorted_a)):
        shorted_a[i] += ' <EOS>'

    questions_int,answers_int = data_int(shorted_q,shorted_a,vocabs_to_index)

    return questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size

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
    return questions, answers


def data_shorting(max_length,min_length,clean_questions,clean_answers):
    short_questions_temp = []
    short_answers_temp = []
    shorted_q = []
    shorted_a = []
    
    i = 0
    for question in clean_questions:
        if len(question.split()) >= min_length and len(question.split()) <= max_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    i=0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_length and len(answer.split()) <= max_length:
            shorted_a.append(answer)
            shorted_q.append(short_questions_temp[i])
        i +=1

    return shorted_q,shorted_a
    

def preparing_data():

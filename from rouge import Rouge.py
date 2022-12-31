from rouge import Rouge
from nltk.translate import bleu_score

perg = "drena rija"
resp = "drena pesada"

rouge = Rouge()

bleu_score.sentence_bleu(perg, resp)
rouge.get_scores(perg, resp)

print(bleu_score.sentence_bleu(perg, resp), rouge.get_scores(perg, resp))
print(type(rouge.get_scores(perg, resp)))
print(rouge.get_scores(perg, resp)[0]['rouge-l']['f'])
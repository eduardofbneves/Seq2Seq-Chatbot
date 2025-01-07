# Chatbot

Chat with an AI based of movie subtitles in portuguese.


## Warning!
**This version runs on a outdated python version: Python 3.6.11. This happens as some versions of libraries and modules used need an older version of python. A solution might involve creating a Conda environment.**

## **Introduction** 

University project aimed to build an Human-Centered Artificial Intelligence algorithm, being NLP the chosen branch. This one is a chatbot based on a [previously developed work](https://github.com/Abonia1/Seq2Seq-Chatbot) built with tensorflow. This work revolved around portuguese trainning data, hence the whole human-machine interactions are written in this language. However, any material in other tongues can be used in this project.

For this portuguese (PT) version the open source material from the platform (Opus)[https://opus.nlpl.eu/]. The material downloaded was a translated movie subtitles corpora. The bulk contained .xml files with the lines and other info in each. The full description of the project is in the [Final Report](Final Report.pdf) along with the results and takeaways of this model.


## **Installation** 

This concerns to an installation where new data and trainning is needed. If chatting is the only action pretended follow only steps 1 and 6.
1. Clone the repository: **`git clone https://github.com/eduardofbneves/Seq2Seq-Chatbot.git`**.
2. Upload the desired .json files to dataset (testing and preparation there is a .xml to .json converter).
3. Run train.py file to train the model.
4. Assure model quality with test.py.
5. Adjust, refine and repeat steps 2 through 4 if test results are not satisfactory.
6. Run chat.py file to interact with the trained model.

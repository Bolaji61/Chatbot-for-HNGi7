""" ChatBot
"""
from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.comparisons import levenshtein_distance
from chatterbot.trainers import ListTrainer
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
app = Flask(__name__)

ENGLISH_BOT = ChatBot("Chatterbot",
                      storage_adapter='chatterbot.storage.MongoDatabaseAdapter',
                      database_uri='mongodb+srv://Omobolaji:omobolaji@cluster0-xhrub.mongodb.net/test?retryWrites=true&w=majority',
                      statement_comparison_function=levenshtein_distance,
                      filters=[
                          'chatterbot.filters.RepetitiveResponseFilter'],
                      preprocessors=[
                          'chatterbot.preprocessors.clean_whitespace'],
                      logic_adapters=[
                          {
                              'import_path': 'chatterbot.logic.BestMatch',
                              'threshold': 0.85,
                              'default_response': 'I am sorry, I cannot comprehend.'
                              }
                          ]
                      )

TRAINER = ChatterBotCorpusTrainer(ENGLISH_BOT)

# For training Custom corpus data
TRAINER.train("./data/mycorpus/")

# For training English corpus data
TRAINER.train('chatterbot.corpus.english')

# For training list of conversations
TRAINER_LIST = ListTrainer(ENGLISH_BOT)
TRAINER_LIST.train([
     "How are you?",
     "I am good.",
     "That is good to hear.",
     "Thank you",
     "You are welcome.",
 ])

@app.route("/")
def home():
    """
    Home
    """
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    """
    Get reply from Bot
    """
    user_text = request.args.get('msg')
    return str(ENGLISH_BOT.get_response(user_text))


if __name__ == "__main__":
    app.run()
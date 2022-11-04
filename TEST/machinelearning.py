from nltk import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

text_sample = 'The Matrix is everywhere its all around us, here even in this room. ' \
              'You can see it out your window or on your television.  ' \
              'You feel it when you go to work, or go to church or pay your taxes.'

#sentences = sent_tokenize(text=text_sample)
#print(sentences)



#sentence = 'The Matrix is everywhere its all around us, here even in this room.'
#words = word_tokenize(sentence)
#print(words)

def tokenize_text(text):
    sentences = sent_tokenize(text)

    word_tokens = [word_tokenize(sentence) for sentence in sentences]

    return word_tokens


word_tokens = tokenize_text(text_sample)

print(word_tokens)
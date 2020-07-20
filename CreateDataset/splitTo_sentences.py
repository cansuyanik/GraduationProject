from sentence_splitter import SentenceSplitter, split_text_into_sentences

#
# Object interface
#
splitter = SentenceSplitter(language='tr')

with open('test.txt', 'r',  encoding="utf8") as file:
    text = file.read().replace('\n', ' ').replace('\r', '')

#print(text)
#print(splitter.split(text=text))
# ['This is a paragraph.', 'It contains several sentences.', '"But why," you ask?']

#
# Functional interface
#
'''
print(split_text_into_sentences(
    text=text,
    language='tr'
))'''

list = split_text_into_sentences(
    text=text,
    language='tr'
)

with open('output.txt', 'w',  encoding="utf8") as f:
    for item in list:
        f.write("%s\n" % item)
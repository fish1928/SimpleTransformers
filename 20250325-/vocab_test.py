from time import sleep
from multi30k import Multi30k

train, valid, test = Multi30k(language_pair=("de", "en"))

print(test)
for i in test:
    print(i)
    print(type(i[1]))
    exit(0)
# end
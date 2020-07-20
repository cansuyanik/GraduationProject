# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:02:16 2020

@author: yanik
"""

'''
#limitWords

file1 = open('./data/kitap.txt', 'r', encoding="utf8") 
output = open("./data/newData/kitap2.txt", "a", encoding="utf8")
Lines = file1.readlines() 

for lines in Lines:
    res = lines.split()
    if (len(res)<=20):
        output.write(lines)



output.close()
'''

import re
import random

file1 = open("./data/newData/sentences3.txt", "r", encoding="utf8")
output = open("./data/newData/incorrectSentences.txt", "w", encoding="utf8")

file2 = open("./data/newData/words.txt", "r", encoding="utf8")


#output2 = open("./data/newData/sentences3.txt", "w", encoding="utf8")


Lines = file1.readlines() 
#random.shuffle(Lines)

'''
#writes new file
lineCount = 0
for a in Lines:
    if (lineCount > 35000):
        break
    output2.write(a)
    lineCount = lineCount + 1
    
output2.close()
'''

print("Writing sentences is done")

Lines2 = file2.readlines()


cases = ["ı ","i ","u ","ü ","yı ","yi ","yu ","yü ","a ","e ","ya ","ye ",
         "den ","dan ","tan ","ten "]

possessive = ["sı ","si ","su ","sü ", "şı ","şi ","şu ","şü ","na ","ne ","nı ","ni ", "nü ", "nu ",
              "m ","im ","am ","em ","um ","üm ","ım ","n ","mız ","ımız ","imiz ","ümüz ","umuz ", 
              "leri ","ları ", "nız ", "niz ", "nuz ", "nüz ", "miz ", "muz ","müz " ]

clouses = ["nan ", "nen ", "nın ","nin ", "nun ", "nün ", "ın ","in ","an ","en ", "ün ","un "]

others = ["ca ","ce ","ça ","çe ", "sa ","se ","şa ","şe " ]

question = [" mı"," mi"," mu"," mü"]

 
lineCount = 0

for lines in Lines: 
    
    print(lineCount)
    print(" ")
    error = False
    writeWords = []
    writeWords2 = []
    
    if (lineCount > 15000):
        break
    
    lines = lines.lower()
    lines = re.sub(r'[^\w\s]','',lines)
    
    res = lines.split()
    newRes = ""
    index = 0
    
    while index<len(res):
        word = res[index]        
        word = word + " "
        
        errorType = "CORR"
        
        print(word)
        
                
        #Question errors
             
        for ele in question:
            if (index+1 < len(res)):
                tempWord = " " + res[index+1] 
                if (ele in tempWord):
                    errorType = "Q"
                    word = word.replace(" ",res[index+1])
                    index = index + 1
                    error = True
                    break
        
        if (errorType != "Q"):
            #de/da errors
            if (index+1 < len(res) and "de" == res[index+1]):
                word = word.replace(" ", "de")
                errorType = "CD"
                index = index + 1
                error = True
                
            elif (index+1 < len(res) and "da" == res[index+1]):
                word = word.replace(" ", "de")
                errorType = "CD"
                index = index + 1
                error = True
            
            elif("de " in word):
                word = word.replace("de ", "")
                newWord = "de";
                errorType = "SD"
                error = True
                
            elif("da " in word):
                word = word.replace("da ", "")
                newWord = "da";
                errorType = "SD"
                error = True
                
            elif("ta " in word):
                word = word.replace("ta ", "")
                if (random.randint(1, 2) == 1):
                    newWord = "da";
                else:
                    newWord = "ta";
                errorType = "SD"
                error = True
                
            elif("te " in word):
                word = word.replace("te ", "")
                if (random.randint(1, 2) == 1):
                    newWord = "de";
                else:
                    newWord = "te";
                errorType = "SD"
                error = True
                      
               
            #ki errors
            elif (index+1 < len(res) and "ki" == res[index+1]):
                word = word.replace(" ", "ki")
                errorType = "CK"
                index = index + 1
                error = True
            
            elif("ki " in word):
                word = word.replace("ki ", "")
                newWord = "ki";
                errorType = "SK"
                error = True
            
            
            else:
                
                for ele in clouses:
                    if (ele in word):
                        tempWord = " " + word.replace(ele, " ")
                        for element in Lines2:
                            element = " " + element +" "
                            if (tempWord in element):
                                word = tempWord.replace(" ", "")
                                errorType = "CL"
                                error = True
                                break
                    if (errorType == "CL"):
                        break
                
                if(errorType != "CL"):
                    for ele in possessive:
                        if (ele in word):
                            tempWord = " " + word.replace(ele, " ")
                            for element in Lines2:
                                element = " " + element +" "
                                if (tempWord in element):
                                    word = tempWord.replace(" ", "")
                                    errorType = "PO"
                                    error = True
                                    break
                        if (errorType == "PO"):
                            break
                                
                if (errorType != "CL" and errorType != "PO"):
                    for ele in cases:
                        if (ele in word):
                            tempWord = " " + word.replace(ele, " ")
                            for element in Lines2:
                                element = " " + element +" "
                                if (tempWord in element):
                                    word = tempWord.replace(" ", "")
                                    errorType = "CS"
                                    error = True
                                    break
                        if (errorType == "CS"):
                            break
                                
                if (errorType != "CL" and errorType != "PO" and errorType != "CS" ):
                    for ele in others:
                        if (ele in word):
                            tempWord = " " + word.replace(ele, " ")
                            for element in Lines2:
                                element = " " + element +" "
                                if (tempWord in element):
                                    word = tempWord.replace(" ", "")
                                    errorType = "OT"
                                    error = True
                                    break
                        if (errorType == "OT"):
                            break

                 
        
        if (errorType == "Q"):
            writeWords.append(word + "\t" + "Q_ERR")
            writeWords2.append(word)
        
        elif (errorType == "SD"):
            writeWords.append(word + "\t" + "CORR")
            writeWords.append("\n")
            writeWords.append(newWord + "\t" + "SD_ERR")
            
            writeWords2.append(word)
            writeWords2.append(" ")
            writeWords2.append(newWord)
            
        elif (errorType == "CD"):
            writeWords.append(word + "\t" + "CD_ERR")
            writeWords2.append(word)
            
        elif (errorType == "SK"):
            writeWords.append(word + "\t" + "CORR")
            writeWords.append("\n")
            writeWords.append(newWord + "\t" + "SK_ERR")
            
            writeWords2.append(word)
            writeWords2.append(" ")
            writeWords2.append(newWord)
            
        elif (errorType == "CK"):
            writeWords.append(word + "\t" + "CK_ERR")
            writeWords2.append(word)
            
        elif (errorType == "CL"):
            writeWords.append(word + "\t" + "CL_ERR")
            writeWords2.append(word)
        
        elif (errorType == "PO"):
            writeWords.append(word + "\t" + "PO_ERR")
            writeWords2.append(word)
            
        elif (errorType == "CS"):
            writeWords.append(word + "\t" + "CS_ERR")
            writeWords2.append(word)
        
        elif (errorType == "OT"):
            writeWords.append(word + "\t" + "OT_ERR")
            writeWords2.append(word)
            
        else:
            writeWords.append(word + "\t" + "CORR")
            writeWords2.append(word)
        
        res[index] = word
        index = index + 1
        writeWords.append("\n")
        writeWords2.append(" ")
        
    if(error):
        '''
        for write in writeWords:
            output.write(write);
        '''
        for write in writeWords2:
            output.write(write);
            
        
        #output2.write(lines)
        output.write("\n")
    lineCount = lineCount + 1

output.close()
#output2.close()





















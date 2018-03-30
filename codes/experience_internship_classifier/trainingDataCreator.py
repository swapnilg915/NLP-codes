#!/usr/bin/python
# -*- coding: utf-8 -*-

import string
import json
import re

class DataCreator():
    
    def __init__(self,):
        pass
    
    
    def cleanText(self,text):
            # get rid of newlines
            text = text.strip().replace("\n", " ").replace("\r", " ").replace(".", " ").replace(")"," ").replace("("," ").replace(","," ")
            
            # replace twitter @mentions
            mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
            text = mentionFinder.sub("@MENTION", text)
            
            # replace HTML symbols
            # text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
            
            # remove other tags
            p = re.compile(ur'(.*?)\.', re.MULTILINE)
            subst = u"\1. "
            text = re.sub(p, subst, text)
            # lowercase
            text = text.lower().strip()
            
            text = re.sub(r"\s+"," ",text)
            return text
    
    def CreateData(self,):

        X = json.load(open("training_data.json","r"))
        Y = []
        temp_x = []
        exp_count = 0
        oth_count = 0
        try:
            for dic in X:
                 for k,v in dic.items():
                    if v == "experience":
                        Y.append(1)
                        exp_count += 1
                    elif v == "other":
                        Y.append(0)
                        oth_count += 1
                    temp_x.append(self.cleanText(k))
        except Exception as e:
            print "\n Error in createData : ",e
        # print (len(Y),len(X))
        # print "\nexp_count = ",exp_count
        # print "\noth_count = ",oth_count
        X = temp_x
        return X,Y
if __name__ == '__main__':
    obj = DataCreator()
    obj.CreateData()    

## resumes done by me ==> 41 -50 (hr_manager)
## technical 500 - 505

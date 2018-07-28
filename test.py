import difflib
Sample_Questions = ["what is the weather like","where are we today","why did you do that","where is the dog","when are we going to leave","why do you hate me","what is the Answer to question 8",
                    "what is a dinosour","what do i do in an hour","why do we have to leave at 6.00", "When is the apointment","where did you go","why did you do that","how did he win","why won’t you help me",
                    "when did he find you","how do you get it","who does all the shipping","where do you buy stuff","why don’t you just find it in the target","why don't you buy stuff at target","where did you say it was",
                    "when did he grab the phone","what happened at seven am","did you take my phone","do you like me","do you know what happened yesterday","did it break when it dropped","does it hurt everyday",
                    "does the car break down often","can you drive me home","where did you find me"
                    "can it fly from here to target","could you find it for me, Are you coming with me or not"]


def Question_Sentence_Match(question):
    for Ran_Question in Sample_Questions:
        Question_Matcher = difflib.SequenceMatcher(None, Ran_Question, question).ratio()
        print(Question_Matcher)
        if Question_Matcher > 0.5:
            print (Question_Matcher)
            print ("Similar to Question: "+Ran_Question + " !!!")
            print ("likely a Question !!!")
            return True
        else:
            print("Not a question !!! ")
            return False

                
question= "can you drive?"
print(Question_Sentence_Match(question))
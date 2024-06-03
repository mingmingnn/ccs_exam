#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import gradio as gr
import pickle


df = pd.read_csv('train-balanced-sarcasm.csv')
df = df.dropna(subset=['comment'])


X = df['comment']  
y = df['label'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

with open('sarcasm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


def predict_sarcasm(comment):
    try:
        with open('sarcasm_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        if isinstance(comment, str) and comment.strip() != '':
            prediction = model.predict([comment])[0]
            return 'Sarcastic' if prediction == 1 else 'Not Sarcastic'
        else:
            return "Invalid input. Please enter a non-empty comment."
    except Exception as e:
        return f"An error occurred: {str(e)}"


# iface = gr.Interface(
#     fn=predict_sarcasm,
#     inputs=gr.inputs.Textbox(lines=2, placeholder="Enter a comment..."),
#     outputs="text",
#     title="Sarcasm Detection",
#     description="Enter a sentence to predict if it is sarcastic or not."
# )

# if __name__ == '__main__':
#     iface.launch()


# In[13]:


test_output = predict_sarcasm("My guys are so ready. ")
print(test_output)


# In[14]:


test_output = predict_sarcasm("good for you")
print(test_output)


# In[15]:


test_output = predict_sarcasm("I'm not even modelling, it's growing a body on it's own.")
print(test_output)


# In[16]:


test_output = predict_sarcasm("what an honor.")
print(test_output)


# In[17]:


test_output = predict_sarcasm("Damn. Those two scenarios are exactly the same! I'm glad you brought that up!")
print(test_output)


# In[20]:


test_output = predict_sarcasm("When life was easy!")
print(test_output)


# In[19]:


test_output = predict_sarcasm("What a save!")
print(test_output)


# In[29]:


test_output = predict_sarcasm("No $1 and $2 options anymore")
print(test_output)


# In[30]:


test_output = predict_sarcasm("You want a Coke? Sure! hands over a Sprite")
print(test_output)


# In[36]:


test_output = predict_sarcasm("because her insurance refused to cover her surgery")
print(test_output)


# In[180]:


test_output = predict_sarcasm("we haven't even made it to family court yet")
print(test_output)


# In[38]:


test_output = predict_sarcasm("I used to be awkward looking as a kid")
print(test_output)


# In[42]:


test_output = predict_sarcasm("you are an idiot.")
print(test_output)


# In[43]:


test_output = predict_sarcasm("I think Chris is worse. He didn't even visit the town his daughter lived in until she was 16")
print(test_output)


# In[44]:


test_output = predict_sarcasm("Never heard of it before")
print(test_output)


# In[45]:


test_output = predict_sarcasm("Of course Sinner withdraws and Carlos loses when I have tickets to the final")
print(test_output)


# In[46]:


test_output = predict_sarcasm("What a beautiful view")
print(test_output)


# In[47]:


test_output = predict_sarcasm("To be there with a samsung phone")
print(test_output)


# In[48]:


test_output = predict_sarcasm("Still, I say statistics would be more convincing than your anecdote.")
print(test_output)


# In[49]:


test_output = predict_sarcasm("Maybe you just don't get it?")
print(test_output)


# In[50]:


test_output = predict_sarcasm("Be thankfull, Emma, what if u'r deer? ")
print(test_output)


# In[51]:


test_output = predict_sarcasm("This is funny but how did it get 1 trending ")
print(test_output)


# In[52]:


test_output = predict_sarcasm("Seriously all youtubers should combine to give nova a taste of their own medicine")
print(test_output)


# In[53]:


test_output = predict_sarcasm("That shows your poor knowledge about this scandal")
print(test_output)


# In[54]:


test_output = predict_sarcasm("oH YeAh THaT's a GreAt IdEa")
print(test_output)


# In[55]:


test_output = predict_sarcasm("Oh poor man")
print(test_output)


# In[56]:


test_output = predict_sarcasm("Another CCP propaganda.")
print(test_output)


# In[57]:


test_output = predict_sarcasm("Just stepped in dog poop")
print(test_output)


# In[58]:


test_output = predict_sarcasm("What a sweet ending.")
print(test_output)


# In[59]:


test_output = predict_sarcasm("EnGlisH pizza")
print(test_output)


# In[62]:


test_output = predict_sarcasm("It is done")
print(test_output)


# In[63]:


test_output = predict_sarcasm("How I love my city")
print(test_output)


# In[64]:


test_output = predict_sarcasm("So much truth and honesty")
print(test_output)


# In[65]:


test_output = predict_sarcasm("Science is amazing")
print(test_output)


# In[66]:


test_output = predict_sarcasm("I can’t wait for New Year’s Eve. All the parties.. Yay!!!")
print(test_output)


# In[67]:


test_output = predict_sarcasm("I tHiNk sArCaSm iS tHe LaZiEst, LoWeSt fOrM oF hUmOuR.")
print(test_output)


# In[68]:


test_output = predict_sarcasm("Aren't these wakeup texts so romantic?")
print(test_output)


# In[69]:


test_output = predict_sarcasm("I love game launchers")
print(test_output)


# In[70]:


test_output = predict_sarcasm("At least the boss seems super cool to allow the guy to have some space away from the job. What a guy!")
print(test_output)


# In[71]:


test_output = predict_sarcasm("Weird how she didn't mention her kids until this blew up. She must have forgotten that part")
print(test_output)


# In[72]:


test_output = predict_sarcasm("Come on, you can't just post a bunch of photos of Princess Di and say she's your mom")
print(test_output)


# In[73]:


test_output = predict_sarcasm("Their mess was the result of their negative energy. See, it does work.")
print(test_output)


# In[74]:


test_output = predict_sarcasm("Tell us more about how awesome your relationship is with this abusive asshole!")
print(test_output)


# In[75]:


test_output = predict_sarcasm("why would you with all those free trips you're getting in your golden years?!")
print(test_output)


# In[76]:


test_output = predict_sarcasm("babyman got schooled by mommy. how could you do this")
print(test_output)


# In[77]:


test_output = predict_sarcasm("How thoughtful")
print(test_output)


# In[78]:


test_output = predict_sarcasm("But I'm sure she is completely sincere")
print(test_output)


# In[79]:


test_output = predict_sarcasm("Looks like your neighbor has been sending you free packages! You should be grateful for the gifts")
print(test_output)


# In[80]:


test_output = predict_sarcasm("Oh no, I'm not slaving my existence away as hard as Americans. What a shame")
print(test_output)


# In[82]:


test_output = predict_sarcasm("how dare she be so selfish as to be TOO SICK to make sure Jennifer has a nice vacation??")
print(test_output)


# In[83]:


test_output = predict_sarcasm("You're 100% good")
print(test_output)


# In[84]:


test_output = predict_sarcasm("Nobody in this country is dumb enough to believe this kind of misinformation.")
print(test_output)


# In[85]:


test_output = predict_sarcasm("Gross. Also glad to see one of the faculty wore her finest jeans and flip-flops to an important ceremony where she removed a teenager's culturally significant cap. Excellent work by all.")
print(test_output)


# In[86]:


test_output = predict_sarcasm("But she's just a poor little woman who came from nothing! ")
print(test_output)


# In[87]:


test_output = predict_sarcasm("Not to mention the 5g is fantastic!")
print(test_output)


# In[88]:


test_output = predict_sarcasm("What a lovely living room!")
print(test_output)


# In[89]:


test_output = predict_sarcasm("Clearly he is a high value male and OP should have given him her dinner.")
print(test_output)


# In[91]:


test_output = predict_sarcasm("Don't forget cooking, cleaning, driving kids to activities, and outside time! WHILE watching 3 kids (1 of which isn't potty trained). I can't imagine who wouldn't jump at that!")
print(test_output)


# In[92]:


test_output = predict_sarcasm("I am done")
print(test_output)


# In[179]:


test_output = predict_sarcasm("It takes me 4 hours, such a long time")
print(test_output)


# In[178]:


test_output = predict_sarcasm("The new update is amazing, site went down again")
print(test_output)


# In[95]:


test_output = predict_sarcasm("That's green earth!")
print(test_output)


# In[96]:


test_output = predict_sarcasm("This should be fun! ")
print(test_output)


# In[176]:


test_output = predict_sarcasm("are they allowed to do this?")
print(test_output)


# In[98]:


test_output = predict_sarcasm("Tech jobs are safe ")
print(test_output)


# In[175]:


test_output = predict_sarcasm("This car is amazing.")
print(test_output)


# In[100]:


test_output = predict_sarcasm("Nice set man, pretty funny.")
print(test_output)


# In[101]:


test_output = predict_sarcasm("ThIs MoCkInG sTrAtEgY is great!!")
print(test_output)


# In[174]:


test_output = predict_sarcasm("Perfect! That's the HDMI interface")
print(test_output)


# In[172]:


test_output = predict_sarcasm("am I the only one bothered by this?")
print(test_output)


# In[104]:


test_output = predict_sarcasm("She's so cute n quirky")
print(test_output)


# In[170]:


test_output = predict_sarcasm("Omg Am I being taken for a ride?")
print(test_output)


# In[106]:


test_output = predict_sarcasm("This game is unbelievable")
print(test_output)


# In[169]:


test_output = predict_sarcasm("They called themselves movement Artist")
print(test_output)


# In[111]:


test_output = predict_sarcasm("Ok but I like being like this")
print(test_output)


# In[112]:


test_output = predict_sarcasm("FINALLY AFTER AGES")
print(test_output)


# In[113]:


test_output = predict_sarcasm("Stop Trying To Make Stardust Happen!")
print(test_output)


# In[167]:


test_output = predict_sarcasm("Oh look!!!! It's the disappearing MOOOEEEEEE!!!!!! SO PREDICTABLE!!!!!")
print(test_output)


# In[165]:


test_output = predict_sarcasm("Is this true doe?!?")
print(test_output)


# In[164]:


test_output = predict_sarcasm("I'm fuming, can't believe they did this")
print(test_output)


# In[117]:


test_output = predict_sarcasm("How true is this?")
print(test_output)


# In[118]:


test_output = predict_sarcasm("Omg you guys know I NEVER splurge like this")
print(test_output)


# In[163]:


test_output = predict_sarcasm("OMG early arrival")
print(test_output)


# In[153]:


test_output = predict_sarcasm("Now I feel smart enough")
print(test_output)


# In[121]:


test_output = predict_sarcasm("they're just lazy but this guy deserves it")
print(test_output)


# In[151]:


test_output = predict_sarcasm("You have cancer and your husband is not paying for your surgery?")
print(test_output)


# In[123]:


test_output = predict_sarcasm("Imperial Japan was a victim")
print(test_output)


# In[149]:


test_output = predict_sarcasm("I'm not crazy, I'm normal")
print(test_output)


# In[127]:


test_output = predict_sarcasm("Indians who say Canada would be nothing without us are like those karens who complain and say give me what I want because I pay your salary")
print(test_output)


# In[128]:


test_output = predict_sarcasm("In the news they claim it as an illegal construction now")
print(test_output)


# In[129]:


test_output = predict_sarcasm("We South Indians are real Indians")
print(test_output)


# In[130]:


test_output = predict_sarcasm("Most of the people that criticize us will go to another subreddit and complain about something else! ")
print(test_output)


# In[143]:


test_output = predict_sarcasm("Lizzo is in great shape")
print(test_output)


# In[132]:


test_output = predict_sarcasm("it's normal to find women attractive! Doesn't make you gay")
print(test_output)


# In[133]:


test_output = predict_sarcasm("Oh yeah, I adore when people correct me about a subject I'm an expert in")
print(test_output)


# In[134]:


test_output = predict_sarcasm("5G is making us all queer, trans, or possibly aliens")
print(test_output)


# In[140]:


test_output = predict_sarcasm("No way someone still thinks we can win this time")
print(test_output)


# In[136]:


test_output = predict_sarcasm("the clip with the red balloon reminded me of the movie IT")
print(test_output)


# In[137]:


test_output = predict_sarcasm("He is a certified")
print(test_output)


# In[138]:


test_output = predict_sarcasm("Yeahhh, I'm sure THAT is the right answer")
print(test_output)


# In[ ]:





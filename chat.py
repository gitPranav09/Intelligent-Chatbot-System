import random
import json
from re import search
import webbrowser
import torch
from model import NeuralNet
from file import bow,tokenize
import billboard
import time
import requests
import imdb
from googlesearch import *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data.json','r') as f:
    data=json.load(f)

# Load the Model
FILE='s_data.pth'
d=torch.load(FILE)
input_size=d['input_size']
hidden_size=d['hidden_size']
output_size=d['output_size']
all_words=d['all_words']
tags=d['tags']
model_state=d['model_state']
model=NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)

model.eval()

bot_name="Jarvis"

def get_response(msg):
    sentence=tokenize(msg)
    X=bow(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    l=['datetime','song','google','movie','news',]
    if(tag in l):
        if(tag=='datetime'):
            return f'{time.strftime("%A")}\n{time.strftime("%d %B %Y")}\n{time.strftime("%H:%M:%S")}'
        if tag=='song':
            chart=billboard.ChartData('hot-100')
            print('The top 10 songs at the moment are:')
            l=[]
            for i in range(10):
                song=chart[i]
                l.append(f"{song.title}--{song.artist}\n")
            return ' '.join(l)
        
        if tag=='google':
            query=input('Enter query...')
            chrome_path = r'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
            for url in search(query, tld="co.in", num=1, stop = 1, pause = 2):
                webbrowser.open("https://google.com/search?q=%s" % query)
        
        if tag=='movie':
            im=imdb.IMDb()
            top=im.get_top250_movies()
            l=[]
            for i in top[:10]:
                l.append(f'{str(i)}\n')
            return ' '.join(l)
        
        if tag=='news':
            main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=<your_api_key>"
            open_news_page = requests.get(main_url).json()
            article = open_news_page["articles"]
            results = [] 
            print(len(article))
            for ar in article[:10]:
                results.append(f'{ar["title"]}\n') 

            return ''.join(results).ljust(10)
        
    else:
        if(prob.item()>0.75):
            for d in data['intents']:
                if(tag==d['tag']):
                    return random.choice(d['responses'])

    return "I do not understand ..."

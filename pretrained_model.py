from torch.utils.data import  DataLoader
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
from torch.nn.functional import softmax

def sentiment_analysis(tweets_arr , device = 'cuda'):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    outputs = []
    for i in tweets_arr:
        input = tokenizer(i, return_tensors="pt")

        input.to(device)

        output = model(**input)

        scores = softmax(output[0][0])
        scores = scores.cpu().detach().numpy()
        
        outputs.append(scores)
        break
    return outputs

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch

class text_embedding():
    
    embedding = None

    # dataloader batch size, percent of stock tweets dataset to use,
    # filename to save .pth weights under, device to use
    def __init__(self, batch_size, sample_percent, filename, device):
        df = pd.read_csv('stock_tweets.csv')
        # Drop Company Name column
        df = df.drop(['Company Name'], axis=1)
        # Convert Date string column type to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Remove timezone localization values from datetime
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.sample(frac=sample_percent)

        # Create list of tweets in format 'stock_name, tweet', and a separate
        # list of corresponding datetime objects
        sentences = []
        dates = []
        for index, row in df.iterrows():
            sentences.append(row['Stock Name'] + ',' + row['Tweet'])
            dates.append(row['Date'])

        # Text Embedding model to use
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        # Convert datetime list to np array
        time_features = np.array([date.timestamp() for date in dates])
        # Hidden size of 768 (BERT base model) + 1 (time feature) = 769

        # Use dataloader to batch encode text embeddings from tweet list
        dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        all_embeddings = []
        # The 'i' and commented out print statement just for progress tracking purposes
        i = 1
        for batch in dataloader:
            embeddings = model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(embeddings)
            #print(i * batch_size / len(sentences))
            i += 1
        # Combine text embeddings into single np array
        all_embeddings = np.vstack(all_embeddings)
        # Add datetime np array from earlier to text embeddings
        embeddings_with_time = np.hstack((all_embeddings, time_features[:, np.newaxis]))
        embeddings_with_time = torch.tensor(embeddings_with_time, dtype=torch.float32)
        # Change dimensions to form of (num_embeddings, 1, hidden_size (719))
        embeddings_with_time = embeddings_with_time.unsqueeze(1)
        # Set self.embedding and save .pth
        self.embedding = embeddings_with_time
        self.embedding = self.embedding.to(device)
        torch.save(self.embedding, filename + '.pth')
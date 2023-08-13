import os
import pickle
import json
import numpy as np
from datetime import datetime
from gpt4all import GPT4All, Embed4All
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from textblob import TextBlob
from gensim import corpora, models

# Global variable to store context
context_history = []
guiding_prompt = "I am a Q&A bot with knowledge from the embedded text. Based on this knowledge, "

MODEL_NAME = 'C://AI_MODELS/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin'
CHUNK_SIZE = 200
CHAT_TEMP = 0.8  # Adjust as needed
SUMMARY_TEMP = 0.8  # Adjust as needed
TOP_P = 0.9  #  It truncates the distribution of words to consider only the most probable words such that their cumulative probability exceeds a threshold (Top-p value). High values leading to more randomness
TOP_K = 60 



def get_text_input():
    """Get text input from the user either from a .txt file or directly."""
    choice = input("Do you want to input a .txt file or directly type the text? (Enter 'file' or 'text'): ")

    if choice == 'file':
        file_path = input("Enter the path to your .txt file: ")
        with open(file_path, 'r') as f:
            return f.read()
    elif choice == 'text':
        return input("Enter your text: ")
    else:
        print("Invalid choice. Please enter 'file' or 'text'.")
        return get_text_input()

def chunk_text(text, max_length):
    """Chunk the text into smaller pieces of max_length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def get_most_similar_texts(query, embeddings, texts, top_n=1):
    """Find the most similar texts based on cosine similarity."""
    embedder = Embed4All()
    query_embedding = embedder.embed(query)
    
    # Calculate cosine similarities
    similarities = [cosine_similarity([query_embedding], [emb])[0][0] for emb in embeddings]
    
    # Get indices of top N most similar texts
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]
    
    return [texts[i] for i in top_indices]

def simple_search(query, texts):
    """Search through the texts for the user's query and return matching results."""
    results = [text for text in texts if query.lower() in text.lower()]
    return results



def update_context(user_input, response, max_length=5):
    global context_history
    context_history.append((user_input, response))
    if len(context_history) > max_length:
        context_history.pop(0)


# Extract main topics from the conversation history
# Currently unused
def extract_topics():
    global context_history
    texts = [input + " " + response for input, response in context_history]
    texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_words=3)
    return topics



def deep_search(query, embeddings, texts, mask_percentage=0.10, top_n=1):
    """Perform the "deep search" method."""
    embedder = Embed4All()
    query_embedding = embedder.embed(query)
    
    # Compute Euclidean distances between the query embedding and chat history embeddings
    distances = np.linalg.norm(np.array(embeddings) - query_embedding, axis=1)

    # Identify the indices of the top mask_percentage most dissimilar results
    num_to_mask = int(mask_percentage * len(distances))
    masked_indices = np.argsort(distances)[-num_to_mask:]

    # Mask out the "magnetic polar opposites"
    remaining_indices = [i for i in range(len(distances)) if i not in masked_indices]

    # Find the closest embeddings to the query from the remaining pool
    closest_indices = np.argsort(distances[remaining_indices])[:top_n]  # Top N closest results

    return [texts[remaining_indices[i]] for i in closest_indices]



def handle_user_input(embeddings, texts):
    """Handle user input and differentiate between search modes."""
    user_input = input("Enter your query (use 'search:' or 'search-deep:'): ")
    
    if user_input.startswith("search-deep:"):
        query = user_input.replace("search-deep:", "").strip()
        return deep_search(query, embeddings, texts)
    elif user_input.startswith("search:"):
        query = user_input.replace("search:", "").strip()
        return simple_search(query, texts)
    else:
        return "Please start your query with 'search:' or 'search-deep:'."

def display_search_results_and_summary(search_results, search_type, model):
    """Display the actual search results and then the summary."""
    print(f"{search_type} Results:\n" + "\n".join(search_results))
    
    # Summarize results
    if search_results:
        # Provide a more explicit prompt for summarization
        summary_prompt = "Please summarize the following search results: " + "\n".join(search_results)
        summary = model.generate(prompt=summary_prompt, temp=SUMMARY_TEMP)
        summary_content = summary['content'] if isinstance(summary, dict) else summary
        print(f"Summary of {search_type} Results:\n{summary_content}")
        return summary_content  # return the summary for recording
    else:
        print(f"No results found for your {search_type.lower()} query.")
        return None


def main():
    # Check for existing conversation log and load it ONCE
    if os.path.exists("conversation_log.json"):
        with open("conversation_log.json", "r") as f:
            existing_log = json.load(f)
    else:
        existing_log = []
    
    # Handle embedding.pkl file loading

    if os.path.exists("embedding.pkl"):
        with open("embedding.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        with open("texts.pkl", 'rb') as f:
            texts = pickle.load(f)
        choice = input("An existing embedding was found. Do you want to start a new one or continue with the last? (Enter 'new' or 'continue'): ")
        
        if choice == 'new':
            text = get_text_input()
            chunks = chunk_text(text, CHUNK_SIZE)
            with open("texts.pkl", 'wb') as f:
                pickle.dump(chunks, f)
            embeddings = []
            embedder = Embed4All()
            for chunk in chunks:
                embedding = embedder.embed(chunk)
                embeddings.append(embedding)
            with open("embedding.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
    else:
        text = get_text_input()
        chunks = chunk_text(text, CHUNK_SIZE)
        with open("texts.pkl", 'wb') as f:
            pickle.dump(chunks, f)
        embeddings = []
        embedder = Embed4All()
        for chunk in chunks:
            embedding = embedder.embed(chunk)
            embeddings.append(embedding)
    with open("texts.pkl", 'rb') as f:
        texts = pickle.load(f)

    rake = Rake()
    conversation_log = []

    model = GPT4All(model_name='C://AI_MODELS//llama2_7b_chat_uncensored.ggmlv3.q4_0.bin')
    with model.chat_session():
        while True:
            user_input = input("You: ")

            summary_content = None  # Initialize summary content as None
            search_type = None  # Initialize search type as None

            if user_input.lower().startswith("search-deep:"):
                search_type = "Deep Search"
                search_query = user_input[len("search-deep:"):].strip()
                search_results = deep_search(search_query, embeddings, texts)
                summary_content = display_search_results_and_summary(search_results, "Deep Search", model)

                # Add search results and their summary to the conversation log
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = {
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "search_results": search_results,
                    "summary": summary_content
                }
                conversation_log.append(entry)
            
            elif user_input.lower().startswith("search:"):
                search_type = "Search"
                search_query = user_input[len("search:"):].strip()
                search_results = simple_search(search_query, texts)
                summary_content = display_search_results_and_summary(search_results, "Search", model)

                # Add search results and their summary to the conversation log
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = {
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "search_results": search_results,
                    "summary": summary_content
                }
                conversation_log.append(entry)

            else:
                if user_input.lower() in ['exit', 'quit']:
                    break

                rake.extract_keywords_from_text(user_input)
                keywords = rake.get_ranked_phrases()

                blob = TextBlob(user_input)
                sentiment = blob.sentiment.polarity

                context_texts = get_most_similar_texts(user_input, embeddings, texts)
                context = " ".join(context_texts)

                full_prompt = guiding_prompt + context + " " + user_input

                response = model.generate(prompt=full_prompt, temp=0.5)
                if isinstance(response, dict) and 'content' in response:
                    print(f"Chatbot: {response['content']}")
                    response_content = response['content']
                else:
                    print(f"Chatbot: {response}")
                    response_content = response
                
                update_context(user_input, response_content)

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = {
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "keywords": keywords,
                    "sentiment": "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral",
                    "response": response_content
                }
                conversation_log.append(entry)

    # After finishing the chat session, save the current session logs to the existing logs
    with open("conversation_log.json", "w") as f:
        combined_logs = existing_log + conversation_log
        json.dump(combined_logs, f, indent=4)

if __name__ == "__main__":
    main()

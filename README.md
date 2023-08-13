# shadowFACTS
![shadowfax](https://github.com/EveryOneIsGross/shadowFACTS/assets/23621140/b1a42b05-b7f5-4cec-8c9f-db820aaaa13b)

shadowFACTS is a conversational agent designed to provide contextually relevant responses based on an imported data source. shadowFACTSs name relates to the deep-search algorithm using a euclydian vector mask of farthest related reponses before looking for similarities in a reduced vector space. Otherwise it uses standard cosine searches to add chunks of txt into the agents prompt for some on topic context. This was just a quick tool for me to chat with a .txt while I tried to understand it. 

---

Core Features:

Text Importing:

Users can seamlessly input data either by providing a .txt file or directly typing in the text. This data forms the foundational knowledge from which shadowFACTS operates.

Memory Contextualization:

shadowFACTS has a contextual memory mechanism, keeping track of the most recent conversations. This context history is crucial in ensuring the bot's responses are coherent and in line with the ongoing dialogue.

---

Search Algorithms:

**Simple Search:**

Quickly scans the imported texts for the user's query, returning texts that contain the query verbatim. 

**Deep Search:**

An advanced search mechanism that masks out a certain percentage of the most dissimilar results (considered as "shadows"), enhancing the relevance of search results. It utilizes cosine similarity to find the most pertinent pieces of text.

Embedding Transformation:

First, the user's query is transformed into an embedding vector using Embed4All. This transformation captures the semantic essence of the query in a high-dimensional space.

Euclidean Distance Measurement:

Euclidean distance is then computed between the query's embedding and all the embeddings of the stored text chunks.
This distance provides a measure of how "far" each text chunk is from the user's query in the embedding space. Smaller distances imply higher relevance.

Masking "Shadows":

The most dissimilar results (those with the largest Euclidean distances) are considered as "shadows" and are masked out. This step ensures that the least relevant information is excluded from the search results. The proportion of results that are masked out can be adjusted via the mask_percentage parameter.

Fetching the Most Relevant Texts:

From the remaining pool of text chunks (after masking the shadows), the ones with the smallest Euclidean distances to the query are fetched. These are considered the most contextually relevant pieces of text for the given query.

Advantages of Using Euclidean Distance:

Euclidean distance, in the context of embedding spaces, provides a geometric interpretation of similarity. Points (or text chunks, in this case) that are closer in this space are more similar in terms of their content and meaning.
It's a straightforward and computationally efficient metric, making it suitable for real-time applications like chatbots.

Given:
- \( Q \) = user's query
- \( T \) = set of all text chunks in the bot's knowledge, \( T = \{t_1, t_2, ..., t_n\} \)
- \( E(.) \) = embedding function that transforms a text into its high-dimensional embedding representation
- \( mask\_percentage \) = a predefined threshold to filter out a proportion of the least relevant texts

Steps:

1. Transform the user's query into its embedding:
\[ e_Q = E(Q) \]

2. Compute the embeddings of all the text chunks:
\[ e_{T_i} = E(t_i) \; \text{for} \; i = 1 \; \text{to} \; n \]

3. Compute the Euclidean distance between the query's embedding and the embeddings of all text chunks:
\[ d_i = \|e_Q - e_{T_i}\|_2 \; \text{for} \; i = 1 \; \text{to} \; n \]

Where:
\[ \|e_Q - e_{T_i}\|_2 = \sqrt{\sum_{j=1}^{m} (e_Q^j - e_{T_i}^j)^2} \]
(m is the dimension of the embedding space)

4. Mask out the top \( mask\_percentage \) of the texts based on the highest distances (i.e., least relevant):
\[ M = \{ t_i \; | \; d_i \; \text{is in the top} \; mask\_percentage \; \text{of distances} \} \]
\[ R = T \setminus M \] (R is the remaining set of text chunks after masking)

5. From the remaining texts in R, select the ones with the smallest distances (i.e., most relevant) to return to the user.

This formal representation succinctly describes the methodology behind the "deep search" process, with each step clearly represented in mathematical terms.

---

Embeddings & Similarity Measurement:

shadowFACTS utilizes Embed4All to create embeddings for chunks of the imported text. These embeddings play a crucial role in the "deep search" mechanism, allowing for more nuanced text similarity comparisons. Cosine similarity is employed to identify the most contextually relevant chunks of text for a given query.

Dynamic Prompt Generation:

For standard queries (non-search ones), the bot constructs a prompt that combines a guiding principle, the most contextually relevant portions of the imported text, and the user's input. This ensures that the generated response is both grounded in the bot's foundational knowledge and relevant to the ongoing conversation.
Keyword Extraction & Sentiment Analysis:

ShadowFACTS leverages RAKE (Rapid Automatic Keyword Extraction) to discern the key topics from user input.
Sentiment analysis, powered by TextBlob, determines the emotional tone of the user's input, providing an additional layer of context.

Setup & Persistence:

The bot has mechanisms for saving and loading conversation logs and embeddings, ensuring continuity across sessions.
The chat interface is powered by GPT4All keeping your data local.

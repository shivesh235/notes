## Hands On LLM

### Embdeddings <a id="embeddings"></a>

The sources emphasize that **embeddings are fundamental to how Large Language Models (LLMs) understand and process language**. They serve as **numerical representations of text** (at the level of words, subwords/tokens, sentences, or entire documents) that capture their meaning and relationships in a vector space.

Here's a discussion of embeddings in the larger context of LLMs, drawing from the sources:

*   **The Necessity of Embeddings for LLMs:** LLMs operate on numerical data. Tokenizers first break down text into smaller units called tokens. Then, **each token in the tokenizer's vocabulary is associated with a vector called an embedding**. This **embedding vector is the initial numerical input to the LLM**. Without embeddings, LLMs could not process textual input.

*   **Types of Embeddings Discussed:** The sources cover several types of embeddings:
    *   **Word Embeddings:** Models like **word2vec** were early successful attempts at capturing the meaning of individual words in dense vector representations. They learn semantic relationships by analyzing which words appear near each other in large amounts of text. These embeddings are **static**, meaning a word has the same embedding regardless of its context.
    *   **Token Embeddings:** In the context of LLMs, embeddings are often discussed at the **token level**. These are the numerical representations of the individual tokens produced by the tokenizer and are the direct input to the Transformer architecture.
    *   **Contextualized Word Embeddings:** Unlike static word embeddings, **LLMs generate contextualized word embeddings**. This means that the embedding of a word or token varies depending on the surrounding words in the sentence. This ability to encode context is a key advancement enabled by architectures like Recurrent Neural Networks (RNNs) and, more effectively, the **Transformer using the attention mechanism**. Representation models like BERT are particularly adept at creating these contextualized embeddings.
    *   **Sentence/Text Embeddings:** Many applications require representing entire sentences, paragraphs, or documents as single vectors. **Text embedding models** are specifically designed to produce these embeddings, often by averaging the token embeddings of the text. Libraries like `sentence-transformers` are used to leverage pretrained models for this purpose.
    *   **Multimodal Embeddings:** The sources also introduce **multimodal embedding models** like CLIP, which can create embeddings for different types of data, such as text and images, in the **same vector space**. This allows for comparing and relating information across modalities.

*   **Creation of Embeddings:**
    *   **Word2vec:** Word embeddings in models like word2vec are learned through training a neural network to predict whether two words are likely to appear in the same context. This process updates the embedding vectors to reflect the relationships between words.
    *   **Within LLMs:** For LLMs, the initial token embeddings are often **randomly initialized**. During the **pretraining process** on massive datasets, the model learns the optimal values for these embeddings to effectively represent the patterns and meanings in the language. For contextualized embeddings, the Transformer architecture, particularly the attention mechanism, plays a crucial role in generating context-aware representations of each token based on its relation to other tokens in the input sequence.

*   **Uses of Embeddings in the Context of LLMs:** Embeddings are leveraged in numerous ways in the field of Language AI:
    *   **Input to LLMs:** As mentioned, token embeddings serve as the primary input for LLMs to process and generate text.
    *   **Semantic Similarity:** Embeddings allow for measuring the **semantic similarity** between words, sentences, or documents by calculating the distance between their corresponding vectors in the embedding space. Closer vectors indicate more similar meaning. This is fundamental for tasks like semantic search and clustering.
    *   **Downstream Tasks:** Embeddings are used as features for various downstream NLP tasks:
        *   **Text Classification:** Both task-specific models and general-purpose embedding models can be used for text classification. Embedding the text and then using these embeddings to train a classifier is a common approach.
        *   **Text Clustering and Topic Modeling:** By embedding documents and then applying clustering algorithms to these embeddings, semantically similar documents can be grouped together. Topic modeling techniques like BERTopic leverage embeddings to identify underlying themes in a collection of documents.
        *   **Semantic Search and Retrieval-Augmented Generation (RAG):** Embeddings are central to semantic search, where search queries and documents are embedded, and the most relevant documents are retrieved based on the similarity of their embeddings. In RAG systems, retrieved documents (obtained through embedding similarity) are provided as context to the LLM to help it generate more accurate and factual answers.
        *   **Recommendation Systems:** The concept of embedding is also valuable in domains outside of natural language, such as recommendation systems, where items like songs can be embedded based on user interactions (e.g., playlists) to find similar items.
        *   **Multimodal Applications:** Multimodal embeddings enable tasks like image retrieval based on text queries and vice versa.

*   **Relationship with Tokenizers:** The **tokenizer and the embedding layer of an LLM are tightly coupled**. A pretrained language model is linked with its specific tokenizer, and they cannot be easily interchanged without retraining. The tokenizer determines the vocabulary of tokens, and the embedding layer holds a vector representation for each of these tokens.

*   **Training and Fine-tuning Embedding Models:** The sources discuss methods for creating and improving embedding models. This can involve training from scratch or **fine-tuning pretrained models** on specific tasks or datasets to optimize the quality of the embeddings for those particular use cases. **Contrastive learning**, where the model learns to distinguish between similar and dissimilar pairs of data, is a key technique used in training embedding models, including word2vec and sentence embedding models. Techniques like **multiple negatives ranking (MNR) loss** are used to train models to differentiate a positive example from multiple negative examples.

In summary, embeddings are a cornerstone of LLM technology, enabling the conversion of human language into a numerical format that these models can process. They capture semantic meaning and relationships at different levels of granularity and are crucial for a wide range of LLM applications, from the fundamental task of language understanding and generation to more complex tasks like semantic search, RAG, and multimodal interaction. The quality and nature of embeddings significantly impact the performance and capabilities of LLMs and the systems built upon them.

---

### Definition: Vector representations of tokens/text <a id="definition"></a>

The sources define **vector representations of tokens/text** as the core concept of **embeddings**, which are numerical representations of language units that capture their meaning and relationships. In the larger context of embeddings and Large Language Models (LLMs), these vector representations are fundamental and permeate various aspects of how language is processed and utilized.

Here's a discussion based on the sources:

*   **Core Definition:** The sources explicitly state that **embeddings are vector representations of data, specifically designed to capture its meaning**. For language, this data can be individual words or subwords (tokens), as well as larger segments of text like sentences or documents. These numerical representations, also called vectors or vector representations, are the way computers can "understand" and process human language.

*   **Token Embeddings as Input to LLMs:** At the most granular level, LLMs work with **token embeddings**. When text is input into an LLM, it is first broken down into tokens by a tokenizer. Then, **each token in the tokenizer's vocabulary is converted into a corresponding vector embedding**. These token embeddings are the initial numerical input that the LLM processes through its network layers.

*   **Semantic Meaning in Vector Space:** A key aspect of these vector representations is their ability to capture **semantic meaning**. Models like word2vec aimed to learn these semantic representations by training on vast amounts of text, such that words appearing in similar contexts would have closer vector representations. This concept extends to LLMs, where the goal is to create embeddings where texts with similar meanings are located closer to each other in a high-dimensional vector space.

*   **Contextualized vs. Static Vector Representations:** While early methods like bag-of-words and word2vec created **static** vector representations (where a word had the same vector regardless of context), LLMs, particularly Transformer-based models, generate **contextualized** vector representations. This means that the vector representation of a token is influenced by the surrounding tokens in the sequence, allowing the model to understand nuances in meaning based on context.

*   **Generating Vector Representations:** The process of generating these vector representations varies depending on the technique:
    *   **Bag-of-words:** Creates vectors based on the frequency of words in a text, forming a sparse representation.
    *   **Word2vec:** Uses neural networks to predict the likelihood of words appearing together, learning dense vector embeddings in the process.
    *   **LLMs:** Initial token embeddings are often random and are refined during the pretraining process on large datasets. The Transformer architecture, with its attention mechanism, is crucial for generating contextualized embeddings. Text embedding models can generate sentence or document embeddings, often by aggregating token embeddings.

*   **Applications Leveraging Vector Representations:** The utility of these vector representations (embeddings) is extensive in the realm of LLMs and Language AI:
    *   **Semantic Search:** By embedding search queries and documents into the same vector space, the similarity between them can be measured, enabling semantic search.
    *   **Text Classification:** Vector representations of text can serve as input features for classification models.
    *   **Text Clustering and Topic Modeling:** Grouping semantically similar documents is achieved by clustering their vector representations.
    *   **Retrieval-Augmented Generation (RAG):** Vector representations are used to retrieve relevant documents based on a query, which are then used as context for an LLM to generate answers.
    *   **Recommendation Systems:** Items (like songs) can be represented as vectors based on user behavior, allowing for recommendations based on the similarity of these vector representations.
    *   **Multimodal Applications:** Models like CLIP create vector representations of both images and text in a shared space, enabling cross-modal understanding and retrieval.

In essence, **vector representations of tokens and text (embeddings) are the crucial bridge between human language and the numerical world of LLMs**. They enable LLMs to capture meaning, understand context, and perform a wide array of language-related tasks by operating on these numerical representations in a meaningful vector space.

---

### Types <a id="types"></a>

The sources discuss several **types of embeddings**, which represent different levels of abstraction and serve various purposes within the larger context of Large Language Models (LLMs). These types include word embeddings, token embeddings, contextualized word embeddings, sentence/text embeddings, and multimodal embeddings.

Here's a breakdown of these embedding types in the context of LLMs:

*   **Word Embeddings:** These are **static vector representations of individual words** that aim to capture their semantic meaning. Models like **word2vec** and GloVe learn these embeddings by analyzing the co-occurrence of words in large text corpora. Word embeddings represent the properties of words, and semantically similar words tend to have embeddings that are closer in the vector space. While crucial in the history of Language AI, **static word embeddings have largely been replaced by contextualized embeddings produced by LLMs** for many language processing tasks. However, they are still useful outside of LLMs, such as in recommendation systems.

*   **Token Embeddings:** When text is processed by an LLM, it is first broken down into smaller units called **tokens** by a tokenizer. **Each token in the tokenizer's vocabulary is then mapped to a token embedding**, which is a numerical vector. These **token embeddings serve as the initial input to the language model**. Before training, these vectors are typically initialized randomly. During the training process, the LLM learns to adjust these embeddings to effectively capture the patterns and meaning of the language. A language model holds an embedding vector for each token in its tokenizer's vocabulary.

*   **Contextualized Word Embeddings:** Unlike static word embeddings, LLMs like BERT and those utilizing the Transformer architecture generate **contextualized word embeddings**. This means that the embedding for a specific word **varies depending on the context in which it appears in a sentence**. For example, the word "bank" will have a different embedding when referring to a financial institution versus the bank of a river. This ability to encode context is a key strength of modern LLMs, achieved through mechanisms like the **attention mechanism**. Representation models (encoder-only) like BERT excel at generating these contextualized representations. These embeddings are used for various downstream tasks like named-entity recognition and extractive text summarization.

*   **Sentence/Text Embeddings:** Many applications require representing **entire sentences, paragraphs, or documents as a single vector**. **Text embedding models** are designed to produce these embeddings. One common method to generate text embeddings is by **averaging the token embeddings** of the text. However, high-quality text embedding models are often trained specifically for this task. Libraries like `sentence-transformers` provide access to pretrained models for generating text embeddings, which are crucial for applications like **semantic search, topic modeling, and text classification**.

*   **Multimodal Embeddings:** The sources also introduce the concept of **multimodal embeddings**, which can represent different types of data, such as **text and images, in the same vector space**. Models like **CLIP (Contrastive Language-Image Pre-training)** are examples of multimodal embedding models. By embedding text and images into a shared space, it becomes possible to **compare and relate information across these modalities**, enabling tasks like image retrieval based on text queries and zero-shot classification of images using textual descriptions.

In the larger context of LLMs, these different types of embeddings play crucial roles at various stages and for different purposes. Token embeddings are the fundamental building blocks for processing language within LLMs. Contextualized embeddings enhance the model's understanding by considering the surrounding words. Sentence/text embeddings enable LLMs to work with larger chunks of text for tasks like search and document analysis. Finally, multimodal embeddings extend the capabilities of LLMs beyond text to understand and reason about other types of data, broadening their applicability. The development and refinement of these embedding techniques are central to the ongoing advancements in the field of Large Language Models and Language AI.

* [**Word embeddings (static)**](#word-embeddings)
* [**Contextualized-word-embeddings**](#contextualized-word-embeddings)
* [**Sentence Embeddings**](#sentence-embeddings)

**Word embeddings (static)** <a id="word-embeddings"></a>

The sources state that **static word embeddings** are **vector representations of individual words that aim to capture their semantic meaning**. Models like **word2vec**, **GloVe**, and fastText are examples of methods that generate these types of embeddings. These models learn the embeddings by training on large amounts of textual data, like Wikipedia, by looking at which other words tend to appear next to each other in a given sentence. The idea is that words that are used in similar contexts will have vector representations that are closer to one another in the embedding space, thus capturing a degree of semantic similarity.

Here are some key aspects of static word embeddings in the larger context of embedding types:

*   **Semantic Meaning:** Static word embeddings attempt to **capture the meaning of words by representing their properties** in a vector. For instance, the embedding for "baby" might be closer to "newborn" and "human" than to "apple". This allows for measuring the semantic similarity between words using distance metrics.

*   **Training Process:** Models like word2vec use neural networks to generate these embeddings. They are trained on tasks like predicting whether two words are likely to be neighbors in a sentence. During this training, the embeddings are updated to reflect the relationships between words.

*   **Static Nature:** A key characteristic of these word embeddings is that they are **static**. This means that **a word has the same embedding regardless of the context in which it is used**. For example, the word "bank" will have the same embedding whether it refers to a financial institution or the bank of a river. This is a significant limitation compared to the contextualized embeddings generated by LLMs.

*   **Historical Significance:** Word embeddings like word2vec were **one of the first successful attempts at capturing the meaning of text in embeddings**. They represented a significant improvement over earlier methods like bag-of-words, which treated language as a literal collection of words and ignored the semantic nature of text.

*   **Current Relevance:** While **largely replaced by contextualized word embeddings produced by language models for many language processing tasks**, static word embeddings still have relevance. The sources mention their usefulness **outside of NLP**, for example, in **recommendation systems**. By treating items like songs as words and user interactions (like playlists) as sentences, similar embedding techniques can be used to recommend related items.

*   **Contrast with Contextualized Embeddings:** The development of LLMs brought about **contextualized word embeddings**, where the representation of a word changes based on the surrounding words in a sentence. This overcomes the limitation of static embeddings by allowing the model to understand different meanings of a word depending on its context. Representation models like BERT are specifically designed to generate these contextualized embeddings.

In summary, static word embeddings like those generated by word2vec and GloVe were a crucial step in the history of embedding techniques, enabling a basic form of semantic understanding in computers. However, their inability to capture context-dependent meaning led to the development of contextualized embeddings in LLMs. Despite this, static word embeddings remain relevant in specific applications outside the core processing of LLMs.

**Contextualized word embeddings** <a id="contextualized-word-embeddings"></a>


The sources highlight **contextualized word embeddings** as a significant advancement over static word embeddings in capturing the nuances of language. Here's a discussion of contextualized word embeddings within the larger context of embedding types, drawing from the sources:

*   **Definition and Generation:** **Contextualized word embeddings are dynamic vector representations of words where the embedding for a word changes based on the context in which it appears in a sentence**. Unlike static word embeddings, which assign a single, fixed vector to each word, contextualized embeddings are generated by language models (LLMs) by considering the surrounding words. Models utilizing the **Transformer architecture**, such as **BERT** and the GPT family, are capable of producing these contextualized representations.

*   **The Role of Attention:** The ability to generate contextualized embeddings is largely attributed to mechanisms like the **attention mechanism**. Attention allows the model to focus on the parts of the input sequence that are most relevant to a particular word, thus incorporating context into its representation. This enables the model to understand that a word like "bank" can have different meanings (financial institution vs. riverbank) depending on the other words in the sentence.

*   **Advantages Over Static Word Embeddings:** The primary advantage of contextualized word embeddings is their ability to capture the **meaning of a word as it is used in a specific instance**. Static word embeddings, like those from word2vec, assign the same embedding to a word regardless of its context, which can lead to ambiguity in understanding. Contextualized embeddings provide a richer and more accurate representation of word meaning by taking the surrounding text into account.

*   **Relationship to Token Embeddings:** The process of generating contextualized word embeddings typically starts with **token embeddings**. When a sentence is input into an LLM, it is first broken down into tokens, and each token is associated with an initial token embedding. These token embeddings then flow through the layers of the language model, where the attention mechanism and other computations are applied to generate the final contextualized word embeddings. Therefore, token embeddings serve as the foundation upon which contextualized word embeddings are built.

*   **Use Cases and Applications:** Contextualized word embeddings are crucial for a wide range of downstream tasks. The sources specifically mention applications such as **named-entity recognition (NER)** and **extractive text summarization**, where understanding the context of words is essential. Furthermore, these contextualized vectors are even used in systems like **AI image generation** (e.g., DALL·E, Midjourney, Stable Diffusion), highlighting their versatility in representing meaning across different modalities.

*   **Contrast with Sentence/Text Embeddings:** While contextualized word embeddings represent individual words within a context, **sentence/text embeddings aim to represent larger units of text** (sentences, paragraphs, or documents) as a single vector. Often, sentence embeddings can be derived from contextualized word embeddings (e.g., by averaging them). However, models can also be specifically trained to generate high-quality sentence or text embeddings for tasks like semantic search and topic modeling.

*   **Evolution of Embedding Techniques:** Contextualized word embeddings represent a significant step in the evolution of embedding techniques. They addressed the limitations of earlier static word embeddings and paved the way for more sophisticated language understanding and generation capabilities in LLMs. The success of models like BERT, which heavily rely on contextualized representations, has solidified their importance in modern NLP. Representation models (encoder-only) primarily focus on generating these rich contextualized embeddings.

In essence, contextualized word embeddings are a key innovation in the field of Language AI, enabling LLMs to understand and process language with a much greater degree of sensitivity to context than earlier static methods. They build upon the concept of token embeddings and serve as a crucial intermediate representation that powers a wide array of language understanding tasks, bridging the gap between individual words and the meaning of larger text units.

**Sentence Embeddings** <a id="sentence-embeddings"></a>

The sources describe **sentence embeddings** as **vector representations that capture the meaning of entire sentences, paragraphs, or even documents as a single vector**. This contrasts with word embeddings, which represent individual words, and token embeddings, which represent the smaller units that text is broken down into for processing by language models.

Here's a discussion of sentence embeddings within the larger context of embedding types, drawing from the provided text:

*   **Purpose and Level of Abstraction:** While token embeddings are fundamental to how LLMs process text, and contextualized word embeddings capture word meaning within a sentence, **sentence embeddings operate at a higher level of abstraction**. Their primary goal is to **represent the overall semantic content of a longer piece of text** in a concise numerical vector. This allows for comparisons and operations on entire sentences or documents.

*   **Generation Methods:** The sources outline several ways to produce sentence embeddings:
    *   One common approach is to **average the values of all the token embeddings produced by a language model for the sentence**.
    *   However, the sources emphasize that **high-quality sentence embedding models are often trained specifically for the task of generating these embeddings**.
    *   The **sentence-transformers library** is presented as a popular tool for leveraging **pretrained embedding models** that are optimized for generating sentence embeddings.

*   **Relationship to Word and Token Embeddings:** Sentence embeddings can be seen as building upon the foundation of word or token embeddings. While a language model first processes text into tokens and generates token embeddings (which can become contextualized word embeddings), sentence embedding models take this a step further to produce a single vector for a larger span of text. In some cases, sentence embeddings are derived from the underlying word embeddings, but dedicated models are designed to directly encode the meaning of sequences.

*   **Applications:** The sources highlight the **immense usefulness of sentence embeddings in powering various Language AI applications**. Key examples include:
    *   **Semantic Search:** By embedding search queries and documents into the same vector space, semantic similarity can be measured, allowing for the retrieval of relevant information based on meaning rather than just keyword matches. This is a core component of dense retrieval systems.
    *   **Topic Modeling:** Sentence embeddings can be used as features for clustering algorithms to group semantically similar documents, thereby uncovering underlying topics in a collection of texts. Tools like BERTopic leverage embedding models as a crucial first step in their pipeline.
    *   **Classification:** While the chapter on text classification also discusses task-specific models and generative models, it notes that **embedding models can be used to generate multipurpose embeddings which can then serve as input features for a classifier**.
    *   **Retrieval-Augmented Generation (RAG):** Sentence embeddings are fundamental to the retrieval component of RAG systems, where relevant documents (represented by their embeddings) are fetched based on the embedding of the user's query.

*   **Contrast with Other Embedding Types:**
    *   **Static Word Embeddings:** Sentence embeddings overcome a key limitation of static word embeddings, which cannot capture the meaning of a word in context, let alone the meaning of an entire sentence. Sentence embeddings aim to capture the holistic meaning of a sequence of words.
    *   **Contextualized Word Embeddings:** While contextualized word embeddings provide rich representations of individual words considering their surroundings, sentence embeddings aggregate this information (or are trained directly) to provide a single representation for the entire sentence. This is necessary for tasks that require comparing or operating on whole units of text.

*   **Training Sentence Embedding Models:** Chapter 10 delves into creating text embedding models and discusses techniques like **contrastive learning** used to train models to understand semantic similarity between sentences. Frameworks like **sentence-transformers (SBERT)** are specifically designed for training and using models that produce high-quality sentence embeddings. Fine-tuning existing language models using labeled data (e.g., natural language inference datasets) or unsupervised methods can improve the quality of the generated sentence embeddings for specific tasks or domains.

In summary, sentence embeddings are a crucial type of embedding that represents the meaning of entire text units. They build upon the concepts of token and word embeddings but operate at a higher level, enabling a wide range of applications that require understanding the semantic relationships between sentences and documents. Dedicated models and libraries like sentence-transformers have made it easier to generate and utilize these powerful representations.

---

### Creation <a id="creation"></a>

The sources discuss the **creation of embeddings** as a fundamental process in language AI, enabling computers to understand and process human language. Embeddings are **numeric representations of data (like words, sentences, or even images) that attempt to capture their meaning**.

Here's a breakdown of how the sources discuss the creation of different types of embeddings:

*   **Word Embeddings:**
    *   Early approaches like **word2vec** aimed to capture the meaning of words in embeddings by training on vast amounts of textual data. These models, which emerged in 2013, leverage **neural networks** to learn semantic representations by looking at the words that tend to appear next to each other in sentences.
    *   The training process involves initializing every word in the vocabulary with a random vector embedding. Then, the model attempts to predict whether pairs of words are likely to be neighbors in a sentence, updating the embeddings during this process to reflect the relationships between words. Words that tend to have the same neighbors will have embeddings closer to one another in the embedding space.
    *   The values within these embedding vectors represent properties that, to a computer, translate human language into a usable format.
    *   **Word embeddings created by word2vec are static**, meaning the embedding for a word remains the same regardless of the context in which it is used.
    *   Algorithms like **GloVe** are other examples of methods for creating pretrained word embeddings.

*   **Token Embeddings:**
    *   Before feeding text to a language model, it is split into smaller chunks called **tokens**. For the language model to process this, **tokens need to be turned into numeric representations called embeddings**, known as token embeddings.
    *   The creation of tokens is done by **tokenizers**, which use different methods to break down text (words, subwords, characters, bytes). The choice of tokenization method and the data it's trained on influences the resulting tokens.
    *   A **language model holds an embedding vector for each token in its tokenizer's vocabulary**. These vectors are initially random but are updated during the model's training process to enable the model's learned behavior.

*   **Contextualized Word Embeddings:**
    *   Language models, particularly those utilizing the **Transformer architecture** with the **attention mechanism**, can create **contextualized word embeddings**. Unlike static word embeddings, these representations capture the meaning of a word based on its surrounding context in a sentence.
    *   Encoder-only models (representation models) like BERT are capable of generating these contextualized embeddings.

*   **Text Embeddings (Sentence and Document Embeddings):**
    *   For tasks operating on larger units of text, **text embedding models** are used to produce a single vector that represents the meaning of a sentence, paragraph, or entire document.
    *   One common way to create text embeddings is by **averaging the token embeddings** produced by a language model. However, high-quality text embedding models are often trained specifically for this task.
    *   Libraries like **sentence-transformers** provide tools to load and use pretrained models for creating text embeddings.

*   **Multimodal Embeddings:**
    *   To handle data beyond text, **multimodal embedding models** are created to represent different modalities, such as images and text, in the same vector space.
    *   **CLIP (Contrastive Language-Image Pre-training)** is a well-known model that can compute embeddings for both images and text, allowing for comparison and tasks like image retrieval based on text queries. CLIP uses **contrastive learning** to align image and text embeddings.
    *   Models like **BLIP-2** extend text generation models to include vision by projecting visual features into the text embedding space.

*   **Training and Fine-tuning Embedding Models:**
    *   The creation of effective embedding models often relies on **contrastive learning**. This technique involves training a model to distinguish between similar and dissimilar pairs of data points, allowing it to learn the underlying semantic relationships.
    *   **Sentence-transformers** is a framework that popularizes contrastive learning for creating sentence embeddings. It uses techniques like **bi-encoders** and optimizes the similarity between pairs of sentences.
    *   **Pretrained language models** can be used as a starting point to **create embedding models from scratch** or to **fine-tune existing embedding models** for specific tasks or domains. Fine-tuning can adapt the embeddings to focus on aspects like sentiment or improve performance on specific downstream tasks like classification.
    *   **Natural Language Inference (NLI) datasets** are often used to generate contrastive examples (entailment as positive, contradiction as negative) for training embedding models.

In summary, the creation of embeddings is a crucial process in enabling language models and other AI systems to understand and work with various forms of data. Different techniques and architectures have been developed to create embeddings at different levels of abstraction (words, tokens, sentences, documents) and across different modalities, with contrastive learning being a central paradigm in training these models.

* **Word2vec algorithm**
* **Language Models (e.g. BERT)**

* **Word2vec algorithm**
The sources discuss the **word2vec algorithm** as an early and significant attempt at **"creating" meaningful numeric representations of text in the form of embeddings**. Released in 2013, word2vec aimed to **capture the meaning of text in embeddings** by learning semantic representations of words through training on vast amounts of textual data. This process of learning these representations is closely tied to **contrastive training**.

Here's how the sources explain the creation of word embeddings using word2vec and its connection to contrastive training:

*   **Capturing Meaning through Context:** Word2vec leverages **neural networks** to generate word embeddings by looking at the words that tend to appear next to each other in sentences. The algorithm uses a **sliding window** to generate training examples from text. For a central word, its neighbors within the window are considered related.

*   **The Prediction Task:** The training involves a **classification task** where a neural network attempts to **predict if two words commonly appear in the same context or not**. The network takes two words as input and outputs 1 if they tend to be neighbors and 0 if not.

*   **Learning Relationships and Updating Embeddings:** Initially, every word in the vocabulary is assigned a random vector embedding. During the training process, as the model tries to predict the neighbor relationship, the **embeddings are updated**. If two words tend to have the same neighbors, their embeddings will be closer in the embedding space, and vice versa. This process **distills the relationship between words into the embedding**.

*   **The Role of Contrastive Training:** The sources explicitly state that the word2vec algorithm uses **contrastive training**. To train the model effectively, it's crucial to provide not only positive examples (neighboring words) but also **negative examples**: pairs of words that are not typically neighbors. These negative examples are often generated by **random sampling from the dataset**. By contrasting positive pairs with negative pairs, the model learns to better distinguish between words that are semantically related and those that are not. The model is trained to output 1 for positive pairs and 0 for negative pairs.

*   **Skip-gram and Negative Sampling:** The word2vec algorithm relies on two main ideas: **skip-gram** (the method of selecting neighboring words) and **negative sampling** (adding negative examples by random sampling).

*   **Significance in the Context of Creation:** In the larger context of "Creation," word2vec was a **pioneering method for automatically creating dense vector embeddings** that captured semantic meaning. This was a significant advancement over earlier "bag-of-words" models, which treated language as a literal collection of words and ignored their underlying semantic relationships. Word2vec's ability to "create" these meaningful representations was a crucial step in enabling computers to better "understand" language. The **contrastive training aspect** was fundamental to this creation process, as it provided the learning signal necessary for the model to differentiate and encode semantic relationships into the generated embeddings. These embeddings then became the foundation for various downstream tasks and influenced subsequent developments in creating more sophisticated embedding techniques used in modern language models.

In essence, word2vec, through its use of contrastive training, demonstrated a powerful way to **"create" numerical representations of language that held semantic value**, marking a significant step in the evolution of language AI.

* **Language Models (e.g. BERT)**

The sources discuss language models like **BERT (Bidirectional Encoder Representations from Transformers)** and **DeBERTa (Decoding-enhanced BERT with disentangled attention)** primarily within the context of **"creating" representations of language for understanding**. These models are categorized as **representation models** or **encoder-only models**. Their "creation" involves specific architectures and training procedures designed to enable them to learn and encode the meaning of text.

Here's a breakdown of what the sources say about BERT and DeBERTa in this larger context of "Creation":

*   **Architecture and Purpose:**
    *   BERT was introduced in 2018 as a new architecture that could be leveraged for a wide variety of tasks and served as a foundation of Language AI for years. It is an **encoder-only architecture** that focuses on **representing language**. The architecture stacks encoder blocks consisting of self-attention followed by feedforward neural networks.
    *   DeBERTa is mentioned as one of the many variations of BERT that have been developed over the years.
    *   These representation models, unlike generative models, **do not primarily generate text** but are commonly used for task-specific use cases like **classification**. They mainly focus on **representing language by creating embeddings**. The distinction between representation and generative models is a key concept in the book. Representation models are visually depicted in teal with a vector icon, indicating their focus on vectors and embeddings.

*   **Training Methods:**
    *   The training of these encoder stacks, like BERT, is described as a **difficult task** that BERT approaches by adopting a technique called **masked language modeling**. This method involves masking a part of the input for the model to predict. This prediction task allows BERT to **create more accurate (intermediate) representations of the input**.
    *   This architecture and training procedure make BERT and related architectures **incredible at representing contextual language**. By training BERT on massive datasets like the entirety of Wikipedia, it learns to understand the semantic and contextual nature of text.

*   **Embeddings and Representation:**
    *   BERT-like models are commonly used for **transfer learning**, where they are first pretrained for language modeling and then **fine-tuned for a specific task**. For instance, a pretrained BERT model can be fine-tuned for text classification.
    *   BERT and DeBERTa are also used as **foundation models** that can be **fine-tuned to create task-specific models** (e.g., for sentiment analysis) or **embedding models** that generate general-purpose embeddings for various tasks like semantic search.

*   **Tokenization:**
    *   BERT uses the **WordPiece tokenization method**. The sources provide details on the vocabulary size and special tokens (like `[UNK]`, `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`) used by both the uncased and cased versions of the BERT base model. These tokenizers break down text into tokens, which are then fed into the BERT model to generate representations.

*   **Applications:**
    *   BERT's ability to create meaningful representations led to its adoption by search engines like Google, marking a significant leap in the history of Search by enabling **semantic search**.
    *   BERT-like models are considered solid baselines for various tasks.

In the larger context of "Creation" in Language AI, BERT and DeBERTa represent a significant advancement in **how computers can "create" understanding of language**. They moved beyond simple keyword matching (like bag-of-words models) by "creating" dense vector representations that capture semantic and contextual information. Their architecture and training methodologies laid the groundwork for many subsequent developments in the field of language understanding and continue to be relevant as foundation models for fine-tuning and adaptation to specific tasks and domains. The "creation" of these models involved innovating on neural network architectures (Transformer encoder), training objectives (masked language modeling), and tokenization techniques, ultimately leading to more powerful ways of representing and leveraging language data.

---

### Use cases <a id="use-cases"></a>

The sources highlight a wide array of **use cases for embeddings**, emphasizing their crucial role in various Language AI applications. Embeddings, as **numeric representations that capture the meaning and patterns in language**, serve as a foundational technology that empowers numerous tasks.

Here are some key use cases of embeddings discussed in the sources:

*   **Semantic Search and Retrieval**:
    *   Embeddings enable searching based on the **meaning of text rather than just keywords**. By converting both the search query and the documents into embeddings, systems can retrieve the **nearest neighbors** in the embedding space, indicating semantic similarity.
    *   This is a key component of **dense retrieval systems**.
    *   The sources provide an example of using Cohere for semantic search on a Wikipedia page by embedding sentences and finding those closest to a query.
    *   **Reranking models** also utilize embeddings to score the relevance of initial search results based on their semantic similarity to the query.
    *   **Retrieval-Augmented Generation (RAG)** systems leverage embeddings for retrieving relevant information from external knowledge bases to ground the language model's generation and reduce hallucinations.

*   **Text Classification**:
    *   Embeddings can be used as **features for training classification models**. By converting text into embedding vectors, machine learning models can learn to associate these vectors with specific categories or labels.
    *   The sources discuss using both **task-specific models** (fine-tuned for classification) and **general-purpose embedding models** (where embeddings are used as input for a separate classifier like logistic regression) for text classification.
    *   **Sentence embeddings** can be particularly useful for classifying entire documents or sentences.

*   **Text Clustering and Topic Modeling**:
    *   By converting documents into embeddings, **clustering algorithms** can group semantically similar texts together. The proximity of embeddings in the vector space reflects the similarity of the underlying text.
    *   **Topic modeling techniques** like BERTopic utilize embeddings as a crucial first step in identifying themes within large amounts of text.

*   **Recommendation Systems**:
    *   The concept of embeddings extends beyond language and is used to represent items (like songs) as vectors. By treating playlists as sentences and songs as words, the **word2vec algorithm** can be used to create song embeddings that capture which songs frequently appear together. These embeddings can then be used to recommend similar songs.

*   **Named-Entity Recognition (NER) and Extractive Text Summarization**:
    *   **Contextualized word embeddings** produced by language models capture the meaning of words in their specific context, improving performance on tasks like NER (identifying entities in text) and extractive summarization (highlighting important parts of a text).

*   **Multimodal Applications**:
    *   **Multimodal embedding models** like CLIP can create embeddings for different types of data, such as text and images, in the same vector space. This enables use cases like **image retrieval based on text queries** and vice versa.

*   **Providing Memory to Language Models**:
    *   Embeddings can be used to represent past turns in a conversation, allowing language models to maintain context and remember previous interactions.

The sources emphasize that the effectiveness of embeddings in these use cases stems from their ability to **capture semantic relationships**, **handle context** (in the case of contextualized embeddings), and provide a **numerical representation** that machine learning algorithms can readily process. The development of techniques like **word2vec** was a significant early step in "creating" these meaningful embeddings, and subsequent models, including large language models, have further advanced the creation and application of embeddings across a diverse range of tasks. The process of **contrastive training** is also highlighted as a key method for learning effective embeddings by contrasting similar and dissimilar pairs of data.

* **Semantic Search and Retrieval-Augmented Generation (RAG)**
* **Recommendation systems**
* **Text Clustering and Topic Modeling**
* **Named-entity recognition (NER)**

* **Semantic Search and Retrieval-Augmented Generation (RAG)**

The sources dedicate **Chapter 8** to the discussion of **Semantic Search and Retrieval-Augmented Generation (RAG)**, highlighting them as significant **use cases** for language models. These applications demonstrate how language models, particularly through the use of **embeddings**, can enhance and power information retrieval and knowledge-intensive tasks.

**Semantic Search as a Use Case:**

*   The sources emphasize that **semantic search**, which enables searching by **meaning** rather than just **keyword matching**, was one of the first language model applications to achieve broad industry adoption. Google's use of BERT in Google Search is cited as a prime example of a major leap forward enabled by semantic search. Microsoft Bing also reported significant quality improvements by using large transformer models for search. This immediate and dramatic improvement in established search systems underscores the practical value of language models.
*   The core of semantic search, as described, relies on the concept of **embeddings**. By converting both the search query and the documents into dense vector representations (embeddings), the search problem transforms into finding the **nearest neighbors** in the embedding space. Documents with semantically similar content will have embeddings that are close to each other.
*   The book outlines **three broad categories** of language models used for search:
    *   **Dense Retrieval:** This method relies entirely on the **similarity of text embeddings** to retrieve relevant results. The process involves embedding the search query and then finding the archive of texts with the closest embeddings. The sources provide a practical example using Cohere to perform dense retrieval on a Wikipedia page.
    *   **Reranking:** Language models can also be used as a final step in a search pipeline to **reorder a subset of initial search results** based on their relevance to the query. This step can significantly improve the quality of search results, as demonstrated by Microsoft Bing's enhancements using BERT-like models. Cohere's Rerank endpoint and the use of Sentence Transformers for local reranking are mentioned as practical tools. Reranking models often function as **cross-encoders**, evaluating the query and each potential result simultaneously to assign a relevance score.
    *   The sources also mention the importance of considering the **caveats of dense retrieval**, such as challenges with exact phrase matching and performance in domains different from the training data. **Hybrid search**, combining semantic and keyword search, is suggested as a more robust approach. The chapter also discusses techniques for **chunking long texts** into manageable segments for embedding. Furthermore, it highlights that **fine-tuning embedding models** on query-relevant result pairs can significantly improve dense retrieval performance.

**Retrieval-Augmented Generation (RAG) as a Use Case:**

*   The rapid advancement of text generation models led to users expecting factual answers from them. The issue of **model "hallucinations"** (generating incorrect or outdated information) emerged as a significant problem. **Retrieval-Augmented Generation (RAG)** arose as a leading solution to mitigate this by incorporating search capabilities to provide relevant information to the language model before it generates an answer.
*   RAG systems are described as **text generation systems that integrate search capabilities** to enhance factuality and ground the generation on specific datasets. This enables use cases like "**chat with my data**," where an LLM can be grounded on internal company documents or specific data sources. The increasing trend of search engines incorporating LLMs to summarize results or answer questions also falls under the umbrella of RAG.
*   The transition from a basic search system to a RAG system involves **adding an LLM at the end of the search pipeline**. The top retrieved documents are presented to the LLM along with the user's question, and the LLM is instructed to answer based on the provided context. This generation step is termed **grounded generation**. The sources provide an example of building a basic RAG system using Cohere's LLM and search functionalities.
*   The role of a **prompt template** in a RAG pipeline is crucial, as it serves as the central place to communicate the retrieved relevant documents to the LLM.
*   The book also touches upon **advanced RAG techniques** to further improve performance, including:
    *   **Query rewriting:** Using an LLM to rephrase the user's query to better suit the retrieval step.
    *   **Multi-hop RAG:** Handling complex questions that require a sequence of sequential searches.
    *   **Query routing:** Enabling the model to search across multiple data sources based on the nature of the query.
    *   **Agentic RAG:** Delegating more responsibility to the LLM to determine information needs and utilize various data sources as tools.
*   Finally, the chapter discusses the **evaluation of search systems** using metrics like **mean average precision (MAP)**. For RAG systems, evaluation is more complex and involves considering factors like **fluency, perceived utility, citation recall, citation precision, faithfulness, and answer relevance**.

In the larger context of use cases, semantic search and RAG stand out as powerful applications of language models that address fundamental challenges in information access and knowledge retrieval. They leverage the ability of language models to understand the meaning of text and generate coherent responses, paving the way for more intelligent and helpful AI systems across various domains.

* **Recommendation systems**

The sources indicate that **recommendation systems are a significant use case** that extends the application of embeddings beyond the realm of text and language generation. The concept of assigning **meaningful vector representations (embeddings) to objects** is highlighted as being useful in many domains, including recommender engines.

Here's a breakdown of what the sources say about recommendation systems in the larger context of use cases:

*   **Word Embeddings for Recommendation:** The sources explain how the **word2vec algorithm**, initially designed for creating word embeddings from text, can be adapted for recommendation systems. This is done by treating items (like songs) as words or tokens and sequences of items (like playlists) as sentences.
*   **Recommending Songs by Embeddings:** Chapter 2 specifically discusses an example of **recommending songs using embeddings derived from human-made music playlists**. By analyzing which songs frequently appear together in playlists, the word2vec algorithm can create song embeddings that capture the similarity between songs. These embeddings can then be used to recommend songs that are "similar" (i.e., have close embeddings) to a given song. The source even provides a hypothetical example of giving Michael Jackson's "Billie Jean" as input and receiving recommendations like "Kiss" by Prince & The Revolution and "Unchained" by Van Halen.
*   **Analogy to Language Processing:** This approach leverages the same principles as language processing, where the co-occurrence of words in sentences helps to determine their semantic relationships. In the context of recommendations, the co-occurrence of items in a sequence (like a playlist) helps determine the relationships between those items.
*   **Use of Word Tokenization Outside NLP:** The source notes that while word tokenization is being used less in NLP compared to subword tokenization, its usefulness has led to its application outside of NLP in use cases like recommendation systems.
*   **Embeddings as a General Concept:** The discussion of song recommendations emphasizes that **embeddings are a broadly applicable concept**. Assigning vector representations to objects based on their relationships in a given context can be valuable in various domains beyond language.
*   **Contrastive Training Connection:** The source mentions that the training of word2vec, which is used in the song recommendation example, employs **contrastive training**, a concept that is further explored in Chapter 10 in the context of creating text embedding models. This highlights a common underlying technique used for learning meaningful representations across different use cases.
*   **Part of Broader Applications:** The example of recommendation systems demonstrates how the fundamental concept of embeddings, which is also central to use cases like semantic search, text classification, and topic modeling (as mentioned in Chapter 2 and Part II), underpins a wide range of language AI and related applications.

In summary, the sources present recommendation systems, particularly the example of song recommendation using an adaptation of word2vec, as a compelling use case that showcases the versatility of embeddings. It demonstrates how the principles of learning relationships from sequences, initially developed for language, can be effectively applied to other domains to build intelligent systems. This reinforces the idea that embeddings are a foundational technology enabling a diverse set of applications within and beyond the traditional scope of language AI.

* **Text Clustering and Topic Modeling**

The sources dedicate **Chapter 5** to the discussion of **Text Clustering and Topic Modeling**, highlighting them as important techniques within the larger context of **unsupervised learning** use cases. While supervised methods like classification have been dominant, the potential of unsupervised techniques for grouping similar texts and discovering latent topics is significant.

**Text Clustering as a Use Case:**

*   The sources describe **text clustering** as a technique that aims to **group similar texts based on their semantic content, meaning, and relationships**. This enables use cases such as **efficient categorization of large volumes of unstructured text** and **quick exploratory data analysis**.
*   A common pipeline for text clustering involves three steps:
    *   **Converting documents to embeddings** using an embedding model. The sources emphasize the importance of choosing embedding models optimized for semantic similarity.
    *   **Reducing the dimensionality of embeddings** using a dimensionality reduction model.
    *   **Finding groups of semantically similar documents** with a cluster model.
*   The recent evolution of **language models**, which provide **contextual and semantic representations of text** through **embeddings**, has significantly enhanced the effectiveness of text clustering, moving beyond the limitations of bag-of-words approaches.
*   Beyond categorization and exploration, text clustering also finds use in **finding outliers, speeding up labeling processes, and identifying incorrectly labeled data**.

**Topic Modeling (with a focus on BERTopic) as a Use Case:**

*   **Topic modeling** is presented as an extension of text clustering, aiming to **discover abstract topics that appear in large collections of textual data**. Traditionally, topics are represented by keywords, but more advanced techniques aim for more coherent labels.
*   The sources heavily feature **BERTopic** as a **modular text clustering and topic modeling framework**. Its underlying algorithm involves two main steps:
    *   **Clustering semantically similar documents** using the same pipeline as described for text clustering (embedding, dimensionality reduction, clustering).
    *   **Modeling a distribution over words** in the corpus's vocabulary, often leveraging a **bag-of-words approach enhanced with c-TF-IDF** (class-based term frequency–inverse document frequency) to weigh words based on their relevance to a cluster.
*   A major advantage of **BERTopic** highlighted in the sources is its **modularity**. This allows users to **choose and swap different models for each step of the pipeline** (e.g., different embedding models, clustering algorithms), making it adaptable to various use cases and allowing for the integration of newly released models.
*   The **modularity of BERTopic** also enables various **algorithmic variants** for different use case requirements, such as **guided, semi-supervised, hierarchical, dynamic, multimodal, multi-aspect, online, and zero-shot topic modeling**.
*   The sources demonstrate the application of BERTopic on a dataset of **ArXiv articles in the field of Computation and Language**, showcasing its ability to identify and describe topics within a specific domain.
*   **Large Language Models (LLMs)**, both representation and generative, can be integrated into **BERTopic** to further refine topic representations:
    *   **Reranking techniques**, such as **KeyBERTInspired** (using similarities between document and keyword embeddings) and **Maximal Marginal Relevance (MMR)** (reducing redundancy in keywords), can improve the coherence and diversity of topic representations.
    *   **Text generation models** (like Flan-T5 and GPT-3.5) can be used to **generate more interpretable labels for topics** based on representative documents and keywords, significantly enhancing the understanding of the discovered topics.

In the larger context of use cases, **text clustering and topic modeling (especially with frameworks like BERTopic)** provide powerful unsupervised learning tools for extracting meaningful insights and structure from large amounts of unlabeled textual data. These techniques are crucial in scenarios where manual labeling is infeasible or when the goal is to discover hidden patterns and themes within a dataset. The modularity and the ability to integrate advanced language models make BERTopic a versatile framework applicable to a wide range of analytical and exploratory use cases across various domains.

* **Named-entity recognition (NER)**

The sources discuss **Named-entity recognition (NER)** primarily in **Chapter 11**, framing it as a significant use case for fine-tuning pretrained representation models. NER is defined as a task that involves **identifying specific entities such as people and places in unstructured text**.

In the larger context of use cases, the sources highlight the following points about NER:

*   **Classification at a Word Level:** Unlike document-level classification, NER focuses on **classifying individual tokens and/or words** within a text. This granular level of analysis enables the identification of specific entities.
*   **Practical Applications:** The sources explicitly mention **de-identification and anonymization tasks** as important use cases for NER, particularly when dealing with sensitive data. By identifying entities like names or locations, this information can be masked or removed to protect privacy.
*   **Fine-tuning Representation Models:** The primary approach discussed in the sources is **fine-tuning pretrained BERT models** for NER. This involves adapting a general-purpose language understanding model to the specific task of entity recognition.
*   **Data Preprocessing for NER:** A key aspect of using language models for NER is the **preprocessing of data to account for the word-level classification task**. The input data needs to be structured such that individual words and their corresponding entity labels are aligned.
*   **Tokenization Challenges:** The process of tokenization can split words into subtokens, creating a challenge for word-level labeled data. The sources explain the need to **align the labels with their subtoken counterparts** during tokenization. For example, if "Maarten" is labeled as a person (B-PER), and it's tokenized as "Ma", "##arte", and "##n", the labels should be adjusted (e.g., B-PER for "Ma" and I-PER for "##arte" and "##n") to indicate that these subtokens belong to the same entity.
*   **Common Datasets:** The **CoNLL-2003 dataset** (English version) is presented as a standard dataset used for NER, containing various entity types like person, organization, location, and miscellaneous. The sources also mention other datasets like `wnut_17` (for emerging and rare entities) and `tner/mit_movie_trivia` and `tner/mit_restaurant` (for more specific entity types).
*   **Evaluation:** While the sources discuss evaluation metrics for generative models in **Chapter 12**, they don't explicitly detail NER-specific evaluation metrics in Chapter 11. However, it is implied that the performance of fine-tuned NER models can be assessed.
*   **Adapters for NER:** The sources briefly mention that **specialized adapters** can be downloaded and swapped into transformer architectures for specific tasks like NER, offering an efficient way to adapt models without full fine-tuning.

In essence, the sources position NER as a crucial use case that benefits significantly from the contextual understanding capabilities of transformer-based language models like BERT. Fine-tuning these models on labeled NER datasets enables accurate identification of entities within text, which is vital for various downstream applications, particularly those involving information extraction and anonymization. The challenges in aligning word-level labels with subword tokenization are also highlighted as an important consideration in this use case.

---

### Fine-tuning <a id="fine-tuning"></a>

The sources emphasize that **fine-tuning is a crucial step to enhance the quality and applicability of embedding models** for specific tasks and domains. **Embeddings**, which are numeric representations capturing the meaning and patterns in language, are central to many Language AI use cases. Fine-tuning allows us to adapt general-purpose pretrained embedding models to better suit particular needs.

Here's what the sources say about fine-tuning in the larger context of embeddings:

*   **General Purpose of Fine-tuning:** Fine-tuning involves taking a previously trained model (which includes its initial embeddings) and further training it on a narrower task or specific dataset. This allows the model, and consequently its embeddings, to adapt to exhibit desired behavior or perform well on specific tasks.

*   **Fine-tuning Embedding Models Directly:** Chapter 10 is dedicated to **creating and fine-tuning text embedding models**. The sources state that fine-tuning can increase the representative and semantic power of these models.

*   **Contrastive Learning as a Fine-tuning Technique:** A major technique for **fine-tuning text embedding models is contrastive learning**. This method trains the model such that embeddings of similar documents are closer in vector space, while dissimilar documents are further apart. This is achieved by feeding the model examples of similar (positive) and dissimilar (negative) pairs of documents. The goal is to learn embeddings that are tuned to a specific task, distilling the relevance of classes and their relative meaning into the embeddings.

*   **Supervised Fine-tuning of Embedding Models:** Embedding models can be **fine-tuned using supervised data** where pairs or triplets of (dis)similar documents are available. This process optimizes the embeddings for tasks like semantic similarity or even sentiment similarity, depending on the nature of the training data. For instance, for sentiment classification, the fine-tuning can steer the embeddings so that documents with the same sentiment are closer in the embedding space.

*   **Fine-tuning for Dense Retrieval:** Chapter 8 discusses how the performance of embedding models for **dense retrieval** can be improved through fine-tuning. This involves using training data composed of queries and relevant results. The fine-tuning process aims to make the embeddings of relevant queries closer to the embeddings of the corresponding documents and farther from irrelevant ones.

*   **Augmented SBERT for Fine-tuning:** The sources introduce **Augmented SBERT** as a fine-tuning technique for bi-encoders (like Sentence-BERT) that involves using a small, annotated "gold" dataset to fine-tune a cross-encoder, then using the cross-encoder to label new sentence pairs ("silver" dataset), and finally training the bi-encoder on the combined dataset. This is presented as a way to leverage limited labeled data for more effective fine-tuning of embedding models.

*   **TSDAE for Domain Adaptation:** **TSDAE (Transformer-based Sequential Denoising Auto-Encoder)** is presented as an unsupervised method for training and **fine-tuning embedding models for domain adaptation**. This technique involves training an encoder to reconstruct a sentence from a noisy version of it, forcing the encoder to learn robust and domain-specific embeddings. This pretraining on a target domain can be followed by supervised fine-tuning.

*   **Fine-tuning BERT for Classification (Indirectly Affecting Embeddings):** While Chapter 11 focuses on fine-tuning BERT for classification, this process inherently refines the intermediate representations and the final classification layer, which are built upon contextualized word embeddings. The goal is to adapt the pretrained BERT model to the specific nuances of the classification task, leading to better embeddings that capture the distinctions between classes.

*   **Fine-tuning for Named-Entity Recognition (NER):** Chapter 11 also discusses fine-tuning pretrained representation models like BERT for **Named-Entity Recognition**. This involves training the model to classify individual tokens within a text as belonging to specific entity categories. This fine-tuning process adapts the model's ability to generate contextualized embeddings that are sensitive to the boundaries and types of named entities.

*   **Parameter-Efficient Fine-Tuning (PEFT):** Chapter 12 touches upon **parameter-efficient fine-tuning (PEFT)** techniques like **adapters** and **LoRA (Low-Rank Adaptation)**, which can be used to fine-tune large language models, including those used for generating embeddings, by only updating a small subset of the model's parameters. This is particularly relevant when computational resources are limited. Specialized adapters for tasks like NER can even be downloaded and used.

In summary, the sources highlight that **fine-tuning is a powerful tool for tailoring embedding models to specific tasks, domains, and data characteristics**. Whether through contrastive learning, supervised training, or unsupervised domain adaptation techniques, fine-tuning refines the embedding space to better represent the relationships and semantic nuances relevant to the intended application, ultimately leading to improved performance in downstream tasks like classification, information retrieval, and named-entity recognition.

* **For dense retrieval**
* **Using contrastive learning**
* **For specific tasks**


* **For dense retrieval**

The sources highlight that **fine-tuning is a crucial technique for improving the performance of embedding models used in dense retrieval systems**. In the larger context of fine-tuning, the methods discussed for dense retrieval align with broader fine-tuning strategies covered in the sources, especially in **Chapter 10 ("Creating Text Embedding Models")** and **Chapter 12 ("Fine-Tuning Generation Models")**.

Here's a breakdown of what the sources say about fine-tuning in the context of dense retrieval:

*   **Dense Retrieval relies on embeddings:** Dense retrieval systems work by converting search queries and documents into numerical representations called **embeddings**. The system then retrieves documents whose embeddings are closest to the query embedding in vector space, based on the idea that semantically similar texts will have similar embeddings.

*   **Need for task-specific fine-tuning:** The sources point out that a general-purpose embedding model might not always capture the nuances required for effective retrieval. For instance, a model trained on general internet data might not perform optimally on a specific domain like legal texts. Therefore, **fine-tuning on data relevant to the retrieval task can significantly improve performance**.

*   **Fine-tuning with query-result pairs:** **Chapter 8** explicitly describes the process of **fine-tuning embedding models for dense retrieval**. This involves using **training data composed of queries and their corresponding relevant results**. The goal of fine-tuning is to **make the embeddings of relevant queries closer to the embeddings of the target documents** in the vector space. Conversely, the model also learns to push the embeddings of irrelevant queries farther away from the document embeddings.

*   **Contrastive learning in fine-tuning:** **Chapter 10** details **contrastive learning** as a major technique for training and fine-tuning text embedding models. This method aims to **bring embeddings of similar documents (positive pairs) closer together while pushing dissimilar documents (negative pairs) farther apart**. When fine-tuning for dense retrieval, relevant query-document pairs can be considered positive examples, and irrelevant query-document pairs as negative examples, allowing the model to learn a retrieval-specific embedding space.

*   **Supervised fine-tuning:** The method of using labeled query-result pairs to adjust the embedding space for better retrieval is a form of **supervised fine-tuning**, a broader technique discussed in **Chapter 10** and **Chapter 12**. Supervised fine-tuning uses labeled data to adapt a pretrained model to a specific downstream task, in this case, improving the relevance of retrieved documents for given queries.

*   **Augmented SBERT as a fine-tuning technique:** **Chapter 10** also presents **Augmented SBERT** as a fine-tuning method that can be used for pairwise sentence scoring tasks, which is relevant to assessing the relevance between a query and a document in retrieval. This technique leverages a small "gold" dataset to fine-tune a cross-encoder, which then helps in creating a larger "silver" dataset for further fine-tuning the bi-encoder used for generating embeddings in dense retrieval.

*   **TSDAE for domain adaptation:** **Chapter 10** introduces **TSDAE (Transformer-based Sequential Denoising Auto-Encoder)** as an unsupervised method for **fine-tuning embedding models for domain adaptation**. If a dense retrieval system needs to perform well in a specific domain with limited labeled query-result pairs, TSDAE can be used to pretrain or fine-tune the embedding model on unlabeled data from that domain, potentially improving its performance before or during supervised fine-tuning.

In the larger context of fine-tuning discussed in the sources, the fine-tuning of embedding models for dense retrieval exemplifies how pretrained models can be adapted for specific downstream tasks. This aligns with the general benefits of fine-tuning highlighted throughout the book, such as requiring less data and compute compared to training from scratch, and leading to improved performance on targeted applications. The specific techniques like contrastive learning and supervised training with task-relevant data are recurring themes in the sources when discussing how to adapt language models, including those that generate embeddings.

* **Using contrastive learning**

The sources emphasize that **contrastive learning is a significant and widely used technique for fine-tuning embedding models**. It plays a crucial role in adapting general-purpose pretrained models to perform better on specific tasks by refining the embedding space.

Here's what the sources say about using contrastive learning in the larger context of fine-tuning:

*   **Core Principle of Contrastive Learning:** Contrastive learning aims to train an embedding model such that **similar documents are closer in vector space, while dissimilar documents are further apart**. This is achieved by feeding the model examples of similar (positive) and dissimilar (negative) pairs of documents. This concept is similar to the word2vec method discussed in **Chapter 2**.

*   **Fine-tuning Embedding Models with Contrastive Learning:** **Chapter 10 ("Creating Text Embedding Models")** specifically highlights contrastive learning as a major technique for **fine-tuning text embedding models**. By presenting a pretrained embedding model with task-specific positive and negative pairs, the fine-tuning process adjusts the embeddings to better reflect the desired similarity or dissimilarity for that task.

*   **Generating Contrastive Examples for Fine-tuning:** To fine-tune with contrastive learning, it's essential to have data that constitutes similar/dissimilar pairs. The sources mention using **Natural Language Inference (NLI) datasets** to generate such examples. Entailment examples can serve as positive pairs, while contradiction examples can be used as negative pairs.

*   **Contrastive Loss Functions:** During fine-tuning with contrastive learning, specific loss functions are employed to optimize the embeddings. **Cosine similarity loss** is mentioned, which aims to minimize the cosine distance between embeddings of similar sentences and maximize the distance between dissimilar ones. **Multiple negatives ranking (MNR) loss** is another loss function discussed, where the model tries to identify the correct matching sentence from a set of negative examples.

*   **Augmented SBERT as a Fine-tuning Technique:** **Chapter 10** introduces **Augmented SBERT** as a fine-tuning method that leverages contrastive learning for pairwise sentence scoring tasks. This involves fine-tuning a cross-encoder on a small "gold" dataset and using it to label a larger "silver" dataset, which is then combined with the gold data to fine-tune the bi-encoder using contrastive learning principles.

*   **SetFit for Few-Shot Classification:** **Chapter 11 ("Fine-Tuning BERT for Classification")** discusses **SetFit**, an efficient fine-tuning method for few-shot classification that utilizes contrastive learning. SetFit generates positive and negative sentence pairs based on the class labels of the available data and then fine-tunes a SentenceTransformers model using contrastive learning on these generated pairs. The goal is to create embeddings tuned to the classification task, where documents belonging to the same class have closer embeddings.

*   **Word2vec as an Early Example:** The sources note that **word2vec**, discussed in **Chapters 1 and 2**, is an early and popular example of contrastive learning in NLP. It learns word representations by contrasting neighboring words (positive pairs) with randomly sampled non-neighboring words (negative pairs). This demonstrates that the core idea of contrastive learning has been foundational in developing effective embeddings, even before the rise of large language models.

In essence, the sources present contrastive learning as a powerful paradigm for fine-tuning embedding models. By explicitly teaching the model what constitutes similarity and dissimilarity through carefully constructed examples, contrastive learning enables the resulting embeddings to be highly effective for various downstream tasks, including semantic search (**Chapter 8**), text classification (**Chapter 4**, **Chapter 11**), and more generally, for creating high-quality text embedding models (**Chapter 10**) that can be used in a wide array of language AI applications (**Chapter 3**, **Chapter 10**). The fine-tuning process using contrastive learning adapts the pretrained model's understanding of language to the specific requirements of the target task or domain.


* **For specific tasks**

The sources extensively discuss the crucial role of **fine-tuning** in adapting pretrained language models for **specific tasks**. In the larger context of fine-tuning, the sources highlight several key aspects:

*   **Pretraining as the foundation:** Large language models (LLMs) typically undergo a pretraining phase on vast amounts of text data, allowing them to learn general grammar, context, and language patterns. This creates a foundation model. However, these base models generally do not follow specific instructions or perform well on particular tasks without further adaptation.

*   **Fine-tuning as task adaptation:** Fine-tuning, the second step in the LLM training paradigm, involves taking a pretrained model and further training it on a narrower, task-specific dataset. This process allows the LLM to adapt to specific tasks or exhibit desired behaviors, such as classification or instruction following. Fine-tuning is generally less compute-intensive and requires less data than pretraining.

*   **Examples of fine-tuning for specific tasks:** The sources provide numerous examples of how fine-tuning is applied to various NLP tasks:
    *   **Text Classification:** **Chapter 4** and **Chapter 11** detail fine-tuning for text classification. This can involve fine-tuning a representation model like BERT on a labeled classification dataset. The goal is to adjust the model's parameters so that it can accurately assign predefined labels to input text. The sources discuss fine-tuning task-specific models and also using general-purpose embedding models as feature extractors followed by a classifier.
    *   **Creating Text Embedding Models:** **Chapter 10** focuses on creating and fine-tuning embedding models. Fine-tuning existing embedding models on task-specific data, such as query-result pairs for dense retrieval, allows the model to generate embeddings that are more relevant for the target application. Contrastive learning is presented as a major technique for fine-tuning embedding models, where the model learns to bring embeddings of similar documents closer and dissimilar documents farther apart.
    *   **Dense Retrieval:** **Chapter 8** specifically discusses fine-tuning embedding models for dense retrieval. By training on query-relevant document pairs, the embedding model learns to produce embeddings where relevant queries and documents have high similarity scores.
    *   **Few-Shot Classification:** **Chapter 11** introduces SetFit, a method for efficiently fine-tuning embedding models for classification using only a small number of labeled examples. This demonstrates how fine-tuning can be adapted for low-resource scenarios.
    *   **Named-Entity Recognition (NER):** **Chapter 11** also covers fine-tuning a pretrained BERT model for NER, where the task is to classify individual tokens within a text. This requires specific data preprocessing to focus on the word-level classification.
    *   **Text Generation (Instruction Tuning and Preference Tuning):** **Chapter 12** is dedicated to fine-tuning generation models. It describes supervised fine-tuning (SFT) using instruction data (question-response pairs) to make the model follow instructions. Furthermore, it discusses preference tuning (alignment) using data that indicates preferred outputs, which further aligns the model with human expectations for helpfulness and safety. Parameter-efficient fine-tuning (PEFT) techniques like LoRA are also discussed as ways to fine-tune large generation models more efficiently.

*   **Benefits of fine-tuning:** The sources emphasize that fine-tuning allows for the adaptation of powerful pretrained models to specific needs with less data and computation than training from scratch. It also enables the creation of domain-specific models by fine-tuning on data from the target domain.

*   **Evaluation of fine-tuned models:** The sources touch upon the evaluation of models fine-tuned for specific tasks. For classification, metrics like F1 score are mentioned. For semantic search, mean average precision (MAP) is discussed. For generative models, various word-level metrics and LLM-as-a-judge methods are introduced.

In summary, the sources present fine-tuning as a versatile and essential technique for leveraging the general knowledge of pretrained language models and adapting them to perform effectively on a wide range of specific NLP tasks. It involves training the pretrained model on task-specific data, allowing it to learn the nuances and requirements of the target application. The choice of fine-tuning method and data depends on the specific task and available resources.

----

### Evaluation <a id="evaluation"></a>

The sources discuss the evaluation of embeddings in several contexts, primarily focusing on assessing their quality for various downstream tasks. Here's a comprehensive overview of what the sources say about evaluation in the larger context of embeddings:

*   **Semantic Similarity:** One fundamental aspect of evaluating embeddings, especially word embeddings like those produced by word2vec, is assessing their ability to capture semantic similarity. This can be done using **distance metrics** to judge how close one word is to another in the embedding space. The training process of word2vec itself is an implicit form of evaluation, where embeddings are updated based on the model's ability to predict whether two words are likely to be neighbors. If words that tend to have the same neighbors end up with closer embeddings, the method is considered effective.

*   **Downstream Task Performance:** The most crucial evaluation of embeddings often comes from their performance on specific downstream tasks. The sources provide several examples:
    *   **Classification:** In **text classification**, embeddings can be used as features for training a classifier. The evaluation here would involve standard classification metrics like **accuracy** and **F1 score**. The choice of embedding model can significantly impact classification performance.
    *   **Clustering:** For **text clustering**, the quality of embeddings is evaluated by how well they group semantically similar documents together. While the sources don't explicitly mention specific metrics for clustering embedding quality, the process of inspecting the resulting clusters and the modularity of tools like BERTopic allowing integration of different embedding techniques suggest that the interpretability and coherence of clusters serve as a form of evaluation. Choosing embedding models optimized for semantic similarity is highlighted as important for clustering.
    *   **Semantic Search and Retrieval:** Evaluating embeddings for **dense retrieval** in semantic search involves using metrics from the Information Retrieval (IR) field, such as **mean average precision (MAP)**. This requires a test suite consisting of queries and relevance judgments indicating which documents in the archive are relevant to each query. The goal of fine-tuning embedding models for dense retrieval is to improve these evaluation metrics by making the embeddings of relevant queries and documents closer in vector space.
    *   **Recommendation Systems:** In **recommendation systems**, like the music recommender example using a word2vec-like approach to embed songs, the evaluation might implicitly involve user satisfaction or engagement with the recommendations, though specific metrics aren't detailed in the provided excerpts.

*   **Benchmarking Embedding Models:** To systematically evaluate and compare different embedding models, the **Massive Text Embedding Benchmark (MTEB)** was developed. This benchmark spans a variety of embedding tasks across multiple datasets and languages, providing a leaderboard to compare state-of-the-art models. The MTEB allows for evaluating not only accuracy but also inference speed, which is crucial for many real-world applications.

*   **Specific Evaluation Tools and Techniques:** The `sentence-transformers` library provides tools like `EmbeddingSimilarityEvaluator` for evaluating the quality of sentence embeddings by comparing them against human-annotated similarity scores (e.g., on the STSB benchmark). This evaluator calculates metrics like **Pearson correlation** and **Spearman correlation** between the cosine similarity of the embeddings and the ground truth scores.

*   **Importance of Task-Specific Fine-tuning and Evaluation:** The sources emphasize that the "accuracy" of an embedding model is often defined by its effectiveness for a particular purpose. Fine-tuning an embedding model using contrastive learning on task-specific data, such as NLI datasets, aims to align the embedding space with the desired notion of similarity for that task. Therefore, the evaluation should ideally reflect the model's performance on the target task.

In summary, the evaluation of embeddings is a multifaceted process. It involves assessing their ability to capture semantic relationships, their performance on specific downstream tasks like classification, clustering, and retrieval, and their standing on comprehensive benchmarks like MTEB. The choice of evaluation metrics and methods depends heavily on the intended application of the embeddings, and fine-tuning is often performed to optimize embeddings for better evaluation results on those specific tasks.

* **Similarity measures**
* **Benchmarks**

* **Similarity measures**

The sources highlight the importance of **similarity measures** in the **evaluation** of embeddings across various tasks. These measures, such as **cosine similarity, Manhattan distance, and Euclidean distance**, play a crucial role in quantifying how well embeddings capture the semantic relationships between different pieces of text or different modalities.

Here's how the sources discuss these similarity measures in the context of evaluation:

*   **Semantic Similarity of Word Embeddings:** The sources explicitly state that **embeddings** are tremendously helpful as they allow us to measure the **semantic similarity** between two words. Various **distance metrics** can be used to judge how close one word is to another in the embedding space. Figure 1-9 illustrates that words with similar meanings tend to be closer in dimensional space, and this proximity is quantified using these similarity measures. While the specific metrics aren't named in this early discussion of word embeddings, the concept of distance implies the use of measures like cosine, Manhattan, or Euclidean.

*   **Evaluation of Sentence Embeddings:** When discussing the creation and evaluation of **text embedding models** in **Chapter 10**, the sources provide concrete examples of using similarity measures. The `EmbeddingSimilarityEvaluator` from the `sentence-transformers` library is used to evaluate how well a trained model captures semantic similarity based on human-labeled scores in the Semantic Textual Similarity Benchmark (STSB). This evaluator calculates several metrics, including **Pearson correlation** and **Spearman correlation** between the cosine similarity (and also Manhattan and Euclidean distances) of the generated embeddings and the ground truth similarity scores. The results show Pearson cosine, Pearson Manhattan, and Pearson Euclidean scores, demonstrating how these different measures are used to assess the quality of the embeddings in capturing semantic similarity. A higher correlation indicates that the embedding model's similarity scores, based on these distance metrics, align better with human judgments.

*   **Similarity in Contrastive Learning:** **Contrastive learning**, a major technique for training embedding models, relies on the idea that similar documents should be closer in vector space and dissimilar documents further apart. The sources mention that after generating positive and negative pairs, their embeddings are calculated, and **cosine similarity** is applied to determine if the pairs are negative or positive. The model is then optimized based on these similarity scores.

*   **Multimodal Embedding Evaluation:** In the context of **multimodal embeddings**, specifically CLIP (Contrastive Language-Image Pre-training), **cosine similarity** is used to compare the embeddings of images and texts. During training, the goal is to maximize the cosine similarity between embeddings of similar image/caption pairs and minimize it for dissimilar pairs. The sources also show an example of calculating the **similarity score** between a text embedding and an image embedding by normalizing the embeddings and then calculating their dot product, which is equivalent to cosine similarity for normalized vectors. This demonstrates how cosine similarity is used to evaluate the alignment of embeddings across different modalities.

*   **Semantic Search and Dense Retrieval:** **Dense retrieval** systems rely on the concept of embeddings and retrieve the nearest neighbors of a search query based on the similarity of their embeddings. The sources explain that embeddings can be thought of as points in space, and texts with similar meanings are close to each other. When a search query is embedded, the system finds the nearest documents in that embedding space, implying the use of distance or similarity measures (like cosine, Euclidean, or dot product which relates to cosine for normalized vectors) to determine which documents are the most relevant.

*   **Classification with Embeddings:** For **text classification**, embeddings of documents can be compared to embeddings of labels using **cosine similarity** to determine the most relevant label. The label with the highest cosine similarity to the document's embedding is chosen as the predicted class. This highlights the use of cosine similarity as an evaluation mechanism for how well embeddings can differentiate between different categories.

*   **Benchmarking Embedding Models:** The **Massive Text Embedding Benchmark (MTEB)** is mentioned as a great place to start when selecting models to generate embeddings. While the sources don't explicitly list all the similarity measures used in MTEB, the fact that it benchmarks models across several tasks implies that various evaluation metrics, including those based on cosine, Manhattan, and Euclidean distances, would be used depending on the specific task (e.g., semantic similarity tasks likely use these measures).

In summary, the sources illustrate that **cosine similarity, Manhattan distance, and Euclidean distance (and related measures like dot product)** are fundamental tools for **evaluating the quality and utility of embeddings** in various natural language processing tasks. They are used to quantify semantic similarity between words and sentences, assess the alignment of multimodal embeddings, evaluate the performance of embeddings in classification and retrieval tasks, and benchmark different embedding models against each other. The choice of which similarity measure to use can depend on the specific task and the desired properties of the embedding space.

* **Benchmarks**

These sources discuss **benchmarks**, specifically the **Massive Text Embedding Benchmark (MTEB)** and the **General Language Understanding Evaluation (GLUE) benchmark**, as crucial tools for the **evaluation** of language models, particularly embedding models and generative models.

Here's a breakdown of what the sources say about these benchmarks in the larger context of evaluation:

*   **MTEB for Evaluating Embedding Models:**
    *   The **MTEB** is highlighted as a "**great place to start**" when selecting models to generate embeddings. It serves as a comprehensive benchmark spanning **eight embedding tasks** that cover **58 datasets** and **112 languages**.
    *   The MTEB was developed to **unify the evaluation procedure** for embedding models.
    *   It provides a **leaderboard** that allows for the public comparison of state-of-the-art embedding models across various tasks.
    *   Evaluation on the MTEB considers not only **accuracy** but also **inference speed**, which is important for real-world applications like semantic search.
    *   The MTEB includes diverse tasks, allowing for a more thorough assessment of an embedding model's capabilities. Examples of tasks on MTEB are not explicitly listed, but the mention of "clustering tasks" suggests it includes unsupervised evaluation as well.
    *   While testing on the entire MTEB can be time-consuming, the source uses the Semantic Textual Similarity Benchmark (STSB) for illustrative purposes in training an embedding model.

*   **GLUE for Evaluating Language Understanding:**
    *   The **GLUE benchmark** consists of **nine language understanding tasks** designed to evaluate and analyze model performance.
    *   It covers a **wide degree of difficulty** in language understanding.
    *   The source mentions using data derived from the **Multi-Genre Natural Language Inference (MNLI) corpus**, one of the tasks within the GLUE benchmark, to generate contrastive examples for training and fine-tuning embedding models. This demonstrates how GLUE datasets can be leveraged for training as well as evaluation.
    *   GLUE is also mentioned as a benchmark for evaluating **generative models** on language generation and understanding tasks.

*   **Benchmarks in General for Evaluating Generative Models:**
    *   Benchmarks like **MMLU, GLUE, TruthfulQA, GSM8k, and HellaSwag** are common methods for evaluating **generative models** on language generation and understanding tasks. These benchmarks provide insights into a model's basic language understanding and complex analytical abilities, such as solving math problems.
    *   Specialized models, like those for programming, are evaluated on different benchmarks such as **HumanEval**.
    *   Benchmarks provide a basic understanding of how well a model performs across a variety of tasks.
    *   However, a downside of public benchmarks is the potential for **overfitting**, where models are optimized specifically for these benchmarks, potentially at the expense of other useful capabilities.
    *   Benchmarks might not cover very **specific use cases**, and some require significant computational resources, making iteration difficult.

*   **Leaderboards as Aggregations of Benchmarks:**
    *   Due to the multitude of benchmarks, **leaderboards** like the **Open LLM Leaderboard** have been developed, which contain results across multiple benchmarks (e.g., HellaSwag, MMLU, TruthfulQA, GSM8k).
    *   Models that perform well on these leaderboards are generally considered the "**best**" models, assuming no overfitting. However, the risk of overfitting on leaderboard benchmarks still exists.

In summary, these sources emphasize that **MTEB is a key benchmark specifically designed for evaluating the quality and performance of text embedding models across a diverse set of tasks and languages, considering both accuracy and efficiency**. **GLUE is presented as a benchmark focused on evaluating the language understanding capabilities of various language models, including its utility in generating training data for embedding models**. More broadly, benchmarks like GLUE and MTEB, along with others, serve as **important but not perfect tools for evaluating language models**, providing a standardized way to assess and compare different models. However, the sources also caution against relying solely on benchmark scores due to the potential for overfitting and the fact that they might not fully align with specific real-world applications.

---


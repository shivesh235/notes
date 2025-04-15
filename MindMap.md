# Large Language Models

* **Language Models**

    * [**Tokenization**](Tokenization.md#tokenization)
        * [Definition](Tokenization.md#definition)
        * [Methods](Tokenization.md#methods)
        * [Token Types](Tokenization.md#token-types)
        * [Special Tokens](Tokenization.md#special-tokens)
        * [Impact on model](Tokenization.md#impact-on-model)
        * [Examples](Tokenization.md#examples)

    * [**Embeddings**](Embeddings.md#embeddings)
        * [Definintion](Embeddings.md#definition)
        * [Types](Embeddings.md#types)
        * [Creation](Embeddings.md#creation)
        * [Use Cases](Embeddings.md#use-cases)
        * [Fine-tuning](Embeddings.md#fine-tuning)
        * [Evaluation](Embeddings.md#evaluation)

    * **Language Models**
        * Generative Models
        * Representation Models
        * Encoder-Decoder Models
        * Training
        * Attention Mechanism
        * Quantization
        * Evaluatioin
        * Tet Generation Techiques

    * **Key Concepts**
        * Vocabulary
        * Context
        * Parameters
        * Vector Representations
        * Transformers architecture

    * **Tools and Libraries**
        * Hugging Face Transformers
        * Gensim
        * BERTopic
        * rank_bm25
        * Annoy/FAISS
        * Weaviate/Pinecone
        * llama-cpp-python
        * matplotlib

---

### Briefing Document: Hands-On Large Language Models - Key Concepts

**Source:** Excerpts from "Hands-On_Large_Language_Models_-_Jay_Alammar.pdf"
**Date:** October 26, 2023
**Prepared For:** Interested Parties
**Prepared By:** Gemini
**Subject:** Review of Fundamental Concepts in Large Language Models (LLMs)

This briefing document summarizes key themes and important ideas from the provided excerpts of "Hands-On Large Language Models." The focus is on foundational concepts necessary for understanding how LLMs process and generate text, including tokenization, embeddings, attention mechanisms, and different model architectures.

### 1. Tokenization: Breaking Down Text

**Main Theme:** Tokenization is the initial crucial step in processing text for LLMs, converting raw text into numerical representations (tokens) that the model can understand. Different tokenization methods exist, each with its own advantages and disadvantages.

**Key Ideas and Facts:**

* **Whitespace Splitting:** The most common initial approach is splitting text by whitespace to create individual words. However, this is not universally effective across all languages (e.g., Mandarin).
    * "The most common method for tokenization is by splitting on a whitespace to create individual words. However, this has its disadvantages as some languages, like Mandarin, do not have whitespaces around individual words."
* **Vocabulary Creation:** After tokenization, a vocabulary is created by compiling all unique tokens from the input text.
    * "after tokenization, we combine all unique words from each sentence to create a vocabulary that we can use to represent the sentences."
* **Bag-of-Words:** A simple method to represent text numerically using the vocabulary is the "bag-of-words" model, which counts the occurrences of each word in a sentence, creating a vector representation.
    * "Using our vocabulary, we simply count how often a word in each sentence appears, quite literally creating a bag of words. As a result, a bag-of-words model aims to create representations of text in the form of numbers, also called vectors or vector representations..."
* **Subword Tokenization:** Modern LLMs often use subword tokenization methods like Byte Pair Encoding (BPE) (used by GPT models) and WordPiece (used by BERT). These methods aim to find an optimal set of tokens to represent the training data efficiently.
    * "Popular methods include byte pair encoding (BPE) (widely used by GPT models) and WordPiece (used by BERT). These methods are similar in that they aim to optimize an efficient set of tokens to represent a text dataset, but they arrive at it in different ways."
* **Special Tokens:** Tokenizers include special tokens that serve specific purposes, such as:
    * Beginning of text (```<s>```, [CLS])
    * End of text (<|endoftext|>, [SEP])
    * Unknown token ([UNK], <unk>)
    * Padding token ([PAD], <pad>)
    * Masking token ([MASK])
    * File/Repository tokens (<filename>, <reponame>)
* **Whitespace Handling:** Different tokenizers handle whitespace (including spaces and tabs) differently. Some, like GPT-4 and StarCoder2, have specific tokens for sequences of whitespace, which can be beneficial for understanding code structure.
    * "The GPT-4 tokenizer represents the four spaces as a single token. In fact, it has a specific token for every sequence of whitespaces up to a list of 83 whitespaces."
    * "Similar to GPT-4, it encodes the list of whitespaces as a single token." (referring to StarCoder2)
* **Tokenization-Free Encoding:** Some recent approaches explore "tokenization-free encoding" that operates directly on bytes, potentially being more effective in multilingual scenarios.
    * "like “CANINE: Pre-training an efficient tokenization-free encoder for language representation” outline methods like this, which are also called “tokenization-free encoding.” Other works like “ByT5: Towards a token-free future with pre-trained byte-to-byte models” show that this can be a competitive method, especially in multilingual scenarios."
* **Impact on Model Performance:** Tokenization choices significantly influence a language model's ability to understand and generate text effectively, especially for specific domains like code.
    * "This is an example of where tokenization choices can help the model improve on a certain task." (regarding whitespace handling in code)

### 2. Embeddings: Representing Meaning as Vectors

**Main Theme:** Embeddings are dense vector representations of tokens (and sometimes larger text units) that capture semantic meaning. These vectors allow the model to understand relationships between words and concepts.

**Key Ideas and Facts:**

* **Static vs. Contextual Embeddings:** Early methods like word2vec created static embeddings, where a word had the same embedding regardless of context. Modern LLMs, often based on Transformer architectures, generate contextual embeddings where the representation of a word changes based on its surrounding words.
    * "The training process of word2vec creates static, downloadable representations of words. For instance, the word “bank” will always have the same embedding regardless of the context in which it is used. However, “bank” can refer to both a financial bank as well as the bank of a river. Its meaning, and therefore its embeddings, should change depending on the context."
* **Language Model Vocabulary and Embeddings:** A trained language model holds an embedding vector for each token in its tokenizer's vocabulary. This embedding matrix is a crucial part of the model.
    * "The language model holds an embedding vector for each token in the tokenizer’s vocabulary... When we download a pretrained language model, a portion of the model is this embeddings matrix holding all of these vectors."
* **Generating Token Embeddings:** Models like DeBERTa v3 are designed to produce high-quality token embeddings. The output of such models for a given input is a tensor containing the embedding vector for each token.
    * "The model we’re using here is called DeBERTa v3, which at the time of writing is one of the best-performing language models for token embeddings while being small and highly efficient."
* **Word2vec and Contrastive Training:** The word2vec algorithm learns word embeddings by training a model to predict whether two words are likely to appear in similar contexts (neighbors). This involves contrastive learning, where the model learns to differentiate between related and unrelated word pairs.
    * "A model is then trained on each example to take in two embedding vectors and predict if they’re related or not... It updates the embeddings in the training process to produce the final, trained embeddings."
* **Applications of Embeddings:** Beyond language modeling, embeddings can be used for various tasks like:
    * Semantic Search
    * Recommendation Systems
    * Clustering and Topic Modeling (BERTopic)

### 3. Attention Mechanism: Encoding and Decoding Context

**Main Theme:** The attention mechanism is a core component of Transformer-based LLMs that allows the model to weigh the importance of different parts of the input sequence when processing information. This enables the model to understand long-range dependencies and context.

**Key Ideas and Facts:**

* **Recurrent Neural Networks (RNNs) as a Precursor:** While not the focus of modern LLMs, RNNs were an earlier approach to model sequences.
    * "A step in encoding this text was achieved through recurrent neural networks (RNNs). These are variants of neural networks that can model sequences as..."
* **Parallel Attention Heads:** Transformer architectures utilize multi-head attention, where multiple attention mechanisms run in parallel, allowing the model to attend to different aspects of the input simultaneously.
    * "Figure 3-17 shows the intuition of how attention heads run in parallel with a preceding step of splitting information and a later step of combining the results of all the heads... We get better LLMs by doing attention multiple times in parallel, increasing the model’s capacity to attend to different types of information."
* **Queries, Keys, and Values:** Attention is calculated using three sets of weight matrices (queries, keys, and values) applied to the input embeddings.
    * "Figure 3-26. Attention is conducted using matrices of queries, keys, and values. In multi-head attention, each head has a distinct version of each of these matrices."
* **Optimizations:** Multi-Query and Grouped-Query Attention: To improve efficiency, especially for large models, variations like multi-query attention and grouped-query attention have been developed.
    * "Multi-query attention presents a more efficient attention mechanism by sharing the keys and values matrices across all the attention heads... Instead of cutting the number of keys and values matrices to one of each, it allows us to use more (but less than the number of heads). Figure 3-28 shows these groups and how each group of attention heads shares keys and values matrices."

### 4. Generative Models and Initial Code

**Main Theme:** The book introduces generative models early on, using Phi-3-mini as a primary example. These models are designed to generate new text.

**Key Ideas and Facts:**

* **Phi-3-mini:** A relatively small (3.8 billion parameters) but performant generative model.
    * "The main generative model we use throughout the book is Phi-3-mini, which is a relatively small (3.8 billion parameters) but quite performant model."
* **Resource Efficiency:** Phi-3-mini's small size allows it to run on devices with limited VRAM.
    * "Due to its small size, the model can be run on devices with less than 8 GB of VRAM. If you perform quantization... you can use even less than 6 GB of VRAM."
* **Commercial Use:** Phi-3-mini is licensed under the MIT license, permitting its use for commercial purposes without restrictions.
    * "Moreover, the model is licensed under the MIT license, which allows the model to be used for commercial purposes without constraints!"

### 5. Practical Applications and Techniques

**Main Theme:** The excerpts touch upon various practical applications and techniques related to LLMs and text understanding.

**Key Ideas and Facts:**

* Sentiment Classification
* Text Clustering and Topic Modeling (BERTopic)
* Constrained Generation
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Fine-tuning
* Named-Entity Recognition (NER)
* Model Quantization

### 6. Evaluation and Benchmarking

**Main Theme:** Evaluating the performance of LLMs and embedding models is crucial. Various benchmarks and metrics are used to assess different aspects of their capabilities.

**Key Ideas and Facts:**

* Embedding Evaluation (cosine similarity, Pearson correlation, Spearman correlation, MTEB)
* Classification Evaluation (accuracy, F1-score)
* Token Classification Evaluation (seqeval)

### Conclusion

The provided excerpts from "Hands-On Large Language Models" offer a valuable introduction to the fundamental concepts underlying modern LLMs. Key takeaways include the importance of tokenization in preparing text for processing, the role of embeddings in capturing semantic meaning, the power of the attention mechanism in understanding context, and the versatility of these models for various natural language processing tasks. The book emphasizes practical application through code examples and introduces important techniques like fine-tuning, retrieval augmentation, and model quantization. Understanding these core concepts is essential for anyone looking to work with and build upon the capabilities of large language models.

---

### Timeline of Main Events

This timeline focuses on the key concepts and models discussed in the provided excerpts from "Hands-On Large Language Models." It highlights the evolution and techniques related to tokenization, embeddings, and language models.

**Early Approaches (Pre-2018):**

* Whitespace Tokenization: The most common initial method for breaking text into tokens by splitting on whitespace.
* Bag-of-Words Model: Representing text as numerical vectors based on the frequency of words in a vocabulary.
* Word2vec Algorithm: Creates static word embeddings where each word has a fixed representation regardless of context. Trained using contrastive learning to predict relationships between words.
* Recurrent Neural Networks (RNNs): An early neural network architecture capable of modeling sequences.

**2018:**

* BERT (Bidirectional Encoder Representations from Transformers): Introduced the WordPiece tokenization method (Japanese and Korean voice search).
* BERT base model (uncased): Lowercases all text before tokenization. Vocabulary size of 30,522. Uses special tokens: [UNK], [SEP], [PAD], [CLS], [MASK].
* BERT base model (cased): Preserves capitalization during tokenization. Vocabulary size of 28,996. Uses the same special tokens as the uncased version.
* Tokenization-Free Encoding: Research begins exploring methods that don't rely on explicit tokenization.

**2019:**

* GPT-2 (Generative Pre-trained Transformer 2): Introduced Byte Pair Encoding (BPE) for tokenization ("Neural machine translation of rare words with subword units"). Vocabulary size of 50,257. Uses <|endoftext|> as a special token. Preserves capitalization and handles whitespace and special characters differently than BERT.
* DistilBERT: A smaller, faster, cheaper, and lighter distilled version of BERT.
* ALBERT: A Lite BERT for self-supervised learning of language representations.
* Sentence-BERT (SBERT): Utilizes Siamese BERT networks to generate sentence embeddings.

**2020:**

* DeBERTa (Decoding-enhanced BERT with disentangled attention): Improves upon BERT with disentangled attention mechanisms.

**2022:**

* Flan-T5: Employs the SentencePiece tokenizer, supporting BPE and the unigram language model ("SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing"; "Subword regularization: Improving neural network translation models with multiple subword candidates"). Vocabulary size of 32,100. Uses <unk> and <pad> as special tokens.

**2023:**

* GPT-4: An advancement over GPT-2, with a tokenizer that handles whitespace more efficiently (single tokens for sequences of spaces) and includes tokens specific to code (e.g., elif).

**2024 (Implied/Reference):**

* StarCoder2: An encoder focused on code generation. Uses BPE tokenization (vocabulary size 49,152) and special tokens for repository name (<reponame>) and filename (<filename>), GitHub stars (<gh_stars>), and fill-in-the-middle tasks (<fim_prefix>, <fim_middle>, <fim_suffix>, <fim_pad>). Tokenizes digits individually.
* Galactica: A language model for science, using BPE tokenization (vocabulary size 50,000) with special tokens for citations ([START_REF], [END_REF]), reasoning (<work>), mathematics, amino acid sequences, and DNA sequences. Handles whitespace and tabs as single tokens for sequences.
* Phi-3: Reuses the Llama 2 tokenizer with additional special tokens. A smaller (3.8 billion parameters) but performant generative model.

**Ongoing Concepts and Techniques:**

* Token Embeddings: Vector representations of tokens learned by language models. Contextual embeddings (where the embedding of a word changes based on its usage) address limitations of static embeddings like word2vec.
* Attention Mechanism: Allows models to weigh the importance of different parts of the input sequence when processing information.
* Multi-Head Attention: Running multiple attention mechanisms in parallel to capture different types of information.
* Multi-Query Attention and Grouped-Query Attention: More efficient attention mechanisms that share key and value matrices across attention heads to reduce memory usage and increase speed.
* Fine-Tuning: Adapting pre-trained language models to specific downstream tasks.
* Text Clustering and Topic Modeling (e.g., BERTopic): Using embeddings and clustering algorithms to group similar documents and extract topics.
* Semantic Search and Retrieval-Augmented Generation (RAG): Leveraging embeddings for information retrieval and integrating retrieved information into language model generation.
* Model Quantization: Techniques to reduce the memory footprint and increase the speed of language models by lowering the precision of their parameters.
* Instruction Tuning: Training models on datasets of instructions and desired outputs to improve their ability to follow instructions.
* Preference Tuning (e.g., DPO): Training models based on human preferences between different generated outputs.
* Benchmarks (e.g., GLUE, MMLU, TruthfulQA): Standardized datasets and metrics for evaluating the performance of language models on various tasks.

#### Cast of Characters

This cast includes the principal researchers, teams, or entities associated with the key models, techniques, and concepts discussed in the provided excerpts.

* Jay Alammar: The author of "Hands-On Large Language Models," who explains the concepts and techniques discussed in the excerpts.
* BERT (Researchers/Team): The creators of the BERT family of language models, known for introducing the Transformer architecture widely and the WordPiece tokenization. The paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" is foundational.
* GPT-2 (OpenAI): The developers of GPT-2, which popularized large-scale generative language models and the Byte Pair Encoding (BPE) tokenization in this context. The paper "Language Models are Unsupervised Multitask Learners" is significant.
* Flan-T5 (Researchers/Google): The creators of Flan-T5, highlighting the effectiveness of instruction tuning and the SentencePiece tokenizer. The paper "Scaling Instruction-Finetuned Language Models" details this work.
* GPT-4 (OpenAI): The developers of GPT-4, a more advanced large language model with improvements in tokenization and capabilities. While a specific paper is not directly referenced, it represents a significant step in LLM development.
* StarCoder2 (Stability AI): The team behind StarCoder2, emphasizing code generation capabilities and specialized tokenization for code-related tasks.
* Galactica (Meta AI): The developers of Galactica, a model focused on scientific knowledge and featuring tokenization tailored for scientific content. The paper "Galactica: A Large Language Model for Science" describes this model.
* Phi-3 (Microsoft): The creators of the Phi-3 series, highlighting smaller yet performant generative models and the reuse of the Llama 2 tokenizer with additions.
* Word2vec (Tomas Mikolov et al.): The researchers who developed the word2vec algorithm, a seminal work in creating word embeddings. The paper "Efficient Estimation of Word Representations in Vector Space" is key.
* Byte Pair Encoding (BPE) (Researchers): The original inventors of the Byte Pair Encoding algorithm, adapted for subword tokenization in NLP. The paper "Neural machine translation of rare words with subword units" by Sennrich et al. is a key reference in this context.
* WordPiece (Researchers/Google): The team who introduced the WordPiece tokenization method, particularly associated with BERT and described in the context of Japanese and Korean voice search.
* SentencePiece (Taku Kudo and John Richardson): The developers of the SentencePiece tokenizer, offering flexibility and language independence with BPE and unigram models. Their paper "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing" is important.
* DeBERTa (Pengcheng He et al.): The researchers behind DeBERTa, focusing on improvements to the BERT architecture through disentangled attention. Their paper "DeBERTa: Decoding-enhanced BERT with disentangled attention" details their contributions.
* DistilBERT (Victor Sanh et al.): The authors of the DistilBERT paper, showcasing the effectiveness of knowledge distillation for creating smaller and faster language models.
* ALBERT (Zhenzhong Lan et al.): The researchers who developed ALBERT, focusing on parameter reduction techniques for more efficient language model pre-training.
* Sentence-BERT (Nils Reimers and Iryna Gurevych): The creators of Sentence-BERT, which provides effective sentence embeddings using Siamese networks.
* BERTopic (Maarten Grootendorst): The developer of the BERTopic framework for modular topic modeling using embeddings and a class-based TF-IDF approach.

---

### Transformer Architecture <a id="transformer-architecture"></a>
The core architecture behind modern LLMs, enabling parallel processing of sequence data.

#### Attention Mechanisms <a id="attention-mechanisms"></a>
Allows the model to focus on relevant parts of the input sequence.